r"""Temporary train script.

To run locally:
python -m scripts.trainer Trainer \
  --model_class=Transformer --batch_size=1 \
  --nomultidevice --max_tokens=25 --max_steps=20 \
  - run_train --dataset_path=data/codenet/f=0.01
"""

import dataclasses
import itertools
from typing import Any, List, Optional, Text

from absl import logging
from flax.metrics import tensorboard
from flax.training import checkpoints
from flax.training import common_utils
from flax.training import early_stopping
from flax.training import train_state
import jax
import jax.numpy as jnp
import ml_collections
import optax
import tensorflow_datasets as tfds

from core.data import codenet_paths
from core.data import data_io
from core.data import error_kinds
from core.lib import evaluator
from core.lib import misc_utils
from core.lib import models
from core.lib import optimizer_lib


DEFAULT_DATASET_PATH = codenet_paths.DEFAULT_DATASET_PATH
NUM_CLASSES = error_kinds.NUM_CLASSES

Config = ml_collections.ConfigDict


class TrainState(train_state.TrainState):
  rng: Any


@dataclasses.dataclass
class Trainer:

  config: Config

  def load_dataset(
    self, dataset_path=DEFAULT_DATASET_PATH, split='train',
  ):
    config = self.config
    batch_size = config.batch_size
    epochs = config.epochs
    allowlist = config.allowlist

    padded_shapes = data_io.get_padded_shapes(
        config.max_tokens, config.max_num_nodes, config.max_num_edges)
    if allowlist == 'TIER1_ERROR_IDS':
      allowlist = error_kinds.TIER1_ERROR_IDS
    filter_fn = data_io.make_filter(
        config.max_tokens, config.max_num_nodes, config.max_num_edges,
        config.max_steps, allowlist=allowlist, class_subsample_values={1: 0.0672})

    if split.endswith('-batch'):
      # Prepare a dataset with a single repeating batch.
      split = split[:-len('-batch')]
      return (
          data_io.load_dataset(dataset_path, split=split)
          .filter(filter_fn)
          .take(batch_size)
          .repeat(epochs)
          .padded_batch(batch_size, padded_shapes=padded_shapes)
      )

    # Return the requested dataset.
    return (
        data_io.load_dataset(dataset_path, split=split)
        .filter(filter_fn)
        .repeat(epochs)
        .shuffle(1000)
        .padded_batch(batch_size, padded_shapes=padded_shapes)
    )

  def make_model(self):
    return models.make_model(self.config)

  def create_train_state(self, rng, model):
    """Creates initial TrainState."""
    config = self.config
    fake_input = data_io.get_fake_input(
        config.batch_size, config.max_tokens, config.max_num_nodes, config.max_num_edges)
    rng, params_rng, dropout_rng = jax.random.split(rng, 3)
    variables = model.init(
        {'params': params_rng, 'dropout': dropout_rng},
        fake_input)
    params = variables['params']
    learning_rate = 0.03
    tx = optax.sgd(learning_rate)
    return TrainState.create(
        apply_fn=model.apply, params=params, tx=tx, rng=rng)

  def make_train_step(self):
    @jax.jit
    def train_step(state, batch):
      """The on-device part of a train step."""
      model = self.make_model()

      new_rng, dropout_rng = jax.random.split(state.rng, 2)
      state = dataclasses.replace(state, rng=new_rng)

      def loss_fn(params):
        logits = model.apply(
            {'params': params},
            batch,
            rngs={'dropout': dropout_rng}
        )
        assert len(logits.shape) == 2
        # logits.shape: batch_size, NUM_CLASSES
        labels = jax.nn.one_hot(jnp.squeeze(batch['target'], axis=-1), NUM_CLASSES)
        assert len(labels.shape) == 2
        # labels.shape: batch_size, NUM_CLASSES
        losses = optax.softmax_cross_entropy(
            logits=logits,
            labels=labels)
        assert len(losses.shape) == 1
        # losses.shape: batch_size
        loss = jnp.mean(losses)
        assert len(loss.shape) == 0
        return loss, {
            'logits': logits,
        }

      grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
      (loss, aux), grads = grad_fn(state.params)
      if self.config.multidevice:
        grads = jax.lax.pmean(grads, 'batch')
      # grads = optimizer_lib.clip_grad(grads, clip_by='global_norm', clip_value=1.0)
      state = state.apply_gradients(grads=grads)
      # TODO(dbieber): Optionally compute on-device metrics here.
      return state, {
          'logits': aux['logits'],
          'loss': loss,
      }
    if self.config.multidevice:
      train_step = jax.pmap(
          train_step,
          axis_name='batch',
          in_axes=(None, 0),
          out_axes=(None, 0),
      )
    return train_step

  def run_train(self, dataset_path=DEFAULT_DATASET_PATH, split='train', steps=None):
    config = self.config
    print(f'Training on data: {dataset_path}')
    dataset = self.load_dataset(dataset_path, split=split)
    eval_dataset = self.load_dataset(dataset_path, split='valid')

    rng = jax.random.PRNGKey(0)
    exp_id = codenet_paths.make_experiment_id()
    checkpoint_dir = codenet_paths.make_checkpoints_path(exp_id)
    log_dir = codenet_paths.make_log_dir(exp_id)
    print(f'Checkpoints: {checkpoint_dir}')

    rng, init_rng = jax.random.split(rng)
    model = self.make_model()

    state = self.create_train_state(init_rng, model)
    if config.restore_checkpoint_dir is not None:
      state = checkpoints.restore_checkpoint(config.restore_checkpoint_dir, state)
    train_step = self.make_train_step()

    # TODO(rishab): Store the state of the early stopping.
    es = early_stopping.EarlyStopping(
        min_delta=config.early_stopping_delta,
        patience=config.early_stopping_threshold,
    )
    summary_writer = tensorboard.SummaryWriter(log_dir)
    summary_writer.hparams(config.to_dict())

    recent_accuracies = []
    for step, batch in itertools.islice(enumerate(tfds.as_numpy(dataset)), steps):
      if config.multidevice:
        batch = common_utils.shard(batch)
      state, aux = train_step(state, batch)
      predictions = jnp.squeeze(jnp.argmax(aux['logits'], axis=-1))
      targets = jnp.squeeze(batch['target'])
      batch_accuracy = jnp.sum(predictions == targets) / jnp.sum(jnp.ones_like(targets))
      recent_accuracies.append(batch_accuracy)
      recent_accuracies = recent_accuracies[-500:]
      print(f"""--- Step {step}
Loss: {aux['loss']}
Predictions:
{predictions}
Targets:
{targets}
Batch Accuracy: {100 * batch_accuracy:02.1f}
Recent Accuracy: {100 * jnp.mean(jnp.array(recent_accuracies)):02.1f}""")

      if step % config.save_freq == 0:
        checkpoints.save_checkpoint(checkpoint_dir, state, step, keep=3)

      if step % config.eval_freq == 0:
        if eval_dataset is None:
          logging.info('Validation dataset unspecified. Skipping evaluation.')
          eval_loss = None
        else:
          eval_loss, eval_classification_score = evaluator.evaluate(
              eval_dataset, state, config
          )
        logging.info(
            f'Validation loss: {eval_loss}\n '
            f'Validation {config.eval_metric}: {eval_classification_score}'
        )
        (
            _,
            batch_loss,
            batch_classification_score,
        ) = evaluator.evaluate_batch(batch, state, config)
        summary_writer.scalar('train_loss', batch_loss, step)
        summary_writer.scalar('train_metric', batch_classification_score, step)
        summary_writer.scalar('eval_loss', eval_loss, step)
        summary_writer.scalar('eval_metric', eval_classification_score, step)

        if eval_loss is None:
          eval_loss = batch_loss
        did_improve, es = es.update(-1 * eval_loss)
        if es.should_stop:
          logging.info('Early stopping triggered.')
          break

    # Save final state.
    checkpoints.save_checkpoint(checkpoint_dir, state, step, keep=3)
