"""Train library."""

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
import tensorflow as tf
import tensorflow_datasets as tfds

from core.data import codenet_paths
from core.data import data_io
from core.data import error_kinds
from core.lib import evaluation  # TODO(dbieber): Rename evaluation into metrics.
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
    self, dataset_path=DEFAULT_DATASET_PATH, split='train', epochs=None
  ):
    config = self.config
    batch_size = config.batch_size
    if epochs is None:
      # config.epochs == 0 -> None
      epochs = config.epochs or None
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
        .padded_batch(batch_size, padded_shapes=padded_shapes, drop_remainder=True)
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
    learning_rate = config.learning_rate
    tx = optax.sgd(learning_rate)
    return TrainState.create(
        apply_fn=model.apply, params=params, tx=tx, rng=rng)

  def make_loss_fn(self):
    model = self.make_model()

    def loss_fn(params, batch, dropout_rng):
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

    return loss_fn

  def make_train_step(self):
    loss_fn = self.make_loss_fn()

    @jax.jit
    def train_step(state, batch):
      """The on-device part of a train step."""
      model = self.make_model()

      new_rng, dropout_rng = jax.random.split(state.rng, 2)
      state = dataclasses.replace(state, rng=new_rng)

      grad_fn = jax.value_and_grad(loss_fn, argnums=0, has_aux=True)
      (loss, aux), grads = grad_fn(state.params, batch, dropout_rng)
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

  def make_evaluate_batch(self):
    config = self.config
    loss_fn = self.make_loss_fn()
    if config.multidevice:
      loss_fn = jax.pmap(
          loss_fn,
          axis_name='batch',
          in_axes=(None, 0, None),
          out_axes=0,
      )
    def evaluate_batch(batch, state):
      new_rng, dropout_rng = jax.random.split(state.rng, 2)
      state = dataclasses.replace(state, rng=new_rng)

      # TODO(dbieber): Dropout shouldn't be used during the evaluation.
      # TODO(dbieber): And if it were to be used, we'd want per-device randoms.
      loss, aux = loss_fn(state.params, batch, dropout_rng)

      logits = aux['logits']
      targets = jnp.squeeze(batch['target'], axis=-1)
      if config.multidevice:
        loss = jnp.mean(loss)
        logits = jnp.reshape(logits, (-1,) + logits.shape[2:])
        targets = jnp.reshape(targets, (-1,) + targets.shape[2:])
      # logits.shape: batch_size, NUM_CLASSES
      # targets.shape: batch_size

      metric = evaluation.compute_metric(
          logits, targets, config.eval_metric_name
      )
      return logits, loss, metric
    return evaluate_batch

  def run_eval(self, dataset, state, evaluate_batch):
    config = self.config
    predictions = []
    targets = []
    losses = []
    dataset = (
        dataset
        .filter(lambda x: tf.random.uniform(shape=()) < config.eval_subsample)
        .take(config.eval_max_batches)
    )
    print(f'Evaluating with metric: {config.eval_metric_name}')
    for batch in tfds.as_numpy(dataset):
      if config.multidevice:
        batch = common_utils.shard(batch)
      logits, loss, _ = evaluate_batch(batch, state)
      predictions.append(jnp.argmax(logits, -1))
      targets.append(batch['target'])
      losses.append(loss)
    predictions = jnp.array(jnp.concatenate(predictions))
    targets = jnp.array(jnp.concatenate(targets)).flatten()
    num_examples = targets.shape[0]
    eval_loss = jnp.mean(jnp.array(losses))
    # targets.shape: num_eval_examples
    # predictions.shape: num_eval_examples
    assert predictions.shape == targets.shape
    assert len(predictions.shape) == 1
    eval_metric = evaluation.evaluate(
        targets, predictions, config.eval_metric_name
    )
    return eval_loss, eval_metric, num_examples

  def run_train(self, dataset_path=DEFAULT_DATASET_PATH, split='train', steps=None):
    config = self.config
    print(f'Training on data: {dataset_path}')
    dataset = self.load_dataset(dataset_path, split=split)
    eval_dataset = self.load_dataset(dataset_path, split='valid', epochs=1)
    evaluate_batch = self.make_evaluate_batch()

    study_id = config.study_id
    exp_id = config.experiment_id or codenet_paths.make_experiment_id()
    run_id = config.run_id or codenet_paths.make_run_id()
    run_dir = codenet_paths.make_run_dir(study_id, exp_id, run_id)

    checkpoint_dir = codenet_paths.make_checkpoints_path(run_dir)
    train_dir = codenet_paths.make_log_dir(run_dir, 'train')
    valid_dir = codenet_paths.make_log_dir(run_dir, 'valid')
    print(f'Checkpoints: {checkpoint_dir}')

    rng = jax.random.PRNGKey(0)
    rng, init_rng = jax.random.split(rng)
    model = self.make_model()

    state = self.create_train_state(init_rng, model)
    if config.restore_checkpoint_dir:
      state = checkpoints.restore_checkpoint(config.restore_checkpoint_dir, state)
    train_step = self.make_train_step()

    # TODO(rishab): Store the state of the early stopping.
    es = early_stopping.EarlyStopping(
        min_delta=config.early_stopping_delta,
        patience=config.early_stopping_threshold,
    )
    train_writer = tensorboard.SummaryWriter(train_dir)
    valid_writer = tensorboard.SummaryWriter(valid_dir)
    train_writer.hparams(config.to_dict())

    recent_accuracies = []
    for step_index, batch in itertools.islice(enumerate(tfds.as_numpy(dataset)), steps):
      step = state.step
      if config.multidevice:
        batch = common_utils.shard(batch)
      state, aux = train_step(state, batch)

      # Compute batch eval.
      predictions = jnp.squeeze(jnp.argmax(aux['logits'], axis=-1))
      targets = jnp.squeeze(batch['target'])
      batch_accuracy = jnp.sum(predictions == targets) / jnp.sum(jnp.ones_like(targets))
      recent_accuracies.append(batch_accuracy)
      recent_accuracies = recent_accuracies[-500:]

      if step % config.save_freq == 0:
        checkpoints.save_checkpoint(checkpoint_dir, state, step, keep=3)

      if step % config.eval_freq == 0:
        print(f"""--- Step {step}
Loss: {aux['loss']}
Predictions:
{predictions}
Targets:
{targets}
Batch Accuracy: {100 * batch_accuracy:02.1f}
Recent Accuracy: {100 * jnp.mean(jnp.array(recent_accuracies)):02.1f}""")

        # Run complete evaluation:
        eval_loss, eval_metric, num_examples = self.run_eval(
            eval_dataset, state, evaluate_batch)
        logging.info(
            f'Validation loss ({num_examples} Examples): {eval_loss}\n'
            f'Validation {config.eval_metric_name}: {eval_metric}'
        )
        (
            _,
            batch_loss,
            batch_metric,
        ) = evaluate_batch(batch, state)
        train_writer.scalar('loss', batch_loss, step)
        train_writer.scalar(
            'recent_accuracy', jnp.mean(jnp.array(recent_accuracies)), step)
        train_writer.scalar('train_metric', batch_metric, step)
        valid_writer.scalar('loss', eval_loss, step)
        valid_writer.scalar('eval_metric', eval_metric, step)

        did_improve, es = es.update(-1 * eval_loss)
        if es.should_stop and config.early_stopping_on:
          logging.info('Early stopping triggered.')
          break

        train_writer.flush()
        valid_writer.flush()

    # Save final state.
    checkpoints.save_checkpoint(checkpoint_dir, state, state.step, keep=3)
