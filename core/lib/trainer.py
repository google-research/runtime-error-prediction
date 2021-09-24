"""Train library."""

import dataclasses
import functools
import itertools
import os
import sys

from typing import Any

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

from config.default import EvaluationMetric
from core.data import codenet_paths
from core.data import data_io
from core.data import error_kinds
from core.lib import evaluation  # TODO(dbieber): Rename evaluation into metrics.
from core.lib import metadata
from core.lib import models
from core.lib import optimizer_lib


DEFAULT_DATASET_PATH = codenet_paths.DEFAULT_DATASET_PATH

Config = ml_collections.ConfigDict


class TrainState(train_state.TrainState):
  rng: Any


@dataclasses.dataclass
class Trainer:

  config: Config
  info: Any

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

  def make_model(self, deterministic):
    return models.make_model(self.config, self.info, deterministic)

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
    if config.optimizer == 'sgd':
      tx = optax.sgd(learning_rate)
    elif config.optimizer == 'adam':
      tx = optax.adam(learning_rate)
    # TODO(dbieber): I don't think model.apply is used from here.
    # Instead, it's used from make_loss_fn.
    return TrainState.create(
        apply_fn=model.apply, params=params, tx=tx, rng=rng)

  def make_loss_fn(self, deterministic):
    model = self.make_model(deterministic)
    num_classes = self.info.num_classes

    def loss_fn(params, batch, dropout_rng):
      logits = model.apply(
          {'params': params},
          batch,
          rngs={'dropout': dropout_rng}
      )
      assert len(logits.shape) == 2
      # logits.shape: batch_size, num_classes
      labels = jax.nn.one_hot(jnp.squeeze(batch['target'], axis=-1), num_classes)
      assert len(labels.shape) == 2
      # labels.shape: batch_size, num_classes
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
    loss_fn = self.make_loss_fn(deterministic=False)
    grad_clip_value = self.config.grad_clip_value

    @jax.jit
    def train_step(state, batch):
      """The on-device part of a train step."""
      new_rng, dropout_rng = jax.random.split(state.rng, 2)
      state = dataclasses.replace(state, rng=new_rng)

      grad_fn = jax.value_and_grad(loss_fn, argnums=0, has_aux=True)
      (loss, aux), grads = grad_fn(state.params, batch, dropout_rng)
      if self.config.multidevice:
        grads = jax.lax.pmean(grads, 'batch')
      global_norm = optimizer_lib.compute_global_norm(grads)
      if grad_clip_value:
        grads = optimizer_lib.clip_grads(grads, clip_by='global_norm', clip_value=grad_clip_value)
      state = state.apply_gradients(grads=grads)
      # TODO(dbieber): Optionally compute on-device metrics here.
      return state, {
          'logits': aux['logits'],
          'loss': loss,
          'global_norm': global_norm,
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
    num_classes = self.info.num_classes
    loss_fn = self.make_loss_fn(deterministic=True)
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
      # logits.shape: batch_size, num_classes
      # targets.shape: batch_size

      metric = evaluation.compute_metric(
          logits, targets, num_classes, config.eval_metric_names
      )
      return logits, loss, metric
    return evaluate_batch

  def run_eval(self, dataset, state, evaluate_batch):
    config = self.config
    num_classes = self.info.num_classes
    predictions = []
    targets = []
    losses = []
    dataset = (
        dataset
        .filter(lambda x: tf.random.uniform(shape=()) < config.eval_subsample)
        .take(config.eval_max_batches)
    )
    print(f'Evaluating with metrics: {config.eval_metric_names}')
    for batch in tfds.as_numpy(dataset):
      if config.multidevice:
        batch = common_utils.shard(batch)
      logits, loss, _ = evaluate_batch(batch, state)
      predictions.append(jnp.argmax(logits, -1))
      targets.append(batch['target'])
      losses.append(loss)
    print('Done evaluating.')
    predictions = jnp.array(jnp.concatenate(predictions))
    targets = jnp.array(jnp.concatenate(targets)).flatten()
    num_examples = targets.shape[0]
    eval_loss = jnp.mean(jnp.array(losses))
    # targets.shape: num_eval_examples
    # predictions.shape: num_eval_examples
    assert predictions.shape == targets.shape
    assert len(predictions.shape) == 1
    eval_metrics = evaluation.evaluate(
        targets, predictions, num_classes, config.eval_metric_names)
    return eval_loss, eval_metrics, num_examples

  def run_train(self, dataset_path=DEFAULT_DATASET_PATH, split='train', steps=None):
    config = self.config
    print(f'Training on data: {dataset_path}')
    dataset = self.load_dataset(dataset_path, split=split)
    num_classes = self.info.num_classes
    all_error_kinds = self.info.all_error_kinds
    valid_dataset = self.load_dataset(dataset_path, split='valid', epochs=1)
    evaluate_batch = self.make_evaluate_batch()

    study_id = config.study_id
    exp_id = config.experiment_id or codenet_paths.make_experiment_id()
    run_id = config.run_id or codenet_paths.make_run_id()
    run_dir = codenet_paths.make_run_dir(study_id, exp_id, run_id)

    os.makedirs(run_dir, exist_ok=True)
    metadata_path = codenet_paths.make_metadata_path(run_dir)
    metadata.write_metadata(metadata_path)

    checkpoint_dir = codenet_paths.make_checkpoints_path(run_dir)
    train_dir = codenet_paths.make_log_dir(run_dir, 'train')
    valid_dir = codenet_paths.make_log_dir(run_dir, 'valid')
    print(f'Checkpoints: {checkpoint_dir}')

    rng = jax.random.PRNGKey(0)
    rng, init_rng = jax.random.split(rng)
    model = self.make_model(deterministic=False)

    state = self.create_train_state(init_rng, model)
    if os.path.exists(checkpoint_dir):
      # If we're restoring an interrupted run, that takes priority.
      state = checkpoints.restore_checkpoint(checkpoint_dir, state)
    elif config.restore_checkpoint_dir:
      # Next, if the config says to start from some checkpoint, do so.
      state = checkpoints.restore_checkpoint(config.restore_checkpoint_dir, state)
    train_step = self.make_train_step()

    # TODO(rishab): Store the state of the early stopping.
    es = early_stopping.EarlyStopping(
        min_delta=config.early_stopping_delta,
        patience=config.early_stopping_threshold,
    )
    train_writer = tensorboard.SummaryWriter(train_dir)
    valid_writer = tensorboard.SummaryWriter(valid_dir)

    # Determine the file descriptors for the summary writers.
    pid = os.getpid()
    train_writer_fd = valid_writer_fd = None
    if os.path.exists(f'/proc/{pid}/fd'):
      for fd in os.listdir(f'/proc/{pid}/fd'):
        if train_dir in os.path.realpath(f'/proc/{pid}/fd/{fd}'):
          train_writer_fd = int(fd)
        if valid_dir in os.path.realpath(f'/proc/{pid}/fd/{fd}'):
          valid_writer_fd = int(fd)

    train_writer.hparams(config.to_dict())
    train_writer.flush()
    if train_writer_fd:
      os.fsync(train_writer_fd)
    sys.stdout.flush()

    train_predictions = []
    train_targets = []
    train_losses = []
    for step_index, batch in itertools.islice(enumerate(tfds.as_numpy(dataset)), steps):
      step = state.step
      if config.multidevice:
        batch = common_utils.shard(batch)
      state, aux = train_step(state, batch)

      # Record training batch evaluation data.
      predictions = jnp.squeeze(jnp.argmax(aux['logits'], axis=-1))
      targets = jnp.squeeze(batch['target'])
      loss = jnp.mean(aux['loss'])
      train_predictions.append(predictions)
      train_targets.append(targets)
      train_losses.append(loss)

      # Save checkpoints.
      if step % config.save_freq == 0:
        checkpoints.save_checkpoint(checkpoint_dir, state, step, keep=3)

      # Do batch evaluation.
      if step % config.eval_freq == 0:
        # Evaluate on aggregated training data.
        train_loss = jnp.mean(jnp.array(train_losses))
        train_metrics = evaluation.evaluate(
            jnp.reshape(jnp.array(train_targets), -1),
            jnp.reshape(jnp.array(train_predictions), -1),
            num_classes,
            config.eval_metric_names)
        train_accuracy = train_metrics.get(EvaluationMetric.ACCURACY.value)
        train_accuracy_str = (f'{100 * train_accuracy:02.1f}'
                              if train_accuracy is not None else None)
        batch_metrics = evaluation.evaluate(
            jnp.reshape(targets, -1),
            jnp.reshape(predictions, -1),
            num_classes,
            [EvaluationMetric.ACCURACY.value])
        batch_accuracy = batch_metrics[EvaluationMetric.ACCURACY.value]
        print(f"""--- Step {step}
Loss: {train_loss}
Predictions:
{predictions}
Targets:
{targets}
Train Accuracy: {train_accuracy_str}
Last Minibatch Accuracy: {100 * batch_accuracy:02.1f}""")

        # Evaluate on validation dataset.
        valid_loss, valid_metrics, num_examples = self.run_eval(
            valid_dataset, state, evaluate_batch)
        logging.info(
            f'Validation loss ({num_examples} examples): {valid_loss}\n'
            f'Validation metrics: {valid_metrics}'
        )

        # Write training metrics.
        train_writer.scalar('global_norm', jnp.mean(aux['global_norm']), step)
        train_writer.scalar('loss', train_loss, step)
        write_metric(EvaluationMetric.ACCURACY.value, train_metrics,
                     train_writer.scalar, step)
        write_metric(EvaluationMetric.F1_SCORE.value, train_metrics,
                     train_writer.scalar, step)
        write_metric(
            EvaluationMetric.CONFUSION_MATRIX.value,
            train_metrics,
            train_writer.image,
            step,
            transform_fn=functools.partial(
                evaluation.confusion_matrix_to_image,
                class_names=all_error_kinds))

        # Write validation metrics.
        valid_writer.scalar('loss', valid_loss, step)
        write_metric(EvaluationMetric.ACCURACY.value, valid_metrics,
                     valid_writer.scalar, step)
        write_metric(EvaluationMetric.F1_SCORE.value, valid_metrics,
                     valid_writer.scalar, step)
        write_metric(
            EvaluationMetric.CONFUSION_MATRIX.value,
            valid_metrics,
            valid_writer.image,
            step,
            transform_fn=functools.partial(
                evaluation.confusion_matrix_to_image,
                class_names=all_error_kinds))

        did_improve, es = es.update(-1 * valid_loss)
        if es.should_stop and config.early_stopping_on:
          logging.info('Early stopping triggered.')
          break

        sys.stdout.flush()
        train_writer.flush()
        valid_writer.flush()
        if train_writer_fd:
          os.fsync(train_writer_fd)
        if valid_writer_fd:
          os.fsync(valid_writer_fd)

        # Clear training evaluation data.
        train_predictions.clear()
        train_targets.clear()
        train_losses.clear()

    # Save final state.
    checkpoints.save_checkpoint(checkpoint_dir, state, state.step, keep=3)


def write_metric(metric_name, metrics_dict, summary_fn, step, transform_fn=None):
  """Writes an evaluation metric using a TensorBoard SummaryWriter function."""
  if metric_name in metrics_dict:
    metric = metrics_dict[metric_name]
    if transform_fn is not None:
      metric = transform_fn(metric)
    summary_fn(metric_name, metric, step)
