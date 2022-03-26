# Copyright (C) 2021 Google Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Train library."""

import dataclasses
import functools
import itertools
import os
import shutil
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
import numpy as np
import optax
import tensorflow as tf
import tensorflow_datasets as tfds

from core.data import codenet_paths
from core.data import data_io
from core.data import error_kinds
from core.lib import finetune
from core.lib import metadata
from core.lib import metrics
from core.lib import models
from core.lib import optimizer_lib


DEFAULT_DATASET_PATH = codenet_paths.DEFAULT_DATASET_PATH

Config = ml_collections.ConfigDict
EvaluationMetric = metrics.EvaluationMetric


class TrainState(train_state.TrainState):
  rng: Any


@dataclasses.dataclass
class Trainer:

  config: Config
  info: Any

  def load_dataset(
      self, dataset_path=DEFAULT_DATASET_PATH, split='train', epochs=None,
      include_strings=False, class_subsample_values='default',
  ):
    if class_subsample_values == 'default':
      class_subsample_values = {1: 0.0660801055}

    config = self.config
    batch_size = config.batch_size
    if epochs is None:
      # config.epochs == 0 -> None
      epochs = config.epochs or None
    allowlist = config.allowlist

    padded_shapes = data_io.get_padded_shapes(
        config.max_tokens, config.max_num_nodes, config.max_num_edges, include_strings=include_strings)
    if allowlist == 'TIER1_ERROR_IDS':
      allowlist = error_kinds.TIER1_ERROR_IDS
    filter_fn = data_io.make_filter(
        config.max_tokens, config.max_num_nodes, config.max_num_edges,
        config.max_steps, allowlist=allowlist, class_subsample_values=class_subsample_values,
        use_in_dataset_field=config.use_in_dataset_field)

    if config.binary_targets:
      map_fn = functools.partial(data_io.binarize_targets, dataset_path=dataset_path)
    else:
      map_fn = lambda x: x

    if split.endswith('-batch'):
      # Prepare a dataset with a single repeating batch.
      split = split[:-len('-batch')]
      return (
          data_io.load_dataset(dataset_path, split=split, include_strings=include_strings)
          .map(map_fn)
          .filter(filter_fn)
          .take(batch_size)
          .repeat(epochs)
          .padded_batch(batch_size, padded_shapes=padded_shapes)
      )

    # Return the requested dataset.
    return (
        data_io.load_dataset(dataset_path, split=split, include_strings=include_strings)
        .map(map_fn)
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
    else:
      raise ValueError('Unexpected optimizer', config.optimizer)
    # TODO(dbieber): I don't think model.apply is used from here.
    # Instead, it's used from make_loss_fn.
    return TrainState.create(
        apply_fn=model.apply, params=params, tx=tx, rng=rng)

  def create_train_state_from_params(self, rng, model, params, step):
    """Creates initial TrainState. Skips init and uses params."""
    config = self.config
    rng, params_rng, dropout_rng = jax.random.split(rng, 3)
    learning_rate = config.learning_rate
    if config.optimizer == 'sgd':
      tx = optax.sgd(learning_rate)
    elif config.optimizer == 'adam':
      tx = optax.adam(learning_rate)
    else:
      raise ValueError('Unexpected optimizer', config.optimizer)
    # TODO(dbieber): I don't think model.apply is used from here.
    # Instead, it's used from make_loss_fn.
    opt_state = tx.init(params)
    return TrainState(
        step=step,
        apply_fn=model.apply,
        params=params,
        tx=tx,
        opt_state=opt_state,
        rng=rng,
    )

  def restore_checkpoint(self, restore_checkpoint_dir, init_rng, model):
    state_dict = checkpoints.restore_checkpoint(restore_checkpoint_dir, None)
    return self.create_train_state_from_params(init_rng, model, state_dict['params'], state_dict['step'])

  def make_loss_fn(self, deterministic):
    model = self.make_model(deterministic)
    num_classes = self.info.num_classes

    def loss_fn(params, batch, dropout_rng):
      logits, aux = model.apply(
          {'params': params},
          batch,
          rngs={'dropout': dropout_rng}
      )
      aux = aux or {}
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
      aux.update({'logits': logits})
      return loss, aux

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
      (loss, loss_aux), grads = grad_fn(state.params, batch, dropout_rng)
      if self.config.multidevice:
        grads = jax.lax.pmean(grads, 'batch')
      global_norm = optimizer_lib.compute_global_norm(grads)
      if grad_clip_value:
        grads = optimizer_lib.clip_grads(grads, clip_by='global_norm', clip_value=grad_clip_value)
      state = state.apply_gradients(grads=grads)
      # TODO(dbieber): Optionally compute on-device metrics here.
      aux = {
          'logits': loss_aux['logits'],
          'loss': loss,
          'global_norm': global_norm,
      }
      for key in [
          'raise_decisions',
          EvaluationMetric.INSTRUCTION_POINTER.value,
          'instruction_pointer_orig',
          'localization_logits',
      ]:
        if key in loss_aux:
          aux[key] = loss_aux[key]
      return state, aux
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
      return {
          'logits': logits,
          'loss': loss,
          'localization_logits': aux.get('localization_logits'),
      }
    return evaluate_batch

  def run_eval(self, dataset, state, evaluate_batch):
    config = self.config
    num_classes = self.info.num_classes
    predictions = []
    logits_list = []
    localization_predictions = []
    targets = []
    localization_targets = []
    localization_num_targets = []
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
      evaluate_batch_outputs = evaluate_batch(batch, state)
      logits = np.array(evaluate_batch_outputs['logits'])
      loss = np.array(evaluate_batch_outputs['loss'])
      if evaluate_batch_outputs.get('localization_logits') is not None:
        localization_targets.append(np.array(batch['target_node_indexes']))
        localization_num_targets.append(np.array(batch['num_target_nodes']))

        localization_logits = np.array(evaluate_batch_outputs['localization_logits'])
        # localization_logits.shape: [device,] batch_size[/device], num_nodes
        localization_predictions.append(np.argmax(localization_logits, -1))

      logits_list.append(logits)
      predictions.append(np.argmax(logits, -1))
      targets.append(np.array(batch['target']))
      losses.append(loss)
    print('Done evaluating.')
    logits_np = np.concatenate(logits_list)
    # logits_np.shape: num_eval_examples, num_classes
    predictions = np.concatenate(predictions)
    targets = np.concatenate(targets).flatten()
    num_examples = targets.shape[0]
    eval_loss = np.mean(np.array(losses))
    # targets.shape: num_eval_examples
    # predictions.shape: num_eval_examples
    assert predictions.shape == targets.shape
    assert len(predictions.shape) == 1
    if localization_targets:
      localization_targets = np.concatenate(localization_targets)
      localization_num_targets = np.concatenate(localization_num_targets)
      localization_predictions = np.concatenate(localization_predictions)
      # localization_targets.shape: num_eval_examples, [batch_per_device,], max_target_nodes
      if config.multidevice:
        localization_targets = np.reshape(localization_targets, (-1,) + localization_targets.shape[2:])
        localization_num_targets = np.reshape(localization_num_targets, (-1,) + localization_num_targets.shape[2:])
        localization_predictions = np.reshape(localization_predictions, (-1,) + localization_predictions.shape[2:])
    else:
      localization_targets = None
      localization_num_targets = None
      localization_predictions = None
    eval_metrics = metrics.evaluate(
        targets, predictions, logits_np, num_classes,
        localization_targets,
        localization_num_targets,
        localization_predictions,
        config.eval_metric_names,
        self.info)
    return eval_loss, eval_metrics, num_examples

  def run_test(self, dataset_path=DEFAULT_DATASET_PATH, split='test', steps=None):
    config = self.config
    # Ensure eval_subsample==1 and eval_max_batches==-1 are set to test on the full split.
    assert config.eval_subsample == 1
    assert config.eval_max_batches == -1

    print(f'Testing on data: {dataset_path}')
    print(f'Using model: {config.model_class}')
    dataset = self.load_dataset(dataset_path, split=split, epochs=1, class_subsample_values=None)
    num_classes = self.info.num_classes
    all_error_kinds = self.info.all_error_kinds
    evaluate_batch = self.make_evaluate_batch()

    study_id = config.study_id
    exp_id = config.experiment_id or codenet_paths.make_experiment_id()
    run_id = config.run_id or codenet_paths.make_run_id()
    run_dir = codenet_paths.make_run_dir(study_id, exp_id, run_id)

    os.makedirs(run_dir, exist_ok=True)
    metadata_path = codenet_paths.make_metadata_path(run_dir)
    metadata.write_metadata(metadata_path)
    
    test_dir = codenet_paths.make_log_dir(run_dir, 'test')

    rng = jax.random.PRNGKey(0)
    rng, init_rng = jax.random.split(rng)
    model = self.make_model(deterministic=False)

    checkpoint_dir = codenet_paths.make_checkpoints_path(run_dir)
    assert config.restore_checkpoint_dir
    # shutil.copytree(config.restore_checkpoint_dir, checkpoint_dir)
    state = self.restore_checkpoint(config.restore_checkpoint_dir, init_rng, model)
    # Copy the restored checkpoint into the checkpoint_dir.
    step = state.step
    print(f'Step: {step}')

    test_writer = tensorboard.SummaryWriter(test_dir)

    # Determine the file descriptors for the summary writer.
    pid = os.getpid()
    test_writer_fd = None
    if os.path.exists(f'/proc/{pid}/fd'):
      for fd in os.listdir(f'/proc/{pid}/fd'):
        if test_dir in os.path.realpath(f'/proc/{pid}/fd/{fd}'):
          test_writer_fd = int(fd)

    test_writer.hparams(config.to_dict())
    test_writer.flush()
    if test_writer_fd:
      os.fsync(test_writer_fd)
    sys.stdout.flush()

    test_loss, test_metrics, num_examples = self.run_eval(dataset, state, evaluate_batch)

    test_writer.scalar('loss', test_loss, step)
    test_writer.scalar('num_examples', num_examples, step)
    metrics.write_metric(EvaluationMetric.ACCURACY.value, test_metrics,
                         test_writer.scalar, step)
    metrics.write_metric(EvaluationMetric.WEIGHTED_F1_SCORE.value, test_metrics,
                         test_writer.scalar, step)
    metrics.write_metric(EvaluationMetric.MACRO_F1_SCORE.value, test_metrics,
                         test_writer.scalar, step)
    metrics.write_metric(EvaluationMetric.BINARY_F1_SCORE.value, test_metrics,
                         test_writer.scalar, step)
    metrics.write_metric(EvaluationMetric.BINARY_AUC.value, test_metrics,
                         test_writer.scalar, step)
    metrics.write_metric(EvaluationMetric.BINARY_RECALL_AT_90.value, test_metrics,
                         test_writer.scalar, step)
    metrics.write_metric(EvaluationMetric.WEIGHTED_F1_SCORE_ERROR_ONLY.value, test_metrics,
                         test_writer.scalar, step)
    metrics.write_metric(
        EvaluationMetric.CONFUSION_MATRIX.value,
        test_metrics,
        test_writer.image,
        step,
        transform_fn=functools.partial(
            metrics.confusion_matrix_to_image, class_names=all_error_kinds))
    metrics.write_metric(
        EvaluationMetric.LOCALIZATION_ACCURACY.value,
        test_metrics, test_writer.scalar, step)

    sys.stdout.flush()
    test_writer.flush()
    if test_writer_fd:
      os.fsync(test_writer_fd)


  def run_train(self, dataset_path=DEFAULT_DATASET_PATH, split='train', steps=None):
    config = self.config
    info = self.info
    print(f'Training on data: {dataset_path}')
    print(f'Traning using model: {config.model_class}')
    class_subsample_values = info.class_subsample_values
    dataset = self.load_dataset(dataset_path, split=split, class_subsample_values=class_subsample_values)
    num_classes = self.info.num_classes
    all_error_kinds = self.info.all_error_kinds
    valid_dataset = self.load_dataset(dataset_path, split='valid', epochs=1, class_subsample_values=class_subsample_values)
    evaluate_batch = self.make_evaluate_batch()

    study_id = config.study_id
    exp_id = config.experiment_id or codenet_paths.make_experiment_id()
    run_id = config.run_id or codenet_paths.make_run_id()
    run_dir = codenet_paths.make_run_dir(study_id, exp_id, run_id)
    print(run_dir)
    if steps == 0:
      steps = None  # Run forever.

    os.makedirs(run_dir, exist_ok=True)
    metadata_path = codenet_paths.make_metadata_path(run_dir)
    metadata.write_metadata(metadata_path)

    checkpoint_dir = codenet_paths.make_checkpoints_path(run_dir)
    top_checkpoint_dir = codenet_paths.make_top_checkpoints_path(run_dir)
    train_dir = codenet_paths.make_log_dir(run_dir, 'train')
    valid_dir = codenet_paths.make_log_dir(run_dir, 'valid')
    print(f'Checkpoints: {checkpoint_dir}')

    rng = jax.random.PRNGKey(config.seed)
    rng, init_rng = jax.random.split(rng)
    model = self.make_model(deterministic=False)

    if os.path.exists(checkpoint_dir):
      # If we're restoring an interrupted run, that takes priority.
      state = self.restore_checkpoint(checkpoint_dir, init_rng, model)
    elif config.restore_checkpoint_dir:
      # Next, if the config says to start from some checkpoint, do so.
      if config.finetune == 'IPAGNN':
        # The checkpoint we're loading from will have different parameters.
        state = self.create_train_state(init_rng, model)
        state = finetune.finetune_from_ipagnn(state, config.restore_checkpoint_dir, config)
      elif config.finetune == 'LSTM':
        # The checkpoint we're loading from will have different parameters.
        state = self.create_train_state(init_rng, model)
        state = finetune.finetune_from_lstm(state, config.restore_checkpoint_dir, config)
      else:
        assert config.finetune == 'ALL'
        state = self.restore_checkpoint(config.restore_checkpoint_dir, init_rng, model)
    else:
      # Initialize random.
      state = self.create_train_state(init_rng, model)
    train_step = self.make_train_step()

    if config.save_freq >= 50000:
      keep_num = 100
    else:
      # If saving very frequently, don't keep as many checkpoints on disk.
      keep_num = 10

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

    train_logits = []
    train_predictions = []
    train_targets = []
    train_localization_predictions = []
    train_localization_targets = []
    train_localization_num_targets = []
    train_losses = []
    print('Starting training')
    for step_index, batch in itertools.islice(enumerate(tfds.as_numpy(dataset)), steps):
      step = state.step
      if config.multidevice:
        batch = common_utils.shard(batch)
      state, aux = train_step(state, batch)

      # Record training batch evaluation data.
      logits = np.array(aux['logits'])
      predictions = np.squeeze(np.argmax(logits, axis=-1))
      targets = np.squeeze(np.array(batch['target']))
      loss = np.mean(np.array(aux['loss']))
      train_logits.append(logits)
      train_predictions.append(predictions)
      train_targets.append(targets)
      train_losses.append(loss)
      if aux.get('localization_logits') is not None:
        localization_targets = np.array(batch['target_node_indexes'])
        localization_num_targets = np.array(batch['num_target_nodes'])
        localization_predictions = np.argmax(np.array(aux['localization_logits']), axis=-1)
        train_localization_targets.append(localization_targets)
        train_localization_num_targets.append(localization_num_targets)
        train_localization_predictions.append(localization_predictions)
      else:
        localization_predictions = None
        localization_targets = None
        localization_num_targets = None

      # Save checkpoints.
      if step % config.save_freq == 0:
        checkpoints.save_checkpoint(checkpoint_dir, state, step, keep=keep_num)

      # Do batch evaluation.
      if step % config.eval_freq == 0:
        # Evaluate on aggregated training data.
        train_loss = np.mean(np.array(train_losses))
        if train_localization_targets:
          train_localization_targets_np = np.concatenate(train_localization_targets)
          train_localization_num_targets_np = np.concatenate(train_localization_num_targets)
          train_localization_predictions_np = np.concatenate(train_localization_predictions)
          if config.multidevice:
            train_localization_targets_np = np.reshape(train_localization_targets_np, (-1,) + train_localization_targets_np.shape[2:])
            train_localization_num_targets_np = np.reshape(train_localization_num_targets_np, (-1,) + train_localization_num_targets_np.shape[2:])
            train_localization_predictions_np = np.reshape(train_localization_predictions_np, (-1,) + train_localization_predictions_np.shape[2:])
          # train_localization_targets_np.shape: num_examples, max_target_nodes
          # train_localization_num_targets_np.shape: num_examples, 1
          # train_localization_predictions_np.shape: num_examples
        else:
          train_localization_targets_np = None
          train_localization_num_targets_np = None
          train_localization_predictions_np = None
        train_logits_np = np.concatenate(train_logits)
        if config.multidevice:
          train_logits_np = np.reshape(train_logits_np, (-1,) + train_logits_np.shape[2:])
        train_metrics = metrics.evaluate(
            np.reshape(np.array(train_targets), -1),
            np.reshape(np.array(train_predictions), -1),
            train_logits_np,
            num_classes,
            train_localization_targets_np,
            train_localization_num_targets_np,
            train_localization_predictions_np,
            config.eval_metric_names,
            self.info)
        train_accuracy = train_metrics.get(EvaluationMetric.ACCURACY.value)
        train_accuracy_str = (f'{100 * train_accuracy:02.1f}'
                              if train_accuracy is not None else None)
        if localization_predictions is not None:
          # localization_targets.shape: [device,] batch_size, max_target_nodes
          # localization_num_targets.shape: [device,] batch_size, 1
          # localization_predictions.shape: [device,] batch_size
          if config.multidevice:
            localization_targets = np.reshape(localization_targets, (-1,) + localization_targets.shape[2:])
            localization_num_targets = np.reshape(localization_num_targets, (-1,) + localization_num_targets.shape[2:])
            localization_predictions = np.reshape(localization_predictions, (-1,) + localization_predictions.shape[2:])
          # localization_targets.shape: batch_size, max_target_nodes
          # localization_num_targets.shape: batch_size, 1
          # localization_predictions.shape: batch_size
        if config.multidevice:
          logits = np.reshape(logits, (-1,) + logits.shape[2:])
        batch_metrics = metrics.evaluate(
            np.reshape(targets, -1),
            np.reshape(predictions, -1),
            logits,
            num_classes,
            localization_targets,
            localization_num_targets,
            localization_predictions,
            [EvaluationMetric.ACCURACY.value],
            self.info)
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
        train_writer.scalar('global_norm', np.mean(np.array(aux['global_norm'])), step)
        train_writer.scalar('loss', train_loss, step)
        metrics.write_metric(EvaluationMetric.ACCURACY.value, train_metrics,
                             train_writer.scalar, step)
        metrics.write_metric(EvaluationMetric.WEIGHTED_F1_SCORE.value, train_metrics,
                             train_writer.scalar, step)
        metrics.write_metric(EvaluationMetric.MACRO_F1_SCORE.value, train_metrics,
                             train_writer.scalar, step)
        metrics.write_metric(EvaluationMetric.BINARY_F1_SCORE.value, train_metrics,
                             train_writer.scalar, step)
        metrics.write_metric(EvaluationMetric.BINARY_AUC.value, train_metrics,
                             train_writer.scalar, step)
        metrics.write_metric(EvaluationMetric.BINARY_RECALL_AT_90.value, train_metrics,
                             train_writer.scalar, step)
        metrics.write_metric(EvaluationMetric.WEIGHTED_F1_SCORE_ERROR_ONLY.value, train_metrics,
                             train_writer.scalar, step)
        metrics.write_metric(
            EvaluationMetric.CONFUSION_MATRIX.value,
            train_metrics,
            train_writer.image,
            step,
            transform_fn=functools.partial(
                metrics.confusion_matrix_to_image, class_names=all_error_kinds))
        metrics.write_metric(
            EvaluationMetric.INSTRUCTION_POINTER.value,
            aux,
            train_writer.image,
            step,
            transform_fn=functools.partial(
                metrics.instruction_pointers_to_images,
                multidevice=config.multidevice))
        metrics.write_metric(
            EvaluationMetric.LOCALIZATION_ACCURACY.value,
            train_metrics, train_writer.scalar, step)

        # Write validation metrics.
        valid_writer.scalar('loss', valid_loss, step)
        metrics.write_metric(EvaluationMetric.ACCURACY.value, valid_metrics,
                             valid_writer.scalar, step)
        metrics.write_metric(EvaluationMetric.WEIGHTED_F1_SCORE.value, valid_metrics,
                             valid_writer.scalar, step)
        metrics.write_metric(EvaluationMetric.MACRO_F1_SCORE.value, valid_metrics,
                             valid_writer.scalar, step)
        metrics.write_metric(EvaluationMetric.BINARY_F1_SCORE.value, valid_metrics,
                             valid_writer.scalar, step)
        metrics.write_metric(EvaluationMetric.BINARY_AUC.value, valid_metrics,
                             valid_writer.scalar, step)
        metrics.write_metric(EvaluationMetric.BINARY_RECALL_AT_90.value, valid_metrics,
                             valid_writer.scalar, step)
        metrics.write_metric(EvaluationMetric.WEIGHTED_F1_SCORE_ERROR_ONLY.value, valid_metrics,
                             valid_writer.scalar, step)
        metrics.write_metric(
            EvaluationMetric.CONFUSION_MATRIX.value,
            valid_metrics,
            valid_writer.image,
            step,
            transform_fn=functools.partial(
                metrics.confusion_matrix_to_image, class_names=all_error_kinds))
        metrics.write_metric(
            EvaluationMetric.LOCALIZATION_ACCURACY.value,
            valid_metrics, valid_writer.scalar, step)

        primary_metric_value = valid_metrics[config.eval_primary_metric_name]
        # Higher is better for primary_metric_value_pos.
        primary_metric_value_pos = config.eval_primary_metric_scale * primary_metric_value

        did_improve, es = es.update(-1 * primary_metric_value_pos)
        if did_improve:
          checkpoints.save_checkpoint(top_checkpoint_dir, state, state.step, keep=3, overwrite=True)

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
        train_logits.clear()
        train_predictions.clear()
        train_targets.clear()
        train_losses.clear()
        train_localization_targets.clear()
        train_localization_num_targets.clear()
        train_localization_predictions.clear()

    # Save final state.
    checkpoints.save_checkpoint(checkpoint_dir, state, state.step, keep=keep_num)
