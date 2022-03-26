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

"""Metrics utility functions."""

import enum
import io
from typing import Tuple

import imageio
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from sklearn import metrics


class EvaluationMetric(enum.Enum):
  """Evaluation metric kinds."""

  def _generate_next_value_(name, start, count, last_values):
    return name.lower()

  ACCURACY = enum.auto()
  WEIGHTED_F1_SCORE = enum.auto()
  WEIGHTED_F1_SCORE_ERROR_ONLY = enum.auto()
  MACRO_F1_SCORE = enum.auto()
  BINARY_F1_SCORE = enum.auto()
  BINARY_AUC = enum.auto()
  BINARY_RECALL_AT_90 = enum.auto()
  CONFUSION_MATRIX = enum.auto()
  INSTRUCTION_POINTER = enum.auto()
  LOCALIZATION_ACCURACY = enum.auto()


def all_metric_names() -> Tuple[str]:
  """"Returns a tuple of all evaluation metric names."""
  return tuple(m.value for m in EvaluationMetric)


def evaluate(targets, predictions, logits, num_classes,
             localization_targets, localization_num_targets, localization_predictions,
             eval_metric_names, info):
  # Diagnose unknown metrics.
  unknown_metric_names = set(eval_metric_names).difference(all_metric_names())
  if unknown_metric_names:
    raise ValueError(f'Unknown metric names: {unknown_metric_names}')

  # Compute metrics.
  # logits.shape: num_eval_examples, num_classes
  results = {}
  if EvaluationMetric.ACCURACY.value in eval_metric_names:
    results[EvaluationMetric.ACCURACY.value] = (
        jnp.sum(predictions == targets) / jnp.sum(jnp.ones_like(targets)))
  if EvaluationMetric.MACRO_F1_SCORE.value in eval_metric_names:
    results[EvaluationMetric.MACRO_F1_SCORE.value] = metrics.f1_score(
        targets, predictions, average='macro')
  if EvaluationMetric.WEIGHTED_F1_SCORE.value in eval_metric_names:
    results[EvaluationMetric.WEIGHTED_F1_SCORE.value] = metrics.f1_score(
        targets, predictions, average='weighted')
  if EvaluationMetric.BINARY_F1_SCORE.value in eval_metric_names:
    results[EvaluationMetric.BINARY_F1_SCORE.value] = compute_binary_f1_score(
        targets, logits, info)
  if EvaluationMetric.BINARY_AUC.value in eval_metric_names:
    results[EvaluationMetric.BINARY_AUC.value] = compute_binary_auc(
        targets, logits, info)
  if EvaluationMetric.BINARY_RECALL_AT_90.value in eval_metric_names:
    results[EvaluationMetric.BINARY_RECALL_AT_90.value] = compute_recall_at_precision(
        targets, logits, info, target_precision=0.90)
  if EvaluationMetric.WEIGHTED_F1_SCORE_ERROR_ONLY.value in eval_metric_names:
    results[EvaluationMetric.WEIGHTED_F1_SCORE_ERROR_ONLY.value] = compute_weighted_f1_score_error_only(
        targets, predictions, info)
  if EvaluationMetric.CONFUSION_MATRIX.value in eval_metric_names and num_classes < 40:
    results[EvaluationMetric.CONFUSION_MATRIX.value] = metrics.confusion_matrix(
        targets,
        predictions,
        labels=range(num_classes),
        normalize='true')
  if EvaluationMetric.LOCALIZATION_ACCURACY.value in eval_metric_names:
    localization_accuracy = compute_localization_accuracy(
        localization_targets, localization_num_targets, localization_predictions
    )
    if localization_accuracy is not None:
      results[EvaluationMetric.LOCALIZATION_ACCURACY.value] = localization_accuracy
  return results


def write_metric(metric_name,
                 metrics_dict,
                 summary_fn,
                 step,
                 transform_fn=None):
  """Writes an evaluation metric using a TensorBoard SummaryWriter function."""
  if metric_name in metrics_dict:
    metric = metrics_dict[metric_name]
    if transform_fn is not None:
      metric = transform_fn(metric)
    summary_fn(metric_name, metric, step)


def figure_to_image(figure, dpi=None, close=True):
  """Converts the matplotlib plot specified by `figure` to a NumPy image.

  Args:
    figure: A matplotlib plot.

  Returns:
    A 3-D NumPy array representing the image of the matplotlib plot.
  """
  # Save the plot to a PNG in memory.
  buf = io.BytesIO()
  figure.savefig(buf, format='png', dpi=dpi, bbox_inches='tight')
  buf.seek(0)
  # Convert PNG buffer to NumPy array.
  image = imageio.imread(buf, format='png')
  buf.close()
  if close:
    plt.close(figure)
  return image


def make_figure(*,
                data,
                title,
                xlabel,
                ylabel,
                interpolation='nearest',
                **kwargs):
  """"Creates a matplotlib plot from the given data."""
  fig = plt.figure()
  ax = fig.add_subplot(111)
  ax.set_title(title)
  plt.imshow(data, interpolation=interpolation, **kwargs)
  ax.set_aspect('equal')
  ax.set_xlabel(xlabel)
  ax.set_ylabel(ylabel)
  plt.colorbar(orientation='vertical')
  return fig


def instruction_pointer_to_image(instruction_pointer):
  """Converts the given instruction pointer array to an image."""
  instruction_pointer_figure = make_figure(
      data=instruction_pointer,
      title='Instruction Pointer',
      xlabel='Timestep',
      ylabel='Node')
  return figure_to_image(instruction_pointer_figure)


def confusion_matrix_to_image(cm, class_names):
  """Returns an image tensor representing the confusion matrix plotted.

  Args:
    cm: a `[num_classes, num_classes]` confusion matrix of integer classes.
    class_names: an `[num_classes]` array of the names of the integer classes.

  Returns:
    A `[1, height, width, channels]` PNG image tensor representing the confusion
    matrix.
  """
  cm_display = metrics.ConfusionMatrixDisplay(cm, display_labels=class_names)
  cm_display.plot(xticks_rotation=45)
  figure = cm_display.figure_
  image = figure_to_image(figure)
  return np.expand_dims(image, 0)


def instruction_pointers_to_images(instruction_pointer, multidevice: bool):
  """Converts the given batched instruction pointer to images."""
  if multidevice:
    # instruction_pointer: device, batch_size / device, timesteps, num_nodes
    instruction_pointer = instruction_pointer[0]

  # instruction_pointer: batch_size / device, timesteps, num_nodes
  instruction_pointer = jnp.transpose(instruction_pointer[:, :16, :],
                                      (1, 2, 0))
  # instruction_pointer: logging_slice_size, num_nodes, timesteps
  instruction_pointer_image_list = [
      instruction_pointer_to_image(ip)
      for ip in instruction_pointer
  ]
  instruction_pointer_image_leading_dim_max = max(
      image.shape[0] for image in instruction_pointer_image_list)
  instruction_pointer_image_list = [
      pad(image, instruction_pointer_image_leading_dim_max)
      for image in instruction_pointer_image_list
  ]
  return jnp.array(instruction_pointer_image_list)


def pad(array, leading_dim_size: int):
  """Pad the leading dimension of the given array."""
  leading_dim_difference = max(0, leading_dim_size - array.shape[0])
  leading_pad_width = [(0, leading_dim_difference)]
  trailing_pad_widths = [(0, 0)] * (array.ndim - 1)
  return jnp.pad(array, leading_pad_width + trailing_pad_widths)


def compute_localization_accuracy(
    localization_targets, localization_num_targets, localization_predictions):
  if localization_predictions is None:
    return None

  def is_correct(targets, num_targets, prediction):
    # targets.shape: max_num_targets
    # num_targets.shape: scalar.
    is_example = num_targets > 0
    mask = jnp.arange(targets.shape[0]) < num_targets
    # mask.shape: max_num_nodes
    correct = targets == prediction
    # correct.shape: max_num_nodes
    correct_and_valid = jnp.logical_and(mask, correct)
    # correct_and_valid.shape: max_num_nodes
    overall_correct = jnp.max(correct_and_valid, axis=-1)
    # overall_correct.shape: scalar.
    return overall_correct, is_example
  is_corrects, is_examples = jax.vmap(is_correct)(
      localization_targets, localization_num_targets, localization_predictions)
  # is_corrects.shape: num_examples
  total_correct = jnp.sum(is_corrects)
  total_examples = jnp.maximum(1, jnp.sum(is_examples))
  return total_correct / total_examples


def compute_binary_targets(targets, info):
  targets = jnp.array(targets)
  error_ids = jnp.array(info.error_ids)
  def matches(t, idx):
    return t == idx
  def matches_any(t, indexes):
    matches_each = jax.vmap(matches, in_axes=(None, 0))(t, indexes)
    # matches_each.shape: batch_size, num_indexes
    return jnp.max(matches_each, axis=-1)
  # matches = jax.vmap(lambda t: jnp.equals(targets, t))(, out_axes=1)
  binary_targets = jax.vmap(matches_any, in_axes=(0, None))(targets, error_ids)
  # ms.shape: batch_size
  # In binary_targets, True indicates the target is error and False no-error.
  return binary_targets


def compute_binary_predictions(logits, info):
  logits = jnp.array(logits)
  get_logits = jax.vmap(lambda index: logits[:, index], out_axes=1)
  no_error_logits = get_logits(jnp.array(info.no_error_ids))
  error_logits = get_logits(jnp.array(info.error_ids))
  # no_error_logits.shape: batch_size, num_no_error_classes
  # error_logits.shape: batch_size, num_error_classes

  no_error_ps = jax.scipy.special.logsumexp(no_error_logits, axis=-1)
  error_ps = jax.scipy.special.logsumexp(error_logits, axis=-1)
  # no_error_ps.shape: batch_size
  # error_ps.shape: batch_size
  binary_predictions = error_ps >= no_error_ps
  # binary_predictions.shape: batch_size
  # True indicates the prediction is error, False indicates no-error.
  return binary_predictions


def compute_binary_probabilities(logits, info):
  logits = jnp.array(logits)
  get_logits = jax.vmap(lambda index: logits[:, index], out_axes=1)
  no_error_logits = get_logits(jnp.array(info.no_error_ids))
  error_logits = get_logits(jnp.array(info.error_ids))
  # no_error_logits.shape: batch_size, num_no_error_classes
  # error_logits.shape: batch_size, num_error_classes
  no_error_ps = jax.scipy.special.logsumexp(no_error_logits, axis=-1)
  error_ps = jax.scipy.special.logsumexp(error_logits, axis=-1)
  # no_error_ps.shape: batch_size
  # error_ps.shape: batch_size
  binary_logits = jnp.stack([error_ps, no_error_ps], axis=-1)
  # binary_logits.shape: batch_size, 2
  return jax.nn.softmax(binary_logits)  # P(error), P(no-error)

def compute_binary_f1_score(targets, logits, info):
  binary_predictions = compute_binary_predictions(logits, info)
  binary_targets = compute_binary_targets(targets, info)
  metric = metrics.f1_score(binary_targets, binary_predictions, average='binary')
  return metric


def compute_binary_auc(targets, logits, info):
  binary_targets = jnp.int32(compute_binary_targets(targets, info))
  binary_probabilities = compute_binary_probabilities(logits, info)
  error_probabilities = binary_probabilities[:, 0]  # P(error)
  return metrics.roc_auc_score(binary_targets, error_probabilities)


def compute_recall_at_precision(targets, logits, info, target_precision):
  binary_targets = jnp.int32(compute_binary_targets(targets, info))
  binary_probabilities = compute_binary_probabilities(logits, info)
  error_probabilities = binary_probabilities[:, 0]  # P(error)
  precisions, recalls, thresholds = metrics.precision_recall_curve(binary_targets, error_probabilities, pos_label=1)
  for precision, recall in zip(precisions, recalls):
    # The last precision value is 1, starts from ~0.
    # The last recall value is 0, starts from ~1.
    if precision >= target_precision:
      return recall
  return 0


def compute_precision_at_recall(targets, logits, info, target_recall):
  binary_targets = jnp.int32(compute_binary_targets(targets, info))
  binary_probabilities = compute_binary_probabilities(logits, info)
  error_probabilities = binary_probabilities[:, 0]  # P(error)
  precisions, recalls, thresholds = metrics.precision_recall_curve(binary_targets, error_probabilities, pos_label=1)
  for precision, recall in reversed(list(zip(precisions, recalls))):
    # The first precision value is 1, tends toward 0.
    # The first recall value is 0, tends toward 1.
    if recall >= target_recall:
      return precision
  return 0


def compute_weighted_f1_score_error_only(targets, predictions, info):
  labels = info.error_ids
  metric = metrics.f1_score(targets, predictions, labels=labels, average='weighted')
  return metric
