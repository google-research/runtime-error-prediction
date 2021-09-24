"""Metrics utility functions."""

import imageio
import io
import itertools

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from sklearn import metrics
import tensorflow as tf

from config.default import EvaluationMetric


def evaluate(targets, predictions, num_classes, eval_metric_names):
  # Diagnose unknown metrics.
  unknown_metric_names = set(eval_metric_names).difference(
      EvaluationMetric.all_metric_names())
  if unknown_metric_names:
    raise ValueError(f'Unknown metric names: {unknown_metric_names}')

  # Compute metrics.
  results = {}
  if EvaluationMetric.ACCURACY.value in eval_metric_names:
    results[EvaluationMetric.ACCURACY.value] = (
        jnp.sum(predictions == targets) / jnp.sum(jnp.ones_like(targets)))
  if EvaluationMetric.F1_SCORE.value in eval_metric_names:
    results[EvaluationMetric.F1_SCORE.value] = metrics.f1_score(
        targets, predictions, average='macro')
  if EvaluationMetric.CONFUSION_MATRIX.value in eval_metric_names:
    results[EvaluationMetric.CONFUSION_MATRIX.value] = metrics.confusion_matrix(
        targets,
        predictions,
        labels=range(num_classes),
        normalize='true')
  return results


def compute_metric(logits, targets, num_classes, eval_metric_names):
  predictions = np.array(jnp.argmax(logits, -1))
  targets = np.array(targets)
  metrics = evaluate(targets, predictions, num_classes, eval_metric_names)
  return metrics


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
