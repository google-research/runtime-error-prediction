import io

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
    # TODO(dbieber): Support macro f1.
    results[EvaluationMetric.F1_SCORE.value] = metrics.f1_score(
        targets, predictions, average='micro')
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


def plot_to_image(figure):
  """Converts a matplotlib figure to a PNG image tensor.

  Adapted from: https://www.tensorflow.org/tensorboard/image_summaries.

  The given figure is closed and inaccessible after this call.

  Args:
    figure: A matplotlib figure.

  Returns:
    A `[1, height, width, channels]` PNG image tensor.
  """
  # Save the plot to a PNG in memory.
  buf = io.BytesIO()
  plt.savefig(buf, format='png')
  # Closing the figure prevents it from being displayed directly inside
  # notebooks.
  plt.close(figure)
  buf.seek(0)
  # Convert the PNG buffer to a TensorFlow image.
  image = tf.image.decode_png(buf.getvalue(), channels=4)
  # Add a batch dimension of 1.
  image = tf.expand_dims(image, 0)
  return image


def confusion_matrix_to_image(cm, class_names):
  """Returns an image tensor representing the plotted confusion matrix.

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
  return plot_to_image(figure)
