import jax
import jax.numpy as jnp
import numpy as np

from sklearn import metrics

from config.default import EvaluationMetric
from core.data import error_kinds

NUM_CLASSES = error_kinds.NUM_CLASSES


def evaluate(targets, predictions, eval_metric_names):
  results = {}
  if EvaluationMetric.F1_SCORE.value in eval_metric_names:
    # TODO(dbieber): Support macro f1.
    results[EvaluationMetric.F1_SCORE.value] = metrics.f1_score(
        targets, predictions, average='micro')
  if EvaluationMetric.CONFUSION_MATRIX.value in eval_metric_names:
    results[EvaluationMetric.CONFUSION_MATRIX.value] = metrics.confusion_matrix(
        targets, predictions)
  return results


def compute_metric(logits, targets, eval_metric_names):
  predictions = np.array(jnp.argmax(logits, -1))
  targets = np.array(targets)
  metrics = evaluate(targets, predictions, eval_metric_names)
  return metrics
