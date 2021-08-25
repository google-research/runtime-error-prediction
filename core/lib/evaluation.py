import jax
import jax.numpy as jnp

from sklearn import metrics

NUM_CLASSES = error_kinds.NUM_CLASSES


def evaluate(targets, predictions, eval_metric):
  if eval_metric == 'F1-score':
    # TODO(dbieber): Support macro f1.
    return metrics.f1_score(targets, predictions, average='micro')
  elif eval_metric == 'Confusion matrix':
    return metrics.confusion_matrix(targets, predictions)
  raise ValueError(f'{eval_metric} is not implemented.')



def compute_metric(logits, targets, eval_metric):
  predictions = np.array(jnp.argmax(logits, -1))
  targets = np.array(targets)
  labels = jax.nn.one_hot(jnp.squeeze(targets, axis=-1), NUM_CLASSES)
  metric = evaluate(labels, predictions, eval_metric)
  return metric
