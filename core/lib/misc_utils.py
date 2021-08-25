import os

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax.training import checkpoints, early_stopping

from core.data import error_kinds
from core.lib import evaluation

NUM_CLASSES = error_kinds.NUM_CLASSES


def compute_metric(logits, ground_truth, eval_metric):
  # TODO(dbieber): Rename file.
  predictions = np.array(jnp.argmax(logits, -1))
  ground_truth = np.array(ground_truth)
  labels = jax.nn.one_hot(jnp.squeeze(ground_truth, axis=-1), NUM_CLASSES)
  metric = evaluation.evaluate(ground_truth, predictions, eval_metric)
  return metric
