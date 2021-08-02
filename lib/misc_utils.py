import os

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax.training import checkpoints, early_stopping

from core.data import error_kinds
from lib import evaluation

NUM_CLASSES = error_kinds.NUM_CLASSES


def save_checkpoint(state, workdir):
    os.makedirs(workdir, exist_ok=True)
    step = int(state.step)
    checkpoints.save_checkpoint(workdir, state, step, keep=3)


def compute_metrics(logits, ground_truth, eval_metric):
    predictions = np.array(jnp.argmax(logits, -1))
    ground_truth = np.array(ground_truth)
    metric = evaluation.evaluate(ground_truth, predictions, eval_metric)
    loss = jnp.mean(
        optax.softmax_cross_entropy(
            logits=logits, labels=jax.nn.one_hot(ground_truth, NUM_CLASSES)
        )
    )
    return loss, metric
