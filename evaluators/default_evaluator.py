import jax
import jax.numpy as jnp
import numpy as np
import optax
import tensorflow_datasets as tfds

from core.data import error_kinds
from lib import evaluation, misc_utils
from models import models_lib

NUM_CLASSES = error_kinds.NUM_CLASSES


def evaluate_batch(batch, state, config):
  model = models_lib.ModelFactory()(config.model.name)(config)
  # Potential bug in the following line. Dropout shouldn't be used
  # during the evaluation.
  new_rng, dropout_rng = jax.random.split(state.rng, 2)
  logits = model.apply(
    {'params': state.params},
    batch,
    rngs={'dropout': dropout_rng}
  )
  # logits = model.apply({"params": state.params}, batch, config)
  loss, metric = misc_utils.compute_metrics(
    logits, batch["target"], config.eval_metric
  )
  return logits, loss, metric


def evaluate(dataset, state, config):
  predictions = []
  ground_truth = []
  loss = []
  print(config.eval_metric)
  for batch in tfds.as_numpy(dataset):
    # logits = model.apply({'params': state.params}, batch, config)
    logits, _, _ = evaluate_batch(batch, state, config)
    predictions.append(jnp.argmax(logits, -1))
    ground_truth.append(batch["target"])
    loss.append(
      jnp.sum(
        optax.softmax_cross_entropy(
          logits=logits, labels=jax.nn.one_hot(batch["target"], NUM_CLASSES)
        )
      )
    )
  predictions = np.array(jnp.concatenate(predictions))
  ground_truth = np.array(jnp.concatenate(ground_truth)).flatten()
  eval_loss = sum(loss) / predictions.shape[0]
  assert predictions.shape[0] == ground_truth.shape[0]
  classification_score = evaluation.evaluate(
    ground_truth, predictions, config.eval_metric
  )
  return eval_loss, classification_score