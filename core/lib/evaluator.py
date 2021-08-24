import jax
import jax.numpy as jnp
import numpy as np
import optax
import tensorflow_datasets as tfds

from core.data import error_kinds
from core.lib import evaluation
from core.lib import misc_utils
from core.lib import models

NUM_CLASSES = error_kinds.NUM_CLASSES


def evaluate_batch(batch, state, config):
  model = models.make_model(config)
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
      logits, batch['target'], config.eval_metric
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
    labels = jax.nn.one_hot(jnp.squeeze(batch['target'], axis=-1), NUM_CLASSES)
    predictions.append(jnp.argmax(logits, -1))
    ground_truth.append(batch['target'])
    loss.append(
        jnp.sum(
            optax.softmax_cross_entropy(
                logits=logits,
                labels=labels)
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
