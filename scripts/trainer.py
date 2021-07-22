"""Temporary train script."""

import fire
from flax import linen as nn
from flax.training import train_state
import jax
import jax.numpy as jnp
import optax
import tensorflow_datasets as tfds

from core.data import codenet_paths
from core.data import data_io
from core.data import error_kinds


DEFAULT_DATASET_PATH = codenet_paths.DEFAULT_DATASET_PATH
NUM_CLASSES = error_kinds.NUM_CLASSES


@jax.jit
def train_step(state, batch):
  """The on-device part of a train step."""
  model = MlpModel()

  def loss_fn(params):
    logits = model.apply(
        {'params': params},
        batch,
    )
    loss = jnp.mean(
        optax.softmax_cross_entropy(
            logits=logits,
            labels=jax.nn.one_hot(batch['target'], NUM_CLASSES)))
    return loss, {
        'logits': logits,
    }

  grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
  (_, aux), grads = grad_fn(state.params)
  state = state.apply_gradients(grads=grads)
  # TODO(dbieber): Optionally compute on-device metrics here.
  return state, {
      'logits': aux['logits'],
  }


class MlpModel(nn.Module):

  @nn.compact
  def __call__(self, x):
    x = x['tokens']
    x = nn.Embed(num_embeddings=30000, features=128)(x)
    x = nn.Dense(features=30)(x[:30])
    x = nn.relu(x)
    x = nn.Dense(features=30)(x)
    x = nn.relu(x)
    x = nn.Dense(features=30)(x)
    return x


def create_train_state(rng):
  """Creates initial TrainState."""
  model = MlpModel()
  variables = model.init(rng, {'tokens': jnp.ones([30], dtype=jnp.int64)})
  params = variables['params']
  tx = optax.sgd(0.03)
  return train_state.TrainState.create(
      apply_fn=model.apply, params=params, tx=tx)


def run_train(dataset_path=DEFAULT_DATASET_PATH):
  # Run 100 epochs.
  dataset = data_io.load_dataset(dataset_path).repeat(100).padded_batch(8)
  rng = jax.random.PRNGKey(0)

  rng, init_rng = jax.random.split(rng)
  state = create_train_state(init_rng)

  for batch in tfds.as_numpy(dataset):
    state, aux = train_step(state, batch)
    print(aux)

  return state


if __name__ == '__main__':
  fire.Fire()
