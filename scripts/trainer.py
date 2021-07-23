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
  (loss, aux), grads = grad_fn(state.params)
  state = state.apply_gradients(grads=grads)
  # TODO(dbieber): Optionally compute on-device metrics here.
  return state, {
      'logits': aux['logits'],
      'loss': loss,
  }


class MlpModel(nn.Module):

  @nn.compact
  def __call__(self, x):
    x = x['tokens']
    # x.shape: batch_size, length
    batch_size = x.shape[0]
    x = nn.Embed(num_embeddings=30000, features=128)(x[:, :30])
    # x.shape: batch_size, 30, 128
    x = nn.Dense(features=30)(x)
    # x.shape: batch_size, 30, 30
    x = jnp.reshape(x, (batch_size, -1))
    # x.shape: batch_size, 900
    x = nn.relu(x)
    x = nn.Dense(features=30)(x)
    # x.shape: batch_size, 30
    x = nn.relu(x)
    x = nn.Dense(features=NUM_CLASSES)(x)
    # x.shape: batch_size, NUM_CLASSES
    return x


def create_train_state(rng):
  """Creates initial TrainState."""
  model = MlpModel()
  fake_input = {'tokens': jnp.ones((8, 30,), dtype=jnp.int64)}
  variables = model.init(rng, fake_input)
  params = variables['params']
  tx = optax.sgd(0.03)
  return train_state.TrainState.create(
      apply_fn=model.apply, params=params, tx=tx)


def run_train(dataset_path=DEFAULT_DATASET_PATH):
  # Run 100 epochs.
  dataset = data_io.load_dataset(dataset_path).repeat(1000).padded_batch(8)
  rng = jax.random.PRNGKey(0)

  rng, init_rng = jax.random.split(rng)
  state = create_train_state(init_rng)

  for batch in tfds.as_numpy(dataset):
    state, aux = train_step(state, batch)
    print('---')
    print(f"Loss: {aux['loss']}")
    print(f"Predictions: {jnp.argmax(aux['logits'], axis=-1)}")
    print(f"Targets: {batch['target']}")


if __name__ == '__main__':
  fire.Fire()
