"""Temporary train script."""

import fire
from flax import linen as nn
from flax.training import train_state
import jax
import jax.numpy as jnp
import optax

from core.data import data_io


@jax.jit
def train_step(state, batch):
  """The on-device part of a train step."""
  def loss_fn(params):
    logits = model.apply(
        {'params': params},
        batch,
    )
    loss = jnp.mean(
        optax.softmax_cross_entropy(
            logits=logits, labels=jax.nn.one_hot(batch['label'], 10)))
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
    x = nn.Dense(feature=30)(x[:30])
    x = nn.relu(x)
    x = nn.Dense(feature=30)(x)
    x = nn.relu(x)
    x = nn.Dense(feature=30)(x)
    return x


def create_train_state(rng):
  """Creates initial TrainState."""
  model = MlpModel()
  params = model.init(rng, {'tokens': jnp.ones([30])})['params']
  tx = optax.sgd(0.03)
  return train_state.TrainState.create(
      apply_fn=model.apply, params=params, tx=tx)


def run_train():
  dataset = data_io.load_dataset().padded_batch(8)
  rng = jax.random.PRNGKey(0)

  rng, init_rng = jax.random.split(rng)
  state = create_train_state(init_rng)

  for batch in dataset:
    state, aux = train_step(state, batch)
    print(aux)

  return state


if __name__ == '__main__':
  fire.Fire()
