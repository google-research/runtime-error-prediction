"""Temporary train script."""

import dataclasses
from typing import Any

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
from third_party.flax_examples import transformer_modules


DEFAULT_DATASET_PATH = codenet_paths.DEFAULT_DATASET_PATH
NUM_CLASSES = error_kinds.NUM_CLASSES


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


class TransformerEncoder(nn.Module):

  def setup(self):
    # tokenizer = tokenization.load_tokenizer()
    # vocab_size = tokenizer.vocab_size
    vocab_size = 30000
    transformer_config = transformer_modules.TransformerConfig(
        vocab_size=vocab_size,
        output_vocab_size=vocab_size,
    )
    self.encoder = transformer_modules.Encoder(transformer_config)

  @nn.compact
  def __call__(self, x):
    encoding = self.encoder(x['tokens'])
    # encoding.shape: batch_size, length, emb_dim
    x = encoding[:, 0, :]
    # x.shape: batch_size, emb_dim
    x = nn.Dense(features=NUM_CLASSES)(x)
    # x.shape: batch_size, NUM_CLASSES
    return x


def make_model():
  # model = MlpModel()
  model = TransformerEncoder()
  return model

model = make_model()


class TrainState(train_state.TrainState):
  rng: Any


@jax.jit
def train_step(state, batch):
  """The on-device part of a train step."""

  new_rng, dropout_rng = jax.random.split(state.rng, 2)
  state = dataclasses.replace(state, rng=new_rng)

  def loss_fn(params):
    logits = model.apply(
        {'params': params},
        batch,
        rngs={'dropout': dropout_rng}
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


def create_train_state(rng):
  """Creates initial TrainState."""
  fake_input = {'tokens': jnp.ones((8, 30,), dtype=jnp.int64)}
  rng, params_rng, dropout_rng = jax.random.split(rng, 3)
  variables = model.init(
      {'params': params_rng, 'dropout': dropout_rng},
      fake_input)
  params = variables['params']
  tx = optax.sgd(0.01)
  return TrainState.create(
      apply_fn=model.apply, params=params, tx=tx, rng=rng)


def run_train(dataset_path=DEFAULT_DATASET_PATH):
  # Run 100 epochs.
  dataset = data_io.load_dataset(dataset_path).repeat(1000).padded_batch(8)
  rng = jax.random.PRNGKey(0)

  rng, init_rng = jax.random.split(rng)
  state = create_train_state(init_rng)

  for step, batch in enumerate(tfds.as_numpy(dataset)):
    state, aux = train_step(state, batch)
    print(f'--- Step {step}')
    print(f"Loss: {aux['loss']}")
    print(f"Predictions: {jnp.argmax(aux['logits'], axis=-1)}")
    print(f"Targets: {batch['target'][:, 0]}")


if __name__ == '__main__':
  fire.Fire()
