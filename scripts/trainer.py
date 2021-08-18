"""Temporary train script."""

import dataclasses
from typing import Any

import fire
from flax import linen as nn
from flax.training import train_state
import jax
import jax.numpy as jnp
import ml_collections
import optax
import tensorflow_datasets as tfds

from core.data import codenet_paths
from core.data import data_io
from core.data import error_kinds
from core.models.ipagnn import encoder
from core.models.ipagnn import ipagnn
from core.models.ipagnn import spans
from third_party.flax_examples import transformer_modules


DEFAULT_DATASET_PATH = codenet_paths.DEFAULT_DATASET_PATH
NUM_CLASSES = error_kinds.NUM_CLASSES

Config = ml_collections.ConfigDict


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


class Transformer(nn.Module):

  config: Any

  def setup(self):
    config = self.config
    vocab_size = 30000  # TODO(dbieber): Load from tokenizer / info.
    max_tokens = 896
    max_num_nodes = 80
    max_num_edges = 160
    info = ipagnn.Info(vocab_size=vocab_size)
    transformer_config = transformer_modules.TransformerConfig(
        vocab_size=vocab_size,
        output_vocab_size=vocab_size,
    )
    self.token_embedder = spans.NodeAwareTokenEmbedder(
        transformer_config=transformer_config,
        num_embeddings=vocab_size,
        features=config.model.hidden_size,
        max_tokens=max_tokens,
        max_num_nodes=max_num_nodes,
    )
    self.encoder = encoder.TransformerEncoder(transformer_config)

  @nn.compact
  def __call__(self, x):
    tokens = x['tokens']
    # tokens.shape: batch_size, max_tokens
    tokens_mask = tokens > 0
    # tokens_mask.shape: batch_size, max_tokens
    encoder_mask = nn.make_attention_mask(tokens_mask, tokens_mask, dtype=jnp.float32)
    # encoder_mask.shape: batch_size, 1, max_tokens, max_tokens
    encoded_inputs = self.token_embedder(
        tokens, x['node_token_span_starts'], x['node_token_span_ends'])
    # encoded_inputs.shape: batch_size, max_tokens, hidden_size
    encoding = self.encoder(encoded_inputs, encoder_mask=encoder_mask)
    # encoding.shape: batch_size, max_tokens, hidden_size

    # Mean pooling.
    tokens_mask_ext = tokens_mask[:, :, None]
    x = (
        jnp.sum(encoding * tokens_mask_ext, axis=1)
        / jnp.maximum(1, jnp.sum(tokens_mask_ext, axis=1))
    )
    # x.shape: batch_size, hidden_size
    x = nn.Dense(features=NUM_CLASSES)(x)
    # x.shape: batch_size, NUM_CLASSES
    return x


class TransformerBaseline(nn.Module):

  config: Any

  def setup(self):
    config = self.config
    vocab_size = 30000  # TODO(dbieber): Load from tokenizer / info.
    max_tokens = 896
    max_num_nodes = 80
    max_num_edges = 160
    info = ipagnn.Info(vocab_size=vocab_size)
    transformer_config = transformer_modules.TransformerConfig(
        vocab_size=vocab_size,
        output_vocab_size=vocab_size,
    )
    self.transformer = transformer_modules.Transformer(transformer_config)

  @nn.compact
  def __call__(self, x):
    tokens = x['tokens']
    # tokens.shape: batch_size, max_tokens
    tokens_mask = tokens > 0
    # tokens_mask.shape: batch_size, max_tokens
    # encoder_mask.shape: batch_size, 1, max_tokens, max_tokens
    # encoded_inputs.shape: batch_size, max_tokens, hidden_size
    encoding = self.transformer.encoder(tokens)
    # encoding.shape: batch_size, max_tokens, hidden_size

    # Mean pooling.
    tokens_mask_ext = tokens_mask[:, :, None]
    x = (
        jnp.sum(encoding * tokens_mask_ext, axis=1)
        / jnp.maximum(1, jnp.sum(tokens_mask_ext, axis=1))
    )
    # x.shape: batch_size, hidden_size
    x = nn.Dense(features=NUM_CLASSES)(x)
    # x.shape: batch_size, NUM_CLASSES
    return x


class IPAGNN(nn.Module):

  config: Any

  def setup(self):
    config = self.config
    vocab_size = 30000  # TODO(dbieber): Load from tokenizer / info.
    max_tokens = 896
    max_num_nodes = 80
    max_num_edges = 160
    max_steps = 80
    info = ipagnn.Info(vocab_size=vocab_size)
    transformer_config = transformer_modules.TransformerConfig(
        vocab_size=vocab_size,
        output_vocab_size=vocab_size,
    )
    self.node_span_encoder = spans.NodeSpanEncoder(
        info=info,
        config=config,
        max_tokens=max_tokens,
        max_num_nodes=max_num_nodes,
    )

    self.ipagnn = ipagnn.IPAGNN(
        info=info,
        config=config,
        max_steps=max_steps,
    )

  @nn.compact
  def __call__(self, x):
    tokens = x['tokens']
    # tokens.shape: batch_size, max_tokens
    batch_size = tokens.shape[0]
    encoded_inputs = self.node_span_encoder(
        tokens, x['node_token_span_starts'], x['node_token_span_ends'])
    # encoded_inputs.shape: batch_size, max_num_nodes, hidden_size
    # TODO(dbieber): Compute number of steps in dataset.
    all_steps = jnp.array([70] * batch_size)
    ipagnn_output = self.ipagnn(
        node_embeddings=encoded_inputs,
        edge_sources=x['edge_sources'],
        edge_dests=x['edge_dests'],
        edge_types=x['edge_types'],
        true_indexes=x['true_branch_nodes'],
        false_indexes=x['false_branch_nodes'],
        exit_indexes=x['exit_index'],
        all_steps=all_steps,
    )
    # ipagnn_output['node_embeddings'].shape: batch_size, max_num_nodes, hidden_size
    # ipagnn_output['instruction_pointer'].shape: batch_size, max_num_nodes
    # ipagnn_output['exit_node_embeddings'].shape: batch_size, hidden_size
    # ipagnn_output['exception_node_embeddings'].shape: batch_size, hidden_size

    # TODO(dbieber): Reevaluate how to go from transformer encodings to output.
    exit_node_embeddings = ipagnn_output['exit_node_embeddings']
    print("x['exit_index']")
    print(x['exit_index'])
    print(exit_node_embeddings)
    # exit_node_embeddings.shape: batch_size, emb_dim
    logits = nn.Dense(features=NUM_CLASSES)(exit_node_embeddings)
    # logits.shape: batch_size, NUM_CLASSES
    return logits


def make_sample_config():
  config = Config()
  config.model = Config()
  config.model.hidden_size = 10
  config.model.rnn_layers = 2
  config.model.checkpoint = False
  return config


def make_model():
  config = make_sample_config()
  # model = MlpModel()
  model = Transformer(config=config)
  # model = IPAGNN(config=config)
  return model


class TrainState(train_state.TrainState):
  rng: Any


@jax.jit
def train_step(state, batch):
  """The on-device part of a train step."""
  model = make_model()

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


def create_train_state(rng, model):
  """Creates initial TrainState."""
  batch_size = 8
  max_tokens = 896
  max_num_nodes = 80
  max_num_edges = 160
  fake_input = data_io.get_fake_input(
      batch_size, max_tokens, max_num_nodes, max_num_edges)
  rng, params_rng, dropout_rng = jax.random.split(rng, 3)
  variables = model.init(
      {'params': params_rng, 'dropout': dropout_rng},
      fake_input)
  params = variables['params']
  learning_rate = 0.03
  tx = optax.sgd(learning_rate)
  return TrainState.create(
      apply_fn=model.apply, params=params, tx=tx, rng=rng)


def load_dataset(dataset_path=DEFAULT_DATASET_PATH, split='train'):
  epochs = 1000
  batch_size = 8
  max_tokens = 896
  max_num_nodes = 80
  max_num_edges = 160
  padded_shapes = data_io.get_padded_shapes(
      max_tokens, max_num_nodes, max_num_edges)
  filter_fn = data_io.make_filter(max_tokens, max_num_nodes, max_num_edges)
  return (
      data_io.load_dataset(dataset_path, split=split)
      .repeat(epochs)
      .filter(filter_fn)
      .padded_batch(batch_size, padded_shapes=padded_shapes)
  )


def run_train(dataset_path=DEFAULT_DATASET_PATH):
  print(f'Training on data: {dataset_path}')
  dataset = load_dataset(dataset_path)
  rng = jax.random.PRNGKey(0)

  rng, init_rng = jax.random.split(rng)
  model = make_model()
  state = create_train_state(init_rng, model)

  for step, batch in enumerate(tfds.as_numpy(dataset)):
    state, aux = train_step(state, batch)
    print(f'--- Step {step}')
    print(f"Loss: {aux['loss']}")
    print(f"Predictions: {jnp.argmax(aux['logits'], axis=-1)}")
    print(f"Targets: {batch['target'][:, 0]}")


if __name__ == '__main__':
  fire.Fire()
