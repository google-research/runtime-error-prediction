from typing import Any

import jax
from flax import linen as nn
import jax.numpy as jnp

from core.modules.rnn import encoder
from core.modules.ipagnn import spans
from third_party.flax_examples import transformer_modules


class LSTM(nn.Module):

  config: Any
  info: Any
  transformer_config: transformer_modules.TransformerConfig

  def setup(self):
    config = self.config
    vocab_size = self.info.vocab_size
    max_tokens = config.max_tokens
    max_num_nodes = config.max_num_nodes
    max_num_edges = config.max_num_edges
    input_embedder_type = config.rnn_input_embedder_type
    if input_embedder_type=="token":
      self.input_embedder = spans.NodeAwareTokenEmbedder(
          transformer_config=self.transformer_config,
          num_embeddings=vocab_size,
          features=config.hidden_size,
          max_tokens=max_tokens,
          max_num_nodes=max_num_nodes,
      )
    else:
      self.input_embedder = spans.NodeSpanEncoder(
          info=self.info,
          config=config,
          transformer_config=self.transformer_config,
          max_tokens=max_tokens,
          max_num_nodes=max_num_nodes,
          use_span_index_encoder=False,
          use_span_start_indicators=False,
      )
    self.encoder = encoder.LSTMEncoder(num_layers=config.rnn_layers,
      hidden_dim=config.hidden_size)

  @nn.compact
  def __call__(self, x):
    tokens = x['tokens']
    # tokens.shape: batch_size, max_tokens
    tokens_mask = tokens > 0
    # tokens_mask.shape: batch_size, max_tokens
    encoder_mask = nn.make_attention_mask(tokens_mask, tokens_mask, dtype=jnp.float32)
    # encoder_mask.shape: batch_size, 1, max_tokens, max_tokens
    # NOTE(rgoel): Ensuring the token encoder is still a Transformer to ensure uniformity.
    encoded_inputs = self.input_embedder(
        tokens, x['node_token_span_starts'], x['node_token_span_ends'],
        x['num_nodes'])
    # encoded_inputs.shape: batch_size, max_tokens, hidden_size
    encoded_inputs = self.encoder(encoded_inputs)
    # encoded_inputs.shape: batch_size, max_tokens, hidden_size

    # NOTE(rgoel): Using only the last state. We can change this to
    # pooling across time stamps.
    def get_last_state(inputs, last_token):
      return inputs[last_token-1]

    get_last_state_batch = jax.vmap(get_last_state)

    if input_embedder_type=="token":
      x = get_last_state_batch(encoded_inputs, x['num_tokens'])
    else:
      x = get_last_state_batch(encoded_inputs, x['num_nodes'])
    # x.shape: batch_size, 1, hidden_size
    x = jnp.squeeze(x, axis=1)
    # x.shape: batch_size, hidden_size
    x = nn.Dense(features=self.info.num_classes)(x)
    # x.shape: batch_size, num_classes
    return x, None
