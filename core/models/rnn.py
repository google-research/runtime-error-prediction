from typing import Any

import jax
from flax import linen as nn
import jax.numpy as jnp

from core.modules.ipagnn import encoder
from core.modules.ipagnn import spans
from third_party.flax_examples import transformer_modules, lstm_modules


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
    lstm_config = lstm_modules.LSTMConfig(
      vocab_size=vocab_size, 
      num_layers=config.rnn_layers, 
      hidden_dim=config.hidden_size,)
    self.token_embedder = spans.NodeAwareTokenEmbedder(
        transformer_config=self.transformer_config,
        num_embeddings=vocab_size,
        features=config.hidden_size,
        max_tokens=max_tokens,
        max_num_nodes=max_num_nodes,
    )
    self.encoder = encoder.LSTMEncoder(lstm_config)


  @nn.compact
  def __call__(self, x):
    tokens = x['tokens']
    # tokens.shape: batch_size, max_tokens
    tokens_mask = tokens > 0
    # tokens_mask.shape: batch_size, max_tokens
    encoder_mask = nn.make_attention_mask(tokens_mask, tokens_mask, dtype=jnp.float32)
    # encoder_mask.shape: batch_size, 1, max_tokens, max_tokens
    # TODO(rgoel): Ensuring the token encoder is still a Transformer to ensure uniformity.
    encoded_inputs = self.token_embedder(
        tokens, x['node_token_span_starts'], x['node_token_span_ends'],
        x['num_nodes'])
    # encoded_inputs.shape: batch_size, max_tokens, hidden_size
    encoded_inputs = self.encoder(encoded_inputs)
    # encoded_inputs.shape: batch_size, max_tokens, hidden_size

    def get_last_state(inputs, last_token):
      return inputs[last_token-1]

    get_last_state_batch = jax.vmap(get_last_state)
    # encoded_inputs.shape: batch_size, max_tokens, hidden_size
    x = get_last_state_batch(encoded_inputs, x['num_tokens'])
    # x.shape: batch_size, 1, hidden_size
    x = jnp.squeeze(x, 1)
    # x.shape: batch_size, hidden_size
    x = nn.Dense(features=self.info.num_classes)(x)
    # x.shape: batch_size, num_classes
    return x, None
