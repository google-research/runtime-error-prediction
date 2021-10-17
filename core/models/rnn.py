from typing import Any

from flax import linen as nn
import jax.numpy as jnp

from core.modules.ipagnn import encoder
from core.modules.ipagnn import spans
from third_party.flax_examples import transformer_modules, lstm_modules


class LSTM(nn.Module):

  config: Any
  info: Any
  transformer_config: transformer_modules.TransformerConfig
  lstm_config: lstm_modules.LSTMConfig

  def setup(self):
    config = self.config
    vocab_size = self.info.vocab_size
    max_tokens = config.max_tokens
    max_num_nodes = config.max_num_nodes
    max_num_edges = config.max_num_edges
    self.token_embedder = spans.NodeAwareTokenEmbedder(
        transformer_config=self.transformer_config,
        num_embeddings=vocab_size,
        features=config.hidden_size,
        max_tokens=max_tokens,
        max_num_nodes=max_num_nodes,
    )
    self.encoder = encoder.LSTMEncoder(
        self.lstm_config)

  def get_end_token(self, inputs, num_tokens):
    return inputs[:, num_tokens]

  @nn.compact
  def __call__(self, x):
    tokens = x['tokens']
    # tokens.shape: batch_size, max_tokens
    tokens_mask = tokens > 0
    # tokens_mask.shape: batch_size, max_tokens
    encoder_mask = nn.make_attention_mask(tokens_mask, tokens_mask, dtype=jnp.float32)
    # encoder_mask.shape: batch_size, 1, max_tokens, max_tokens
    encoded_inputs = self.token_embedder(
        tokens, x['node_token_span_starts'], x['node_token_span_ends'],
        x['num_nodes'])
    # encoded_inputs.shape: batch_size, max_tokens, hidden_size
    encoded_inputs = self.encoder(encoded_inputs)
    # encoded_inputs.shape: batch_size, max_tokens, hidden_size
    x = self.get_end_token(encoded_inputs, x['num_tokens'])
    # x.shape: batch_size, hidden_size
    x = nn.Dense(features=self.info.num_classes)(x)
    # x.shape: batch_size, num_classes
    return x, None
