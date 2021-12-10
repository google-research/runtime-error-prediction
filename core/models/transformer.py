from typing import Any

from flax import linen as nn
import jax.numpy as jnp

from core.modules.ipagnn import encoder
from core.modules.ipagnn import spans
from third_party.flax_examples import transformer_modules


class Transformer(nn.Module):

  config: Any
  info: Any
  transformer_config: transformer_modules.TransformerConfig

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
    self.encoder = encoder.TransformerEncoder(
        self.transformer_config)

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
    encoding = self.encoder(encoded_inputs, encoder_mask=encoder_mask)
    # encoding.shape: batch_size, max_tokens, hidden_size

    # Mean pooling.
    tokens_mask_ext = tokens_mask[:, :, None]
    x = (
        jnp.sum(encoding * tokens_mask_ext, axis=1)
        / jnp.maximum(1, jnp.sum(tokens_mask_ext, axis=1))
    )
    # x.shape: batch_size, hidden_size
    x = nn.Dense(features=self.info.num_classes)(x)
    # x.shape: batch_size, num_classes
    return x, None


class MILTransformer(nn.Module):
  """NodeSpanEncoder can give contextual or non-contextual span embeddings.
  Then from each span embedding, we predict P(class|span,input).
  We then aggregate these into P(class|input) predictions.
  And we return overall predictions, as well as localization_logits.
  """

  config: Any
  info: Any
  transformer_config: transformer_modules.TransformerConfig

  def setup(self):
    config = self.config
    max_tokens = config.max_tokens
    max_num_nodes = config.max_num_nodes
    self.node_span_encoder = spans.NodeSpanEncoder(
        info=self.info,
        config=config,
        transformer_config=self.transformer_config,
        max_tokens=max_tokens,
        max_num_nodes=max_num_nodes,
        use_span_index_encoder=False,
        use_span_start_indicators=False,
    )

  @nn.compact
  def __call__(self, x):
    config = self.config
    info = self.info
    tokens = x['tokens']
    # tokens.shape: batch_size, max_tokens
    batch_size = tokens.shape[0]
    encoded_inputs = self.node_span_encoder(
        tokens, x['node_token_span_starts'], x['node_token_span_ends'],
        x['num_nodes'])
    # encoded_inputs.shape: batch_size, max_num_nodes, hidden_size

    num_classes = info.num_classes
    line_level_logits = nn.Dense(
        features=num_classes, name='per_line_output'
    )(encoded_inputs)
    # line_level_logits.shape: batch_size, max_num_nodes, num_classes

    aux = {}
    if config.mil_pool == 'max':
      logits = jnp.max(line_level_logits, axis=1)
      # logits.shape: batch_size, num_classes
      aux['localization_logits']
    raise NotImplementedError()
    return logits, aux
