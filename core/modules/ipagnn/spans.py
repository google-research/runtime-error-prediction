from typing import Any

from flax import linen as nn
import jax
import jax.numpy as jnp

from core.modules.ipagnn import encoder
from third_party.flax_examples import transformer_modules



def add_at_span(x, value, start, end):
  # start and end are inclusive.
  # x.shape: length, features
  # value.shape: features
  arange = jnp.arange(x.shape[0])
  # arange.shape: length
  mask = jnp.logical_and(start <= arange, arange <= end)
  # mask.shape: length
  return jnp.where(~mask[:, None], x, x + value[None, :])


def get_span_encoding_first(x, start, end):
  # x.shape: length, hidden_size
  # start and end are inclusive.
  return x[start]


def get_span_encoding_mean(x, start, end):
  # x.shape: length, hidden_size
  # start and end are inclusive.
  arange = jnp.arange(x.shape[0])
  # arange.shape: length
  mask = jnp.logical_and(start <= arange, arange <= end)
  # mask.shape: length
  values = jnp.where(mask[:, None], x, 0)
  total = jnp.sum(values, axis=0)
  # total.shape: hidden_size
  count = jnp.sum(mask)
  return total / count


def get_span_encoding_max(x, start, end):
  # x.shape: length, hidden_size
  # start and end are inclusive.
  arange = jnp.arange(x.shape[0])
  # arange.shape: length
  mask = jnp.logical_and(start <= arange, arange <= end)
  # mask.shape: length
  min_value = jnp.min(x, axis=0)
  # min_value.shape: hidden_size
  values = jnp.where(mask[:, None], x, min_value)
  # values.shape: length, hidden_size
  return jnp.max(values, axis=0)


def get_span_encoding_sum(x, start, end):
  # x.shape: length, hidden_size
  # start and end are inclusive.
  arange = jnp.arange(x.shape[0])
  # arange.shape: length
  mask = jnp.logical_and(start <= arange, arange <= end)
  values = jnp.where(mask[:, None], x, 0)
  return jnp.sum(values, axis=0)



class SpanIndexEncoder(nn.Module):
  """A "position encoder" for span indexes.

  Computes an embedding for each token indicating the nodes that the token is
  part of the span of. E.g. If token t is in the spans of nodes with index n_1
  and n_2, then the encoding of token t returned by the span index encoder is
  Embed(n_1) + Embed(n_2).

  The token contents themselves are not considered, only the node spans.
  """

  max_tokens: int
  max_num_nodes: int
  features: int

  def setup(self):
    self.embed = nn.Embed(
        num_embeddings=self.max_num_nodes,
        features=self.features,
        embedding_init=nn.initializers.normal(stddev=1.0),
    )

  def __call__(self, node_span_starts, node_span_ends):
    """Assume no batch dimension."""
    # node_span_starts.shape: num_nodes
    # node_span_ends.shape: num_nodes
    zeros = jnp.zeros((self.max_tokens, self.features))
    # zeros.shape: tokens, features
    indexes = jnp.arange(self.max_num_nodes)
    # indexes.shape: num_nodes
    embeddings = self.embed(indexes)
    # embeddings.shape: num_nodes, features

    def get_node_contribution(embedding, span_start, span_end):
      # embedding.shape: features
      # span_start: scalar
      # span_end: scalar
      return add_at_span(zeros, embedding, span_start, span_end)
    # vmap across the node dimension.
    per_node_contributions = jax.vmap(get_node_contribution)(
        embeddings, node_span_starts, node_span_ends)
    # per_node_contributions.shape: num_nodes, max_tokens, features

    # Sum across the node dimension.
    return jnp.sum(per_node_contributions, axis=0)


class NodeAwareTokenEmbedder(nn.Module):
  """Sums learned token-content embeddings and node span index embeddings.

  This includes adding position embeddings too.
  """

  transformer_config: transformer_modules.TransformerConfig
  num_embeddings: int
  features: int
  max_tokens: int
  max_num_nodes: int
  use_span_index_encoder: bool = False
  use_span_start_indicators: bool = False

  def setup(self):
    self.embed = nn.Embed(
        num_embeddings=self.num_embeddings,
        features=self.features,
        embedding_init=nn.initializers.normal(stddev=1.0))
    if self.use_span_index_encoder:
      self.span_index_encoder = SpanIndexEncoder(
          max_tokens=self.max_tokens,
          max_num_nodes=self.max_num_nodes,
          features=self.features
      )
    if self.use_span_start_indicators:
      self.span_start_embedding = self.param(
          'span_start_embedding',
          nn.initializers.variance_scaling(1.0, 'fan_in', 'normal', out_axis=0),
          (1, self.features,),
          jnp.float32
      )
    self.add_position_embeds = transformer_modules.AddPositionEmbs(
        config=self.transformer_config, decode=False, name='posembed_input')

  def __call__(self, tokens, node_span_starts, node_span_ends):
    x = tokens.astype('int32')
    # x.shape: batch_size, max_tokens
    x = self.embed(x)
    # x.shape: batch_size, max_tokens, features
    if self.use_span_index_encoder:
      token_span_encodings = jax.vmap(self.span_index_encoder)(
          node_span_starts, node_span_ends)
      # token_span_encodings.shape: batch_size, max_tokens, features
      x = x + token_span_encodings
    if self.use_span_start_indicators:
      # TODO(dbieber): Add indicator to start-of-span tokens.
      # TODO(dbieber): Be careful not to add start indicators to padding spans.
      raise NotImplementedError()
    # x.shape: batch_size, max_tokens, features
    x = self.add_position_embeds(x)
    return x


class NodeSpanEncoder(nn.Module):
  """Given tokens, nodes, and node spans in token space, encode each node."""

  info: Any
  config: Any

  max_tokens: int  # TODO(dbieber): Move these into info or config.
  max_num_nodes: int
  use_span_index_encoder: bool = False
  use_span_start_indicators: bool = False

  def setup(self):
    vocab_size = self.info.vocab_size
    hidden_size = self.config.hidden_size

    transformer_config = transformer_modules.TransformerConfig(
        vocab_size=vocab_size,
        output_vocab_size=vocab_size,
    )
    self.embed = NodeAwareTokenEmbedder(
        transformer_config=transformer_config,
        num_embeddings=vocab_size,
        features=hidden_size,
        max_tokens=self.max_tokens,
        max_num_nodes=self.max_num_nodes,
        use_span_index_encoder=self.use_span_index_encoder,
        use_span_start_indicators=self.use_span_start_indicators,
    )
    self.encoder = encoder.TransformerEncoder(config=transformer_config)

  def __call__(self, tokens, node_span_starts, node_span_ends):
    config = self.config

    # node_span_starts.shape: batch_size, num_nodes
    # node_span_ends.shape: batch_size, num_nodes
    # tokens.shape: batch_size, max_tokens
    token_embeddings = self.embed(tokens, node_span_starts, node_span_ends)
    # token_embeddings.shape: batch_size, max_tokens, hidden_size
    tokens_mask = tokens > 0
    # tokens_mask.shape: batch_size, max_tokens
    encoder_mask = nn.make_attention_mask(tokens_mask, tokens_mask, dtype=jnp.float32)
    # encoder_mask.shape: batch_size, 1, max_tokens, max_tokens
    encoding = self.encoder(token_embeddings, encoder_mask=encoder_mask)
    # encoding.shape: batch_size, max_tokens, hidden_size

    # Get just the encoding of the first token in each span.
    span_encoding_method = config.span_encoding_method
    if span_encoding_method == 'first':
      get_span_encoding_fn = get_span_encoding_first
    elif span_encoding_method == 'mean':
      get_span_encoding_fn = get_span_encoding_mean
    elif span_encoding_method == 'max':
      get_span_encoding_fn = get_span_encoding_max
    elif span_encoding_method == 'sum':
      get_span_encoding_fn = get_span_encoding_sum

    get_span_encoding_fn_node = jax.vmap(get_span_encoding_fn, in_axes=(None, 0, 0))
    get_span_encoding_fn_batch = jax.vmap(get_span_encoding_fn_node)
    span_encodings = get_span_encoding_fn_batch(
        encoding, node_span_starts, node_span_ends)

    # span_encodings.shape: batch_size, num_nodes, hidden_size
    return span_encodings
