from typing import Any

from flax import linen as nn
import jax
import jax.numpy as jnp

from core.models.ipagnn import encoder
from third_party.flax_examples import transformer_modules



def add_at_span(x, value, start, end):
  # Inclusive [start, end]
  # x.shape: length, features
  # value.shape: features,
  arange = jnp.arange(x.shape[0])
  # arange.shape: length
  mask = jnp.logical_and(start <= arange, arange <= end)
  # mask.shape: length
  return jnp.where(~mask[:, None], x, x + value[None, :])


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
    # node_span_starts.shape: num_nodes,
    # node_span_ends.shape: num_nodes,
    zeros = jnp.zeros((self.max_tokens, self.features))
    # zeros.shape: tokens, features
    indexes = jnp.arange(self.max_num_nodes)
    # indexes.shape: num_nodes,
    embeddings = self.embed(indexes)
    # embeddings.shape: num_nodes, features

    def get_node_contribution(embedding, span_start, span_end):
      # embedding.shape: features,
      # span_start: scalar
      # span_end: scalar
      return add_at_span(zeros, embedding, span_start, span_end)
    # vmap across the node dimension.
    print('embeddings.shape')
    print(embeddings.shape)
    print(node_span_starts.shape)
    print(node_span_ends.shape)
    per_node_contributions = jax.vmap(get_node_contribution)(
        embeddings, node_span_starts, node_span_ends)
    # per_node_contributions.shape: num_nodes, max_tokens, features

    # Sum across the node dimension.
    return jnp.sum(per_node_contributions, axis=0)


class NodeAwareTokenEmbedder(nn.Module):
  """Sums learned token-content embeddings and node span index embeddings.

  This does not include adding position embeddings.
  """

  transformer_config: transformer_modules.TransformerConfig
  num_embeddings: int
  features: int
  max_tokens: int
  max_num_nodes: int

  def setup(self):
    self.embed = nn.Embed(
        num_embeddings=self.num_embeddings,
        features=self.features,
        embedding_init=nn.initializers.normal(stddev=1.0))
    self.span_index_encoder = SpanIndexEncoder(
        max_tokens=self.max_tokens,
        max_num_nodes=self.max_num_nodes,
        features=self.features
    )
    self.add_position_embeds = transformer_modules.AddPositionEmbs(
        config=self.transformer_config, decode=False, name='posembed_input')

  def __call__(self, tokens, node_span_starts, node_span_ends):
    x = tokens.astype('int32')
    # x.shape: batch_size, max_tokens
    x = self.embed(x)
    # x.shape: batch_size, max_tokens, features
    token_span_encodings = jax.vmap(self.span_index_encoder)(
        node_span_starts, node_span_ends)
    # token_span_encodings.shape: batch_size, max_tokens, features
    x = x + token_span_encodings
    # x.shape: batch_size, max_tokens, features
    x = self.add_position_embeds(x)
    return x


class NodeSpanEncoder(nn.Module):
  """Given tokens, nodes, and node spans in token space, encode each node."""

  info: Any
  config: Any

  max_tokens: int  # TODO(dbieber): Move these info info or config.
  max_num_nodes: int

  def setup(self):
    vocab_size = self.info.vocab_size
    hidden_size = self.config.model.hidden_size

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
    )
    self.transformer = encoder.TransformerEncoder(config=transformer_config)

  def __call__(self, tokens, node_span_starts, node_span_ends):
    # tokens.shape: batch_size, length
    token_embeddings = self.embed(tokens, node_span_starts, node_span_ends)
    # token_embeddings.shape: batch_size, length, hidden_size
    encoding = self.transformer(token_embeddings)
    # encoding.shape: batch_size, length, hidden_size

    # Get just the encoding of the first token in each span.
    span_encodings = jax.vmap(lambda a, b: a[b])(encoding, node_span_starts)
    # span_encodings.shape: batch_size, num_nodes, hidden_size
    return span_encodings
