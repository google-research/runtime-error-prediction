# Copyright (C) 2021 Google Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Gated Graph Neural Network."""

from typing import Any

from absl import logging  # pylint: disable=unused-import
from flax import linen as nn
import jax
from jax import lax
import jax.numpy as jnp

from core.modules.ipagnn import encoder
from core.modules.ipagnn import rnn
from core.modules.ipagnn import spans
from third_party.flax_examples import transformer_modules

Embed = nn.Embed
StackedRNNCell = rnn.StackedRNNCell


def create_lstm_cells(n):
  """Creates a list of n LSTM cells."""
  cells = []
  for i in range(n):
    cell = nn.LSTMCell(
        gate_fn=nn.sigmoid,
        activation_fn=nn.tanh,
        kernel_init=nn.initializers.xavier_uniform(),
        recurrent_kernel_init=nn.initializers.orthogonal(),
        bias_init=nn.initializers.zeros,
        name=f'lstm_{i}',
    )
    cells.append(cell)
  return cells


class GGNNLayer(nn.Module):
  """A single layer of GGNN message passing."""

  info: Any
  config: Any
  num_nodes: Any
  hidden_size: Any

  @nn.compact
  def __call__(
      self,
      node_embeddings,
      source_indices,
      dest_indices,
      edge_types,
      num_edges):
    """Apply GGNN layer."""
    config = self.config
    num_nodes = self.num_nodes
    hidden_size = self.hidden_size

    gru_cell = nn.recurrent.GRUCell(name='gru_cell')

    num_edge_types = 6
    max_num_edges = edge_types.shape[0]
    # num_edges.shape: scalar
    assert num_edges.shape == ()
    edge_dense = nn.Dense(  # Used for creating key/query/values.
        name='edge_dense',
        features=num_edge_types * hidden_size,
        kernel_init=nn.initializers.xavier_uniform(),
        bias_init=nn.initializers.normal(stddev=1e-6))

    # node_embeddings.shape: num_nodes, hidden_size
    source_embeddings = node_embeddings[source_indices]
    # source_embeddings.shape: max_num_edges, hidden_size

    new_source_embeddings_all_types = edge_dense(source_embeddings)
    # new_source_embeddings_all_types.shape:
    #   max_num_edges, (num_edge_types*hidden_size)
    new_source_embeddings_by_type = (
        new_source_embeddings_all_types.reshape(
            (-1, num_edge_types, hidden_size)))
    # new_source_embeddings_by_type.shape:
    #    max_num_edges, num_edge_types, hidden_size
    new_source_embeddings = (
        new_source_embeddings_by_type[jnp.arange(max_num_edges), edge_types, :])
    # new_source_embeddings.shape: max_num_edges, hidden_size

    # Set new_source_embeddings to zero for all edges beyond the last edge.
    new_source_embeddings = jnp.where(
        (jnp.arange(max_num_edges) < num_edges)[:, None],
        new_source_embeddings,
        0
    )

    proposed_node_embeddings = jax.ops.segment_sum(
        data=new_source_embeddings,
        segment_ids=dest_indices,
        num_segments=num_nodes,
    )
    # proposed_node_embeddings.shape: num_nodes, hidden_state

    _, outputs = gru_cell(proposed_node_embeddings, node_embeddings)
    return outputs


class GGNNModule(nn.Module):
  """GGNN model. Operates on batches of batch_size."""

  config: Any
  info: Any

  @nn.compact
  def __call__(
      self,
      node_embeddings,
      docstring_embeddings,
      docstring_mask,
      edge_sources,
      edge_dests,
      edge_types,
      num_edges,
      true_indexes,
      false_indexes,
      raise_indexes,
      start_node_indexes,
      exit_node_indexes,
      step_limits):
    config = self.config
    info = self.info

    # start_node_indexes.shape: batch_size, 1
    # exit_node_indexes.shape: batch_size, 1
    start_node_indexes = jnp.squeeze(start_node_indexes, axis=-1)
    exit_node_indexes = jnp.squeeze(exit_node_indexes, axis=-1)
    # start_node_indexes.shape: batch_size
    # exit_node_indexes.shape: batch_size
    # num_edges.shape: batch_size
    batch_size = node_embeddings.shape[0]
    num_edges = jnp.squeeze(num_edges, axis=-1)
    assert num_edges.shape == (batch_size,)

    start_indexes = start_node_indexes
    exit_indexes = exit_node_indexes
    steps_all = jnp.squeeze(step_limits, axis=-1)
    # steps_all.shape: batch_size
    # edge_types = edge_types
    source_indices = edge_sources
    dest_indices = edge_dests
    vocab_size = info.vocab_size
    output_token_vocabulary_size = info.num_classes
    hidden_size = config.hidden_size
    num_nodes = node_embeddings.shape[1]

    max_steps = config.max_steps

    output_dense = nn.Dense(
        name='output_dense',
        features=output_token_vocabulary_size,
        kernel_init=nn.initializers.xavier_uniform(),
        bias_init=nn.initializers.normal(stddev=1e-6))

    # node_embeddings.shape:
    #     batch_size, num_nodes, statement_length, hidden_size

    gnn_layer_single_example = GGNNLayer(
        info=info,
        config=config,
        num_nodes=num_nodes,
        hidden_size=hidden_size,
    )
    gnn_layer = jax.vmap(gnn_layer_single_example)

    # node_embeddings.shape: batch_size, num_nodes, hidden_size
    if config.ggnn_use_fixed_num_layers:
      for step in range(config.ggnn_layers):
        node_embeddings = gnn_layer(
            node_embeddings,
            source_indices,
            dest_indices,
            edge_types,
            num_edges)
    else:
      # Run one layer per allowed step of execution.
      for step in range(max_steps):
        new_node_embeddings = gnn_layer(
            node_embeddings,
            source_indices,
            dest_indices,
            edge_types,
            num_edges)
        # steps_all.shape: batch_size
        valid = jnp.expand_dims(step < steps_all, axis=(1, 2))
        # valid.shape: batch_size, 1, 1
        node_embeddings = jnp.where(
            valid,
            new_node_embeddings,
            node_embeddings)

    def get_final_state(node_embeddings, exit_index):
      if config.ggnn_use_exit_node_embedding:
        return node_embeddings[exit_index]
      else:
        # Mean Pool over num_nodes (only up to the actual num nodes, given by exit_index + 1)
        is_node = jnp.arange(num_nodes) < exit_index
        print(is_node.shape)
        print(node_embeddings.shape)
        node_embeddings = jnp.where(is_node[:, None], node_embeddings, 0)
        # set to 0 anything beyond exit_index
        # sum and divide.
        global_embedding = jnp.sum(node_embeddings, axis=0) / (exit_index + 1)
        assert global_embedding.shape == (hidden_size,)
        return global_embedding
    # node_embeddings.shape: batch_size, num_nodes, hidden_size
    final_states = jax.vmap(get_final_state)(node_embeddings, exit_indexes)
    # final_states.shape: batch_size, hidden_size

    logits = output_dense(final_states)
    # logits.shape: batch_size, output_token_vocabulary_size
    return logits


class GGNN(nn.Module):

  config: Any
  info: Any
  transformer_config: transformer_modules.TransformerConfig
  docstring_transformer_config: transformer_modules.TransformerConfig

  def setup(self):
    config = self.config
    vocab_size = self.info.vocab_size
    max_tokens = config.max_tokens
    max_num_nodes = config.max_num_nodes
    max_num_edges = config.max_num_edges
    max_steps = config.max_steps
    self.node_span_encoder = spans.NodeSpanEncoder(
        info=self.info,
        config=config,
        transformer_config=self.transformer_config,
        max_tokens=max_tokens,
        max_num_nodes=max_num_nodes,
        use_span_index_encoder=False,
        use_span_start_indicators=False,
    )
    if config.use_film or config.use_cross_attention:
      self.docstring_token_encoder = encoder.TokenEncoder(
          transformer_config=self.docstring_transformer_config,
          num_embeddings=vocab_size,
          features=config.hidden_size,
      )
      self.docstring_encoder = encoder.TransformerEncoder(
          config=self.docstring_transformer_config)

    self.ggnn = GGNNModule(
        info=self.info,
        config=config,
    )

  @nn.compact
  def __call__(self, x):
    config = self.config
    info = self.info
    tokens = x['tokens']
    docstring_tokens = x['docstring_tokens']
    # tokens.shape: batch_size, max_tokens
    batch_size = tokens.shape[0]
    encoded_inputs = self.node_span_encoder(
        tokens, x['node_token_span_starts'], x['node_token_span_ends'],
        x['num_nodes'])
    # encoded_inputs.shape: batch_size, max_num_nodes, hidden_size
    if config.use_film or config.use_cross_attention:
      docstring_token_embeddings = self.docstring_token_encoder(
          docstring_tokens)
      docstring_mask = docstring_tokens > 0
      docstring_encoder_mask = nn.make_attention_mask(
          docstring_mask, docstring_mask, dtype=jnp.float32)
      # docstring_token_embeddings.shape: batch_size, max_tokens, hidden_size
      docstring_embeddings = self.docstring_encoder(
          docstring_token_embeddings,
          encoder_mask=docstring_encoder_mask)
    else:
      docstring_embeddings = None
      docstring_mask = None
    logits = self.ggnn(
        node_embeddings=encoded_inputs,
        docstring_embeddings=docstring_embeddings,
        docstring_mask=docstring_mask,
        edge_sources=x['edge_sources'],
        edge_dests=x['edge_dests'],
        edge_types=x['edge_types'],
        num_edges=x['edge_sources_shape'],
        true_indexes=x['true_branch_nodes'],
        false_indexes=x['false_branch_nodes'],
        raise_indexes=x['raise_nodes'],
        start_node_indexes=x['start_index'],
        exit_node_indexes=x['exit_index'],
        step_limits=x['step_limit'],
    )
    return logits, {}
