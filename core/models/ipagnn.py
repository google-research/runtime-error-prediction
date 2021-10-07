"""IPA-GNN models."""

from typing import Any

from flax import linen as nn
import jax
import jax.numpy as jnp

from core.data import error_kinds
from core.modules.ipagnn import ipagnn
from core.modules.ipagnn import logit_math
from core.modules.ipagnn import spans
from core.modules.ipagnn import raise_contributions as raise_contributions_lib
from third_party.flax_examples import transformer_modules


class IPAGNN(nn.Module):

  config: Any
  info: Any
  transformer_config: transformer_modules.TransformerConfig

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

    self.ipagnn = ipagnn.IPAGNNModule(
        info=self.info,
        config=config,
        max_steps=max_steps,
    )

  @nn.compact
  def __call__(self, x):
    config = self.config
    tokens = x['tokens']
    # tokens.shape: batch_size, max_tokens
    batch_size = tokens.shape[0]
    encoded_inputs = self.node_span_encoder(
        tokens, x['node_token_span_starts'], x['node_token_span_ends'],
        x['num_nodes'])
    # encoded_inputs.shape: batch_size, max_num_nodes, hidden_size
    ipagnn_output = self.ipagnn(
        node_embeddings=encoded_inputs,
        edge_sources=x['edge_sources'],
        edge_dests=x['edge_dests'],
        edge_types=x['edge_types'],
        true_indexes=x['true_branch_nodes'],
        false_indexes=x['false_branch_nodes'],
        raise_indexes=x['raise_nodes'],
        start_node_indexes=x['start_index'],
        exit_node_indexes=x['exit_index'],
        step_limits=x['step_limit'],
    )
    # ipagnn_output['exit_node_embeddings'].shape: batch_size, hidden_size
    # ipagnn_output['raise_node_embeddings'].shape: batch_size, hidden_size
    # ipagnn_output['exit_node_instruction_pointer'].shape: batch_size
    # ipagnn_output['raise_node_instruction_pointer'].shape: batch_size

    exit_node_embeddings = ipagnn_output['exit_node_embeddings']
    # exit_node_embeddings.shape: batch_size, hidden_size
    raise_node_embeddings = ipagnn_output['raise_node_embeddings']
    # raise_node_embeddings.shape: batch_size, hidden_size
    exit_node_instruction_pointer = ipagnn_output['exit_node_instruction_pointer']
    # exit_node_instruction_pointer.shape: batch_size
    raise_node_instruction_pointer = ipagnn_output['raise_node_instruction_pointer']
    # raise_node_instruction_pointer.shape: batch_size

    num_classes = self.info.num_classes
    if config.raise_in_ipagnn:
      logits = nn.Dense(
          features=num_classes, name='output'
      )(raise_node_embeddings)  # P(e | yes exception)
      # logits.shape: batch_size, num_classes
      logits = logits.at[:, error_kinds.NO_ERROR_ID].set(-jnp.inf)

      no_error_logits = jax.vmap(logit_math.get_additional_logit)(
          exit_node_instruction_pointer + 1e-9,
          raise_node_instruction_pointer + 1e-9,
          logits)
      # no_error_logits.shape: batch_size
      logits = logits.at[:, error_kinds.NO_ERROR_ID].set(no_error_logits)
    else:
      logits = nn.Dense(
          features=num_classes, name='output'
      )(exit_node_embeddings)
    # logits.shape: batch_size, num_classes

    if config.raise_in_ipagnn:
      ipagnn_output['per_node_raise_contributions'] = raise_contributions_lib.get_raise_contribution_from_batch_and_aux(
          x, ipagnn_output)

    return logits, ipagnn_output
