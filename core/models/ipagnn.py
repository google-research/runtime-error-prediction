"""IPA-GNN models."""

from typing import Any

from flax import linen as nn
import jax
import jax.numpy as jnp

from core.data import error_kinds
from core.modules.ipagnn import ipagnn
from core.modules.ipagnn import logit_math
from core.modules.ipagnn import spans
from third_party.flax_examples import transformer_modules

NUM_CLASSES = error_kinds.NUM_CLASSES


class IPAGNN(nn.Module):

  config: Any

  def setup(self):
    config = self.config
    vocab_size = config.vocab_size  # TODO(dbieber): Load from tokenizer / info.
    max_tokens = config.max_tokens
    max_num_nodes = config.max_num_nodes
    max_num_edges = config.max_num_edges
    max_steps = config.max_steps
    info = ipagnn.Info(vocab_size=vocab_size)
    self.node_span_encoder = spans.NodeSpanEncoder(
        info=info,
        config=config,
        max_tokens=max_tokens,
        max_num_nodes=max_num_nodes,
    )

    self.ipagnn = ipagnn.IPAGNNModule(
        info=info,
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
        tokens, x['node_token_span_starts'], x['node_token_span_ends'])
    # encoded_inputs.shape: batch_size, max_num_nodes, hidden_size
    ipagnn_output = self.ipagnn(
        node_embeddings=encoded_inputs,
        edge_sources=x['edge_sources'],
        edge_dests=x['edge_dests'],
        edge_types=x['edge_types'],
        true_indexes=x['true_branch_nodes'],
        false_indexes=x['false_branch_nodes'],
        exit_indexes=x['exit_index'],
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

    if config.raise_in_ipagnn:
      logits = nn.Dense(features=NUM_CLASSES)(raise_node_embeddings)  # P(e | yes exception)
      # logits.shape: batch_size, NUM_CLASSES
      logits = logits.at[:, error_kinds.NO_DATA_ID].set(-jnp.inf)
      logits = logits.at[:, error_kinds.NO_ERROR_ID].set(-jnp.inf)

      no_error_logits = jax.vmap(logit_math.get_additional_logit)(
          exit_node_instruction_pointer + 1e-9,
          raise_node_instruction_pointer + 1e-9,
          logits)
      # no_error_logits.shape: batch_size
      logits = logits.at[:, error_kinds.NO_ERROR_ID].set(no_error_logits)
    else:
      logits = nn.Dense(features=NUM_CLASSES)(exit_node_embeddings)
      logits = logits.at[:, error_kinds.NO_DATA_ID].set(-jnp.inf)
    # logits.shape: batch_size, NUM_CLASSES
    return logits
