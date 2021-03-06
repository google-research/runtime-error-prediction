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

"""IPA-GNN models."""

from typing import Any

from flax import linen as nn
import jax
import jax.numpy as jnp

from core.data import error_kinds
from core.modules.ipagnn import compressive_ipagnn
from core.modules.ipagnn import encoder
from core.modules.ipagnn import ipagnn
from core.modules.ipagnn import logit_math
from core.modules.ipagnn import spans
from core.modules.ipagnn import raise_contributions as raise_contributions_lib
from third_party.flax_examples import transformer_modules


class IPAGNN(nn.Module):

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

    if config.use_compressive_ipagnn:
      self.ipagnn = compressive_ipagnn.SkipIPAGNN(
          config=config,
          info=self.info,
          max_steps=max_steps,
      )
    else:
      self.ipagnn = ipagnn.IPAGNNModule(
          info=self.info,
          config=config,
          max_steps=max_steps,
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
    ipagnn_output = self.ipagnn(
        node_embeddings=encoded_inputs,
        docstring_embeddings=docstring_embeddings,
        docstring_mask=docstring_mask,
        edge_sources=x['edge_sources'],
        edge_dests=x['edge_dests'],
        edge_types=x['edge_types'],
        true_indexes=x['true_branch_nodes'],
        false_indexes=x['false_branch_nodes'],
        raise_indexes=x['raise_nodes'],
        start_node_indexes=x['start_index'],
        exit_node_indexes=x['exit_index'],
        post_domination_matrix=x['post_domination_matrix'],
        step_limits=x['step_limit'],
    )
    # ipagnn_output['exit_node_embeddings'].shape: batch_size, hidden_size
    # ipagnn_output['raise_node_embeddings'].shape: batch_size, hidden_size
    # ipagnn_output['exit_node_instruction_pointer'].shape: batch_size
    # ipagnn_output['raise_node_instruction_pointer'].shape: batch_size

    exit_node_embeddings = ipagnn_output['exit_node_embeddings']
    # exit_node_embeddings.shape: batch_size, hidden_size
    exit_node_instruction_pointer = ipagnn_output['exit_node_instruction_pointer']
    # exit_node_instruction_pointer.shape: batch_size

    num_classes = info.num_classes
    if config.raise_in_ipagnn:
      raise_node_embeddings = ipagnn_output['raise_node_embeddings']
      # raise_node_embeddings.shape: batch_size, hidden_size
      raise_node_instruction_pointer = ipagnn_output['raise_node_instruction_pointer']
      # raise_node_instruction_pointer.shape: batch_size
      if len(info.no_error_ids) == 1:
        # Multiple error classes; only one No-Error class.
        no_error_id = info.no_error_ids[0]
        logits = nn.Dense(
            features=num_classes, name='output'
        )(raise_node_embeddings)  # P(e | yes exception)
        # logits.shape: batch_size, num_classes
        logits = logits.at[:, no_error_id].set(-jnp.inf)

        no_error_logits = jax.vmap(logit_math.get_additional_logit)(
            exit_node_instruction_pointer + 1e-9,
            raise_node_instruction_pointer + 1e-9,
            logits)
        # no_error_logits.shape: batch_size
        logits = logits.at[:, no_error_id].set(no_error_logits)
      elif len(info.no_error_ids) > 1:
        # Multiple No-Error classes; only one error class.
        if len(info.error_ids) > 1:
          raise NotImplementedError('Multiple error classes and multiple no-error classes.')
        assert len(info.error_ids) == 1
        error_id = info.error_ids[0]
        logits = nn.Dense(
            features=num_classes, name='output'
        )(exit_node_embeddings)  # P(e | no exception)
        # logits.shape: batch_size, num_classes
        logits = logits.at[:, error_id].set(-jnp.inf)
        error_logits = jax.vmap(logit_math.get_additional_logit)(
            raise_node_instruction_pointer + 1e-9,
            exit_node_instruction_pointer + 1e-9,
            logits)
        # error_logits.shape: batch_size
        logits = logits.at[:, error_id].set(error_logits)
      else:
        raise ValueError('Tried using Exception IPA-GNN on data with no errors.')
    else:
      logits = nn.Dense(
          features=num_classes, name='output'
      )(exit_node_embeddings)
    # logits.shape: batch_size, num_classes

    return logits, ipagnn_output
