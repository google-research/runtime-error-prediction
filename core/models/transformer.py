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

from typing import Any

from flax import linen as nn
import jax
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
  """Multiple instance learning Transformer.

  NodeSpanEncoder can give contextual or non-contextual span embeddings.
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
    max_num_nodes = encoded_inputs.shape[1]

    num_classes = info.num_classes
    per_line_logits = nn.Dense(
        features=num_classes, name='per_line_output'
    )(encoded_inputs)
    # per_line_logits.shape: batch_size, max_num_nodes, num_classes

    aux = {}
    if config.mil_pool == 'logsumexp':
      # logits = phi(c) = logsumexp_{l} phi(l, c)
      logits = jax.scipy.special.logsumexp(per_line_logits, axis=1)
      # logits.shape: batch_size, num_classes
      # probs = p(c) \propto exp(phi(c))

      # phi(l) = logsumexp_{c \in errors} phi(l, c)
      if len(info.no_error_ids) == 1:
        no_error_id = info.no_error_ids[0]
        per_line_error_logits = per_line_logits.at[:, :, no_error_id].set(-jnp.inf)
        # per_line_error_logits.shape: batch_size, max_num_nodes, num_classes
        aux['localization_logits'] = jax.scipy.special.logsumexp(
            per_line_error_logits, axis=2)
        # aux['localization_logits'].shape: batch_size, max_num_nodes
      elif len(info.no_error_ids) > 1:
        if len(info.error_ids) > 1:
          raise NotImplementedError('Multiple error classes and multiple no-error classes.')
        assert len(info.error_ids) == 1
        error_id = info.error_ids[0]
        per_line_error_logits = per_line_logits.at[:, :, error_id]
        # per_line_error_logits.shape: batch_size, max_num_nodes
        aux['localization_logits'] = per_line_error_logits
        # aux['localization_logits'].shape: batch_size, max_num_nodes
      else:
        raise ValueError('Tried using MILTransformer on data with no errors.')
    else:
      per_line_probs = jax.nn.softmax(per_line_logits, axis=-1)
      # per_line_probs.shape: batch_size, max_num_nodes, num_classes

      # Compute localization_logits:
      if len(info.no_error_ids) == 1:
        no_error_id = info.no_error_ids[0]
        per_line_no_error_probs = per_line_probs[:, :, no_error_id]
        # per_line_no_error_probs.shape: batch_size, max_num_nodes
        per_line_error_probs = 1 - per_line_no_error_probs
        # per_line_error_probs.shape: batch_size, max_num_nodes
        # Normalization assumes uniform prior over lines.
        normalized_error_probs = per_line_error_probs / jnp.sum(per_line_error_probs, axis=-1, keepdims=True)
        # normalized_error_probs.shape: batch_size, max_num_nodes
        aux['localization_logits'] = jnp.log(normalized_error_probs)
        # aux['localization_logits'].shape: batch_size, max_num_nodes
        assert aux['localization_logits'].shape == (batch_size, max_num_nodes)
      elif len(info.no_error_ids) > 1:
        if len(info.error_ids) > 1:
          raise NotImplementedError('Multiple error classes and multiple no-error classes.')
        assert len(info.error_ids) == 1
        error_id = info.error_ids[0]
        per_line_error_probs = per_line_probs[:, :, error_id]
        # per_line_error_probs.shape: batch_size, max_num_nodes
        # Normalization assumes uniform prior over lines.
        normalized_error_probs = per_line_error_probs / jnp.sum(per_line_error_probs, axis=-1, keepdims=True)
        # normalized_error_probs.shape: batch_size, max_num_nodes
        aux['localization_logits'] = jnp.log(normalized_error_probs)
        # aux['localization_logits'].shape: batch_size, max_num_nodes
        assert aux['localization_logits'].shape == (batch_size, max_num_nodes)
      else:
        raise ValueError('Tried using MILTransformer on data with no errors.')

      # Compute overall logits using pooling:
      if config.mil_pool == 'max':
        max_probs = jnp.max(per_line_probs, axis=1)
        # max_probs.shape: batch_size, num_classes
        logits = jnp.log(max_probs)
        # logits.shape: batch_size, num_classes
      elif config.mil_pool == 'mean':
        mean_probs = jnp.mean(per_line_probs, axis=1)
        # mean_probs.shape: batch_size, num_classes
        logits = jnp.log(mean_probs)
        # logits.shape: batch_size, num_classes
      elif config.mil_pool == 'noisy-or':
        # NOTE: I leave noisy-or unfinished because we are operating in the
        # at-most-one error setting.
        per_line_neg_probs = 1 - per_line_probs
        # per_line_neg_probs.shape: batch_size, max_num_nodes, num_classes
        class_probs = x = 1 - jnp.product(per_line_neg_probs, axis=1)
        # x.shape: batch_size, num_classes
        raise NotImplementedError()
      else:
        raise ValueError('Unexpected mil_pool parameter', config.mil_pool)

    return logits, aux
