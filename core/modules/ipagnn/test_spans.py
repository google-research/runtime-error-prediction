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

"""Tests for ipagnn.py."""

import unittest

import jax
import jax.numpy as jnp

import ml_collections
from core.data import info as info_lib
from core.modules import transformer_config_lib
from core.modules.ipagnn import spans


Config = ml_collections.ConfigDict


def make_sample_config():
  config = Config()
  config.hidden_size = 10
  config.span_encoding_method = 'first'
  config.max_tokens = 64
  config.permissive_node_embeddings = True

  config.transformer_emb_dim = 32
  config.transformer_num_heads = 4
  config.transformer_num_layers = 2
  config.transformer_qkv_dim = 32
  config.transformer_mlp_dim = 64
  config.transformer_dropout_rate = 0.1
  config.transformer_attention_dropout_rate = 0.1
  return config


class NodeSpanEncoderTest(unittest.TestCase):

  def test_call(self):
    tokens_list = [
        66, 31, 99, 84, 125, 10, 377, 10, 214, 223, 315, 222, 237, 214, 10,
        166, 10, 61, 95, 148, 66, 8, 61, 79, 18, 89, 61, 80, 104, 10, 19, 14,
        66, 13, 19, 95, 148, 9, 240, 9, 82, 9, 239, 222]
    num_tokens = len(tokens_list)
    tokens = jnp.array([tokens_list, tokens_list])
    # tokens.shape: batch_size=2, length
    node_span_starts = jnp.array([[3, 7, 10], [4, 7, 11]])
    node_span_ends = jnp.array([[5, 9, 15], [6, 10, 16]])
    num_nodes = jnp.array([3, 3])

    info = info_lib.get_test_info()
    config = make_sample_config()
    transformer_config = transformer_config_lib.make_transformer_config(
        config, 300, True
    )

    encoder = spans.NodeSpanEncoder(
        info, config, transformer_config,
        max_tokens=num_tokens, max_num_nodes=3)
    rng = jax.random.PRNGKey(0)
    rng, params_rng, dropout_rng = jax.random.split(rng, 3)
    variables = encoder.init(
        {'params': params_rng, 'dropout': dropout_rng},
        tokens, node_span_starts, node_span_ends, num_nodes,
    )
    params = variables['params']

    rng, dropout_rng = jax.random.split(rng, 2)
    encodings = encoder.apply(
        {'params': params},
        tokens, node_span_starts, node_span_ends, num_nodes,
        rngs={'dropout': dropout_rng},
    )
    # encodings.shape: batch_size, num_nodes, hidden_size
    encodings_shape = encodings.shape
    self.assertEqual(encodings_shape, (2, 3, 10))

  def test_make_span_attention_mask_single(self):
    tokens_list = [
        66, 31, 99, 84, 125, 10, 377, 10, 214, 223, 315, 222, 237, 214, 10,
        166, 10, 61, 95, 148, 66, 8, 61, 79, 18, 89, 61, 80, 104, 10, 19, 14,
        66, 13, 19, 95, 148, 9, 240, 9, 82, 9, 239, 222]
    num_tokens = len(tokens_list)
    tokens = jnp.array([tokens_list, tokens_list])
    # tokens.shape: batch_size=2, length
    node_span_starts = jnp.array([[3, 7, 10], [4, 7, 11]])
    node_span_ends = jnp.array([[5, 9, 15], [6, 10, 16]])
    num_nodes = jnp.array([3, 3])

    info = info_lib.get_test_info()
    config = make_sample_config()
    config.permissive_node_embeddings = True
    transformer_config = transformer_config_lib.make_transformer_config(
        config, 300, True
    )

    encoder = spans.NodeSpanEncoder(
        info, config, transformer_config,
        max_tokens=num_tokens, max_num_nodes=3)
    rng = jax.random.PRNGKey(0)
    rng, params_rng, dropout_rng = jax.random.split(rng, 3)
    variables = encoder.init(
        {'params': params_rng, 'dropout': dropout_rng},
        tokens, node_span_starts, node_span_ends, num_nodes,
    )
    params = variables['params']

    # Part A. With permissive_node_embeddings == True:
    rng, dropout_rng = jax.random.split(rng, 2)
    encodings = encoder.apply(
        {'params': params},
        tokens, node_span_starts, node_span_ends, num_nodes,
        rngs={'dropout': dropout_rng},
    )

    # 1. same tokens -> same embeddings.
    modified_encodings = encoder.apply(
        {'params': params},
        tokens, node_span_starts, node_span_ends, num_nodes,
        rngs={'dropout': dropout_rng},
    )
    self.assertTrue(jnp.all(jnp.equal(encodings, modified_encodings)))

    # 2. different token outside spans -> different embeddings.
    tokens = tokens.at[:, 20].set(101)  # Change a token outside the spans.
    modified_encodings = encoder.apply(
        {'params': params},
        tokens, node_span_starts, node_span_ends, num_nodes,
        rngs={'dropout': dropout_rng},
    )
    self.assertFalse(jnp.all(jnp.equal(encodings, modified_encodings)))

    # Part B. With permissive_node_embeddings == False:
    config.permissive_node_embeddings = False
    encoder = spans.NodeSpanEncoder(
        info, config, transformer_config,
        max_tokens=num_tokens, max_num_nodes=3)

    encodings = encoder.apply(
        {'params': params},
        tokens, node_span_starts, node_span_ends, num_nodes,
        rngs={'dropout': dropout_rng},
    )
    # 1. same tokens -> same embeddings.
    modified_encodings = encoder.apply(
        {'params': params},
        tokens, node_span_starts, node_span_ends, num_nodes,
        rngs={'dropout': dropout_rng},
    )
    self.assertTrue(jnp.all(jnp.equal(encodings, modified_encodings)))

    # 2. different token outside spans -> same embeddings.
    tokens = tokens.at[:, 20].set(105)  # Change a token outside the spans.
    modified_encodings = encoder.apply(
        {'params': params},
        tokens, node_span_starts, node_span_ends, num_nodes,
        rngs={'dropout': dropout_rng},
    )
    self.assertTrue(jnp.all(jnp.equal(encodings, modified_encodings)))

    # 2. different token inside a span -> different embeddings.
    tokens = tokens.at[:, 3].set(105)  # Change a token inside a span.
    modified_encodings = encoder.apply(
        {'params': params},
        tokens, node_span_starts, node_span_ends, num_nodes,
        rngs={'dropout': dropout_rng},
    )
    self.assertFalse(jnp.all(jnp.equal(encodings, modified_encodings)))


class SpanIndexEncoderTest(unittest.TestCase):

  def test_call(self):
    node_span_starts = jnp.array([0, 2])
    node_span_ends = jnp.array([2, 4])
    num_nodes = 2

    encoder = spans.SpanIndexEncoder(
        max_tokens=5,
        max_num_nodes=2,
        features=3,
    )
    rng = jax.random.PRNGKey(0)
    rng, params_rng = jax.random.split(rng, 2)
    variables = encoder.init(
        {'params': params_rng},
        node_span_starts, node_span_ends, num_nodes
    )
    params = variables['params']

    encodings = encoder.apply(
        {'params': params},
        node_span_starts, node_span_ends, num_nodes,
    )
    # encodings.shape: num_tokens, features
    encodings_shape = encodings.shape
    self.assertEqual(encodings_shape, (5, 3))

    self.assertTrue(jnp.all(encodings[0] == encodings[1]))
    self.assertFalse(jnp.all(encodings[1] == encodings[2]))
    self.assertFalse(jnp.all(encodings[2] == encodings[3]))
    self.assertTrue(jnp.all(encodings[3] == encodings[4]))


if __name__ == '__main__':
  unittest.main()
