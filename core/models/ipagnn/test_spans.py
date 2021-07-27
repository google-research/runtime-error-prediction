"""Tests for ipagnn.py."""

import unittest

import jax
import jax.numpy as jnp

import ml_collections
from core.models.ipagnn import ipagnn
from core.models.ipagnn import spans


Config = ml_collections.ConfigDict


def make_sample_config():
  config = Config()
  config.model = Config()
  config.model.hidden_size = 10
  return config


class NodeSpanEncoderTest(unittest.TestCase):

  def test_call(self):
    tokens_list = [
        66, 31, 99, 84, 125, 10, 377, 10, 214, 223, 315, 222, 237, 214, 10,
        166, 10, 61, 95, 148, 66, 8, 61, 79, 18, 89, 61, 80, 104, 10, 19, 14,
        66, 13, 19, 95, 148, 9, 240, 9, 82, 9, 239, 222]
    tokens = jnp.array([tokens_list, tokens_list])
    # tokens.shape: batch_size=2, length
    node_span_starts = jnp.array([[3, 7, 10], [4, 7, 11]])
    node_span_ends = jnp.array([[5, 9, 15], [6, 10, 16]])

    info = ipagnn.Info(vocab_size=500)
    config = make_sample_config()

    encoder = spans.NodeSpanEncoder(info, config)
    rng = jax.random.PRNGKey(0)
    rng, params_rng, dropout_rng = jax.random.split(rng, 3)
    variables = encoder.init(
        {'params': params_rng, 'dropout': dropout_rng},
        tokens, node_span_starts, node_span_ends)
    params = variables['params']

    rng, dropout_rng = jax.random.split(rng, 2)
    encodings = encoder.apply(
        {'params': params},
        tokens, node_span_starts, node_span_ends,
        rngs={'dropout': dropout_rng},
    )
    # encodings.shape: batch_size, num_nodes, hidden_size
    encodings_shape = encodings.shape
    self.assertEqual(encodings_shape, (2, 3, 512))


class SpanIndexEncoderTest(unittest.TestCase):

  def test_call(self):
    node_span_starts = jnp.array([0, 2])
    node_span_ends = jnp.array([2, 4])

    encoder = spans.SpanIndexEncoder(
        max_tokens=5,
        max_num_nodes=2,
        features=3,
    )
    rng = jax.random.PRNGKey(0)
    rng, params_rng = jax.random.split(rng, 2)
    variables = encoder.init(
        {'params': params_rng},
        node_span_starts,
        node_span_ends
    )
    params = variables['params']

    encodings = encoder.apply(
        {'params': params},
        node_span_starts, node_span_ends,
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
