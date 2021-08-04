import tensorflow as tf

import jax.numpy as jnp
from core.data import codenet_paths


def _int64_feature(value):
  """Constructs a tf.train.Feature for the given int64 value list."""
  return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def _float_feature(value):
  """Constructs a tf.train.Feature for the given float value list."""
  return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def to_tf_example(problem):
  """Constructs a tf.train.Example for the process.RuntimeErrorProblem."""
  return tf.train.Example(features=tf.train.Features(feature={
      'tokens': _int64_feature(problem.tokens),
      'edge_sources': _int64_feature(problem.edge_sources),
      'edge_dests': _int64_feature(problem.edge_dests),
      'edge_types': _int64_feature(problem.edge_types),
      'node_token_span_starts': _int64_feature(problem.node_token_span_starts),
      'node_token_span_ends': _int64_feature(problem.node_token_span_ends),
      'token_node_indexes': _int64_feature(problem.token_node_indexes),
      'true_branch_nodes': _int64_feature(problem.true_branch_nodes),
      'false_branch_nodes': _int64_feature(problem.false_branch_nodes),
      'exit_index': _int64_feature([problem.exit_index]),
      'step_limit': _int64_feature([problem.step_limit]),
      'target': _int64_feature([problem.target]),
  }))


def _int64_sequence_feature():
  return tf.io.FixedLenSequenceFeature(
      [], dtype=tf.int64, allow_missing=True, default_value=0)


def decode_fn(record_bytes):
  return tf.io.parse_single_example(
      record_bytes,
      {
          'tokens': _int64_sequence_feature(),
          'edge_sources': _int64_sequence_feature(),
          'edge_dests': _int64_sequence_feature(),
          'edge_types': _int64_sequence_feature(),
          'node_token_span_starts': _int64_sequence_feature(),
          'node_token_span_ends': _int64_sequence_feature(),
          'token_node_indexes': _int64_sequence_feature(),
          'true_branch_nodes': _int64_sequence_feature(),
          'false_branch_nodes': _int64_sequence_feature(),
          'exit_index': tf.io.FixedLenFeature([1], dtype=tf.int64),
          'step_limit': tf.io.FixedLenFeature([1], dtype=tf.int64),
          'target': tf.io.FixedLenFeature([1], dtype=tf.int64),
      }
  )


def get_fake_input(batch_size, max_tokens, max_num_nodes, max_num_edges):
  return {
      'tokens': jnp.ones((batch_size, max_tokens), dtype=jnp.int32),
      'edge_sources': jnp.zeros((batch_size, max_num_edges), dtype=jnp.int32),
      'edge_dests': jnp.ones((batch_size, max_num_edges), dtype=jnp.int32),
      'edge_types': jnp.zeros((batch_size, max_num_edges), dtype=jnp.int32),
      'node_token_span_starts': jnp.zeros((batch_size, max_num_nodes), dtype=jnp.int32),
      'node_token_span_ends': jnp.ones((batch_size, max_num_nodes), dtype=jnp.int32),
      'true_branch_nodes': jnp.ones((batch_size, max_num_nodes), dtype=jnp.int32),
      'false_branch_nodes': jnp.ones((batch_size, max_num_nodes), dtype=jnp.int32),
      'exit_index': jnp.full((batch_size, 1), max_num_nodes - 1, dtype=jnp.int32),
      'step_limit': jnp.full((batch_size, 1), max_num_nodes, dtype=jnp.int32),
      'target': jnp.zeros((batch_size, 1), dtype=jnp.int32),
  }


def get_padded_shapes(max_tokens, max_num_nodes, max_num_edges):
  return {
      'tokens': [max_tokens],
      'edge_sources': [max_num_edges],
      'edge_dests': [max_num_edges],
      'edge_types': [max_num_edges],
      'node_token_span_starts': [max_num_nodes],
      'node_token_span_ends': [max_num_nodes],
      'token_node_indexes': [max_tokens],
      'true_branch_nodes': [max_num_nodes],
      'false_branch_nodes': [max_num_nodes],
      'exit_index': [1],
      'step_limit': [1],
      'target': [1],
  }


def load_dataset(dataset_path=codenet_paths.DEFAULT_DATASET_PATH):
  return tf.data.TFRecordDataset(
      [dataset_path],
      compression_type=None, buffer_size=None, num_parallel_reads=None
  ).map(decode_fn)
