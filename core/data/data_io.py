import functools

import tensorflow as tf

import jax.numpy as jnp
from core.data import codenet_paths


def _int64_feature(value):
  """Constructs a tf.train.Feature for the given int64 value list."""
  return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def _float_feature(value):
  """Constructs a tf.train.Feature for the given float value list."""
  return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def _bytes_feature(values):
  """Constructs a tf.train.Feature for the given str value list."""
  values = [v.encode('utf-8') for v in values]
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=values))


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

      'problem_id': _bytes_feature([problem.problem_id]),
      'submission_id': _bytes_feature([problem.submission_id]),

      'num_tokens': _int64_feature([len(problem.tokens)]),
      'num_nodes': _int64_feature([len(problem.true_branch_nodes)]),
      'num_edges': _int64_feature([len(problem.edge_sources)]),
  }))


def _int64_sequence_feature():
  return tf.io.FixedLenSequenceFeature(
      [], dtype=tf.int64, allow_missing=True, default_value=0)


def decode_fn(record_bytes, include_strings=False):
  features = {
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

      'num_tokens': tf.io.FixedLenFeature([1], dtype=tf.int64),
      'num_nodes': tf.io.FixedLenFeature([1], dtype=tf.int64),
      'num_edges': tf.io.FixedLenFeature([1], dtype=tf.int64),
  }
  if include_strings:
    features.update({
        'problem_id': tf.io.FixedLenFeature([1], dtype=tf.string),
        'submission_id': tf.io.FixedLenFeature([1], dtype=tf.string),
    })
  return tf.io.parse_single_example(record_bytes, features)


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

      # 'problem_id': jnp.full((batch_size,), 'p12345', dtype=jnp.string),
      # 'submission_id': jnp.full((batch_size,), 's123456789', dtype=jnp.string),

      'num_tokens': jnp.full((batch_size, 1), max_tokens, dtype=jnp.int32),
      'num_nodes': jnp.full((batch_size, 1), max_num_nodes, dtype=jnp.int32),
      'num_edges': jnp.full((batch_size, 1), max_num_edges, dtype=jnp.int32),
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

      'num_tokens': [1],
      'num_nodes': [1],
      'num_edges': [1],
  }


def make_filter(
    max_tokens, max_num_nodes, max_num_edges, max_steps, allowlist=None,
    class_subsample_values=None,
):
  """Makes a tf.Dataset filter function.

  Args:
    max_tokens: Filter out any examples with more than max_tokens tokens.
    max_num_nodes: Filter out any examples with more than max_num_nodes nodes.
    max_num_edges: Filter out any examples with more than max_num_edges edges.
    max_steps: Filter out any examples with step_limit more than max_steps.
    allowlist: (Optional) If set, only admit examples with targets in this list.
    class_subsample_values: (Optional[Dict]) If set, keys indicate which target
      classes to subsample, and values indicate how much to subsample.
      class_subsample_values={1: 0.25} will admit only 25% of examples with
      target 1.
  Returns:
    The filter function, suitable for use with dataset.filter.
  """
  def fn(example):
    # An on-device predicate for filtering out too-large examples.
    allowed = tf.squeeze(
        (example['num_tokens'] <= max_tokens)
        & (example['num_nodes'] <= max_num_nodes)
        & (example['num_edges'] <= max_num_edges)
        & (example['step_limit'] <= max_steps),
        axis=-1
    )
    target = example['target'][0]
    if allowlist is not None:
      # Limit the allowed error_kinds to the allowlist.
      class_ok = False
      for index in allowlist:
        class_ok |= (target == index)
      allowed = allowed & class_ok

    # Filter x% of examples with target == 1 (the most common class).
    if class_subsample_values is not None:
      for key, value in class_subsample_values.items():
        allowed &= ((target != key) | (tf.random.uniform(shape=()) < value))

    return allowed
  return fn


def load_tfrecord_dataset(tfrecord_path, include_strings=False):
  return tf.data.TFRecordDataset(
      [tfrecord_path],
      compression_type=None, buffer_size=None, num_parallel_reads=None
  ).map(functools.partial(decode_fn, include_strings=include_strings))


def load_dataset(dataset_path=codenet_paths.DEFAULT_DATASET_PATH, split='train', include_strings=False):
  tfrecord_path = codenet_paths.make_tfrecord_path(dataset_path, split)
  return load_tfrecord_dataset(tfrecord_path, include_strings=include_strings)
