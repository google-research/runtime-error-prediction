import functools
import os

import numpy as np
import tensorflow as tf

import jax.numpy as jnp
from core.data import codenet_paths
from core.data import tf_io

_int64_feature = tf_io.int64_feature
_float_feature = tf_io.float_feature
_bytes_feature = tf_io.bytes_feature
_int64_scalar_feature = tf_io.int64_scalar_feature
_int64_sequence_feature = tf_io.int64_sequence_feature
_string_scalar_feature = tf_io.string_scalar_feature


def to_tf_example(problem):
  """Constructs a tf.train.Example for the process.RuntimeErrorProblem."""
  return tf.train.Example(features=tf.train.Features(feature={
      'tokens': _int64_feature(problem.tokens),
      'docstring_tokens': _int64_feature(problem.docstring_tokens),
      'edge_sources': _int64_feature(problem.edge_sources),
      'edge_dests': _int64_feature(problem.edge_dests),
      'edge_types': _int64_feature(problem.edge_types),
      'node_token_span_starts': _int64_feature(problem.node_token_span_starts),
      'node_token_span_ends': _int64_feature(problem.node_token_span_ends),
      'token_node_indexes': _int64_feature(problem.token_node_indexes),
      'true_branch_nodes': _int64_feature(problem.true_branch_nodes),
      'false_branch_nodes': _int64_feature(problem.false_branch_nodes),
      'raise_nodes': _int64_feature(problem.raise_nodes),
      'start_index': _int64_feature([problem.start_index]),
      'exit_index': _int64_feature([problem.exit_index]),
      'step_limit': _int64_feature([problem.step_limit]),
      'target': _int64_feature([problem.target]),
      'target_lineno': _int64_feature([problem.target_lineno]),
      'target_node_indexes': _int64_feature(problem.target_node_indexes),
      'num_target_nodes': _int64_feature([len(problem.target_node_indexes)]),
      'post_domination_matrix': _int64_feature(list(problem.post_domination_matrix.flat)),
      'post_domination_matrix_shape': _int64_feature(problem.post_domination_matrix.shape),

      'problem_id': _bytes_feature([problem.problem_id]),
      'submission_id': _bytes_feature([problem.submission_id]),

      'in_dataset': _int64_feature([problem.in_dataset]),
      'num_tokens': _int64_feature([len(problem.tokens)]),
      'num_nodes': _int64_feature([len(problem.true_branch_nodes)]),
      'num_edges': _int64_feature([problem.num_edges]),
  }))


def decode_fn(record_bytes, include_strings=False):
  features = {
      'tokens': _int64_sequence_feature(),
      'docstring_tokens': _int64_sequence_feature(),
      'edge_sources': _int64_sequence_feature(),
      'edge_dests': _int64_sequence_feature(),
      'edge_types': _int64_sequence_feature(),
      'node_token_span_starts': _int64_sequence_feature(),
      'node_token_span_ends': _int64_sequence_feature(),
      'token_node_indexes': _int64_sequence_feature(),
      'true_branch_nodes': _int64_sequence_feature(),
      'false_branch_nodes': _int64_sequence_feature(),
      'raise_nodes': _int64_sequence_feature(),
      'start_index': _int64_scalar_feature(),
      'exit_index': _int64_scalar_feature(),
      'step_limit': _int64_scalar_feature(),
      'target': _int64_scalar_feature(),
      'target_lineno': _int64_scalar_feature(),
      'target_node_indexes': _int64_sequence_feature(),
      'num_target_nodes': _int64_scalar_feature(),
      'post_domination_matrix': _int64_sequence_feature(),
      'post_domination_matrix_shape': _int64_sequence_feature(),

      'in_dataset': _int64_scalar_feature(),
      'num_tokens': _int64_scalar_feature(),
      'num_nodes': _int64_scalar_feature(),
      'num_edges': _int64_scalar_feature(),
  }
  if include_strings:
    features.update({
        'problem_id': _string_scalar_feature(),
        'submission_id': _string_scalar_feature(),
    })
  example = tf.io.parse_single_example(record_bytes, features)
  example['post_domination_matrix'] = tf.reshape(
      example['post_domination_matrix'],
      example['post_domination_matrix_shape']
  )
  return example


def get_fake_input(batch_size, max_tokens, max_num_nodes, max_num_edges):
  return {
      'tokens': jnp.ones((batch_size, max_tokens), dtype=jnp.int32),
      'docstring_tokens': jnp.ones((batch_size, max_tokens), dtype=jnp.int32),
      'edge_sources': jnp.zeros((batch_size, 2 * max_num_edges + 4), dtype=jnp.int32),
      'edge_dests': jnp.ones((batch_size, 2 * max_num_edges + 4), dtype=jnp.int32),
      'edge_types': jnp.zeros((batch_size, 2 * max_num_edges + 4), dtype=jnp.int32),
      'edge_sources_shape': jnp.full((batch_size, 1), 2 * max_num_edges + 4, dtype=jnp.int32),
      'node_token_span_starts': jnp.zeros((batch_size, max_num_nodes), dtype=jnp.int32),
      'node_token_span_ends': jnp.ones((batch_size, max_num_nodes), dtype=jnp.int32),
      'true_branch_nodes': jnp.ones((batch_size, max_num_nodes), dtype=jnp.int32),
      'false_branch_nodes': jnp.ones((batch_size, max_num_nodes), dtype=jnp.int32),
      'raise_nodes': jnp.ones((batch_size, max_num_nodes), dtype=jnp.int32),
      'start_index': jnp.zeros((batch_size, 1), dtype=jnp.int32),
      'exit_index': jnp.full((batch_size, 1), max_num_nodes - 1, dtype=jnp.int32),
      'step_limit': jnp.full((batch_size, 1), max_num_nodes, dtype=jnp.int32),
      'target': jnp.zeros((batch_size, 1), dtype=jnp.int32),
      'target_lineno': jnp.ones((batch_size, 1), dtype=jnp.int32),
      'target_node_indexes': jnp.zeros((batch_size, 1), dtype=jnp.int32),
      'num_target_nodes': jnp.ones((batch_size, 1), dtype=jnp.int32),
      'post_domination_matrix': jnp.ones((batch_size, max_num_nodes, max_num_nodes), dtype=jnp.int32),
      'post_domination_matrix_shape': jnp.array([max_num_nodes, max_num_nodes], dtype=jnp.uint32),

      # We exclude problem_id and submission_id from fake_input, as they are not
      # model inputs.
      # 'problem_id': jnp.full((batch_size,), 'p12345', dtype=jnp.string),
      # 'submission_id': jnp.full((batch_size,), 's123456789', dtype=jnp.string),

      'in_dataset': jnp.ones((batch_size, 1), dtype=jnp.int32),
      'num_tokens': jnp.full((batch_size, 1), max_tokens, dtype=jnp.int32),
      'num_nodes': jnp.full((batch_size, 1), max_num_nodes, dtype=jnp.int32),
      'num_edges': jnp.full((batch_size, 1), max_num_edges, dtype=jnp.int32),
  }


def get_padded_shapes(max_tokens, max_num_nodes, max_num_edges, include_strings=False):
  # We do not expect an error to occur on a line containing more than
  # max_target_nodes statements. Most lines have only a single statement.
  max_target_nodes = 20
  shapes = {
      'tokens': [max_tokens],
      'docstring_tokens': [max_tokens],
      'edge_sources': [2 * max_num_edges + 6],
      'edge_dests': [2 * max_num_edges + 6],
      'edge_types': [2 * max_num_edges + 6],
      'edge_sources_shape': [1],  # Added in trainer.py.
      'node_token_span_starts': [max_num_nodes],
      'node_token_span_ends': [max_num_nodes],
      'token_node_indexes': [max_tokens],
      'true_branch_nodes': [max_num_nodes],
      'false_branch_nodes': [max_num_nodes],
      'raise_nodes': [max_num_nodes],
      'start_index': [1],
      'exit_index': [1],
      'step_limit': [1],
      'target': [1],
      'target_lineno': [1],
      'target_node_indexes': [max_target_nodes],
      'num_target_nodes': [1],
      'post_domination_matrix': [max_num_nodes, max_num_nodes],
      'post_domination_matrix_shape': [2],

      'in_dataset': [1],
      'num_tokens': [1],
      'num_nodes': [1],
      'num_edges': [1],
  }
  if include_strings:
    shapes.update({
        'problem_id': [1],
        'submission_id': [1],
    })
    
  return shapes


def make_filter(
    max_tokens, max_num_nodes, max_num_edges, max_steps, allowlist=None,
    class_subsample_values=None,
    use_in_dataset_field=True,
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

    if use_in_dataset_field:
      allowed &= tf.squeeze(example['in_dataset'] == 1, axis=-1)

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


def load_tfrecords_dataset(tfrecord_paths, include_strings=False):
  return tf.data.TFRecordDataset(
      tfrecord_paths,
      compression_type=None, buffer_size=None, num_parallel_reads=None
  ).map(functools.partial(decode_fn, include_strings=include_strings))


def load_dataset(dataset_path=codenet_paths.DEFAULT_DATASET_PATH, split='train', include_strings=False):
  if 'control_flow_programs_raise' in dataset_path:
    tfrecord_path = codenet_paths.make_tfrecord_path(dataset_path, split)
    return load_tfrecord_dataset(tfrecord_path, include_strings=include_strings)
  elif 'control_flow_programs' in dataset_path:
    split_ranges = {
        'train': range(212),
        'valid': range(212, 234),
        'test': range(234, 256),
    }
    tfrecord_paths = [
        os.path.join(dataset_path, f'control_flow_programs-train.tfrecord-{i:05d}-of-00256')
        for i in split_ranges[split]
    ]
    return load_tfrecords_dataset(tfrecord_paths, include_strings=include_strings)
  else:
    tfrecord_path = codenet_paths.make_tfrecord_path(dataset_path, split)
    return load_tfrecord_dataset(tfrecord_path, include_strings=include_strings)


def binarize_targets(example, dataset_path):
  if 'control_flow_programs_raise' in dataset_path:
    error_class = 1000
    is_error = tf.equal(example['target'], error_class)
    example['target'] = tf.cast(is_error, tf.int64)
    return example
  elif 'control_flow_programs' in dataset_path:
    raise ValueError('Unexpected dataset for binarization.')
  else:
    no_error_class = 1
    is_error = ~tf.equal(example['target'], no_error_class)
    example['target'] = tf.cast(is_error, tf.int64)
    return example
