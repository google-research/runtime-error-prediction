import tensorflow as tf


def _float_feature(value):
  """Constructs a tf.train.Feature for the given value list."""
  return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def to_tf_example(problem):
  """Constructs a tf.train.Example for the process.RuntimeErrorProblem."""
  return tf.train.Example(features=tf.train.Features(feature={
      "tokens": _float_feature(problem.tokens),
      "edge_sources": _float_feature(problem.edge_sources),
      "edge_dests": _float_feature(problem.edge_dests),
      "edge_types": _float_feature(problem.edge_types),
      "node_token_span_starts": _float_feature(problem.node_token_span_starts),
      "node_token_span_ends": _float_feature(problem.node_token_span_ends),
      "target": _float_feature([problem.target]),
  }))


def decode_fn(record_bytes):
  return tf.io.parse_single_example(
      record_bytes,
      {
          "tokens": tf.io.FixedLenFeature([], dtype=tf.float32),
          "edge_sources": tf.io.FixedLenFeature([], dtype=tf.float32),
          "edge_dests": tf.io.FixedLenFeature([], dtype=tf.float32),
          "edge_types": tf.io.FixedLenFeature([], dtype=tf.float32),
          "node_token_span_starts": tf.io.FixedLenFeature([], dtype=tf.float32),
          "node_token_span_ends": tf.io.FixedLenFeature([], dtype=tf.float32),
          "target": tf.io.FixedLenFeature([], dtype=tf.float32),
      }
  )
