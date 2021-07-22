import tensorflow as tf


def _int64_feature(value):
  """Constructs a tf.train.Feature for the given int64 value list."""
  return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def _float_feature(value):
  """Constructs a tf.train.Feature for the given float value list."""
  return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def to_tf_example(problem):
  """Constructs a tf.train.Example for the process.RuntimeErrorProblem."""
  return tf.train.Example(features=tf.train.Features(feature={
      "tokens": _int64_feature(problem.tokens),
      "edge_sources": _int64_feature(problem.edge_sources),
      "edge_dests": _int64_feature(problem.edge_dests),
      "edge_types": _int64_feature(problem.edge_types),
      "node_token_span_starts": _int64_feature(problem.node_token_span_starts),
      "node_token_span_ends": _int64_feature(problem.node_token_span_ends),
      "target": _int64_feature([problem.target]),
  }))


def _int64_sequence_feature():
  return tf.io.FixedLenSequenceFeature(
      [], dtype=tf.int64, allow_missing=True, default_value=0)


def decode_fn(record_bytes):
  return tf.io.parse_single_example(
      record_bytes,
      {
          "tokens": _int64_sequence_feature(),
          "edge_sources": _int64_sequence_feature(),
          "edge_dests": _int64_sequence_feature(),
          "edge_types": _int64_sequence_feature(),
          "node_token_span_starts": _int64_sequence_feature(),
          "node_token_span_ends": _int64_sequence_feature(),
          "target": tf.io.FixedLenFeature([1], dtype=tf.int64),
      }
  )


def load_dataset(dataset_path=DEFAULT_DATASET_PATH):
  return tf.data.TFRecordDataset(
      [dataset_path],
      compression_type=None, buffer_size=None, num_parallel_reads=None
  ).map(decode_fn)
