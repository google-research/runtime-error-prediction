"""Allows loading control_flow_programs raise raw data."""

import functools

import tensorflow as tf

from core.data import tf_io

_int64_feature = tf_io.int64_feature
_float_feature = tf_io.float_feature
_bytes_feature = tf_io.bytes_feature
_int64_scalar_feature = tf_io.int64_scalar_feature
_int64_sequence_feature = tf_io.int64_sequence_feature
_string_scalar_feature = tf_io.string_scalar_feature


def decode_fn(record_bytes):
  features = {
      'source': _string_scalar_feature(),
      'target': _string_scalar_feature(),
      'steps': _int64_scalar_feature(),
  }
  return tf.io.parse_single_example(record_bytes, features)


def load_dataset(tfrecord_paths):
  return tf.data.TFRecordDataset(
      tfrecord_paths,
      compression_type=None, buffer_size=None, num_parallel_reads=1
  ).map(decode_fn)
