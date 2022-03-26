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

"""Allows loading control_flow_programs raw data."""

import functools

import tensorflow as tf

from core.data import tf_io

_int64_feature = tf_io.int64_feature
_float_feature = tf_io.float_feature
_bytes_feature = tf_io.bytes_feature
_int64_scalar_feature = tf_io.int64_scalar_feature
_int64_sequence_feature = tf_io.int64_sequence_feature
_string_scalar_feature = tf_io.string_scalar_feature


def decode_fn(record_bytes, include_strings=False):
  features = {
      # We omit near all features from the dataset except e.g. the raw source.
      # We load only those we use.
      # Example features omitted:
      # 'cfg_forward/steps': _int64_scalar_feature(),
      # 'cfg/linenos': _int64_sequence_feature(),
      'cfg_forward/steps': _int64_scalar_feature(),
      'target_output': _int64_scalar_feature(),
  }
  if include_strings:
    features.update({
        'human_readable_code': _string_scalar_feature()
    })
  return tf.io.parse_single_example(record_bytes, features)


def load_dataset(tfrecord_paths, include_strings=False):
  return tf.data.TFRecordDataset(
      tfrecord_paths,
      compression_type=None, buffer_size=None, num_parallel_reads=32
  ).map(functools.partial(decode_fn, include_strings=include_strings))
