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

import tensorflow as tf


def int64_feature(value):
  """Constructs a tf.train.Feature for the given int64 value list."""
  return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def float_feature(value):
  """Constructs a tf.train.Feature for the given float value list."""
  return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def bytes_feature(values):
  """Constructs a tf.train.Feature for the given str value list."""
  values = [v.encode('utf-8') for v in values]
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=values))


def int64_scalar_feature():
  return tf.io.FixedLenFeature([1], dtype=tf.int64)


def int64_sequence_feature():
  return tf.io.FixedLenSequenceFeature(
      [], dtype=tf.int64, allow_missing=True, default_value=0)


def string_scalar_feature():
  return tf.io.FixedLenFeature([1], dtype=tf.string)
