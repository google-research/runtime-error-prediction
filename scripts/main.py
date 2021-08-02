# coding=utf-8
# Copyright 2021 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Lint as: python3
"""Jax runner binary for the Learned Interpreters project."""

import os

from absl import app
from absl import flags
from absl import logging

from ml_collections.config_flags import config_flags

from trainers import default_trainer

DEFAULT_DATA_DIR = 'data'
DEFAULT_CONFIG = 'config/default.py'

flags.DEFINE_string('data_dir', DEFAULT_DATA_DIR, 'Where to place the data.')
config_flags.DEFINE_config_file(
  name='config',
  default=DEFAULT_CONFIG,
  help_string='config file')
FLAGS = flags.FLAGS


def main(argv):
  del argv  # Unused.

  data_dir = FLAGS.data_dir
  config = FLAGS.config
  default_trainer.trainer(config)

if __name__ == '__main__':
  app.run(main)