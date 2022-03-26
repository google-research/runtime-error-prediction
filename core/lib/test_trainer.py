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

import jax
import pytest
import unittest

from config import default as config_lib
from core.data import codenet_paths
from core.data import data_io
from core.data import info as info_lib
from core.lib import trainer

import tensorflow_datasets as tfds


def validate_single_step(config):
  info = info_lib.get_test_info()
  dataset_path = codenet_paths.TEST_DATASET_PATH
  split = 'train'
  steps = 1
  trainer.Trainer(config=config, info=info).run_train(
      dataset_path=dataset_path, split=split, steps=steps)


class TrainerTest(unittest.TestCase):

  @pytest.mark.slow
  def test_ipagnn(self):
    config = config_lib.get_test_config()
    config.model_class = 'IPAGNN'
    config.batch_size = 16
    config.raise_in_ipagnn = True
    config.max_tokens = 512
    validate_single_step(config)

  # Disabled for timeout.
  # @pytest.mark.slow
  def test_ggnn(self):
    config = config_lib.get_test_config()
    config.model_class = 'GGNN'
    config.batch_size = 16
    config.max_tokens = 512
    validate_single_step(config)

  def test_finetune_from_lstm(self):
    config = config_lib.get_test_config()
    config.model_class = 'LSTM'
    config.batch_size = 16
    config.max_tokens = 512
    config.experiment_id = 'tests-lstm'
    validate_single_step(config)

    config = config_lib.get_test_config()
    config.model_class = 'IPAGNN'
    config.batch_size = 16
    config.max_tokens = 512
    config.finetune = 'LSTM'
    config.restore_checkpoint_dir = 'out/experiments/tests-lstm/checkpoints'
    validate_single_step(config)

  @pytest.mark.slow
  def test_film_ipagnn(self):
    config = config_lib.get_test_config()
    config.model_class = 'IPAGNN'
    config.batch_size = 16
    config.use_film = True
    config.raise_in_ipagnn = False
    config.eval_freq = 1
    config.max_tokens = 512
    validate_single_step(config)

  @pytest.mark.slow
  def test_film_exception_ipagnn(self):
    config = config_lib.get_test_config()
    config.model_class = 'IPAGNN'
    config.batch_size = 16
    config.use_film = True
    config.raise_in_ipagnn = True
    config.modulate_mode = 'concat'
    config.eval_freq = 1
    config.max_tokens = 512
    validate_single_step(config)

  @pytest.mark.slow
  def test_compressive_ipagnn(self):
    config = config_lib.get_test_config()
    config.model_class = 'IPAGNN'
    config.batch_size = 16
    config.use_compressive_ipagnn = True
    config.eval_freq = 1
    config.compressive_max_skip = 3
    config.max_tokens = 512
    validate_single_step(config)


if __name__ == '__main__':
  unittest.main()
