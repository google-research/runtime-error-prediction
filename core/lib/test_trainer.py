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
  def test_cross_attention_ipagnn(self):
    config = config_lib.get_test_config()
    config.model_class = 'IPAGNN'
    config.batch_size = 16
    config.use_cross_attention = True
    config.raise_in_ipagnn = False
    config.eval_freq = 1
    config.max_tokens = 512
    validate_single_step(config)



if __name__ == '__main__':
  unittest.main()
