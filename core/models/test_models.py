import jax
import unittest

from config import default as config_lib
from core.data import codenet_paths
from core.data import data_io
from core.data import info as info_lib
from core.lib import models

import tensorflow_datasets as tfds


def validate_forward_pass(config, info):
  """Creates, initializes, and runs a single forward pass of the indicated model."""
  model = models.make_model(config, info, deterministic=True)

  fake_input = data_io.get_fake_input(
      config.batch_size, config.max_tokens, config.max_num_nodes, config.max_num_edges)
  rng = jax.random.PRNGKey(0)
  rng, params_rng, dropout_rng = jax.random.split(rng, 3)
  variables = model.init(
      {'params': params_rng, 'dropout': dropout_rng},
      fake_input)
  params = variables['params']

  padded_shapes = data_io.get_padded_shapes(
      config.max_tokens, config.max_num_nodes, config.max_num_edges)
  filter_fn = data_io.make_filter(
      config.max_tokens, config.max_num_nodes, config.max_num_edges,
      config.max_steps)
  dataset_path = codenet_paths.TEST_DATASET_PATH
  dataset = (
      data_io.load_dataset(dataset_path, split='train')
      .filter(filter_fn)
      .padded_batch(config.batch_size, padded_shapes=padded_shapes, drop_remainder=True)
  )

  batch = next(iter(tfds.as_numpy(dataset)))
  model.apply(
      {'params': params},
      batch,
      rngs={'dropout': dropout_rng}
  )


class ModelsTest(unittest.TestCase):

  def test_ipagnn(self):
    info = info_lib.get_test_info()
    config = config_lib.get_test_config()
    config.model_class = 'IPAGNN'
    config.raise_in_ipagnn = False
    validate_forward_pass(config, info)

  def test_film_ipagnn(self):
    info = info_lib.get_test_info()
    config = config_lib.get_test_config()
    config.model_class = 'IPAGNN'
    config.use_film = True
    config.raise_in_ipagnn = False
    validate_forward_pass(config, info)

  def test_cross_attention_ipagnn(self):
    info = info_lib.get_test_info()
    config = config_lib.get_test_config()
    config.model_class = 'IPAGNN'
    config.use_cross_attention = True
    config.raise_in_ipagnn = False
    validate_forward_pass(config, info)

  def test_exception_ipagnn(self):
    info = info_lib.get_test_info()
    config = config_lib.get_test_config()
    config.model_class = 'IPAGNN'
    config.raise_in_ipagnn = True
    validate_forward_pass(config, info)

  def test_compressive_ipagnn(self):
    info = info_lib.get_test_info()
    config = config_lib.get_test_config()
    config.model_class = 'IPAGNN'
    config.use_compressive_ipagnn = True
    config.compressive_max_skip = 3
    validate_forward_pass(config, info)

  def test_transformer(self):
    info = info_lib.get_test_info()
    config = config_lib.get_test_config()
    config.model_class = 'Transformer'
    validate_forward_pass(config, info)

  def test_transformer_mil_max(self):
    info = info_lib.get_test_info()
    config = config_lib.get_test_config()
    config.model_class = 'MILTransformer'
    config.mil_pool = 'max'
    validate_forward_pass(config, info)

  def test_transformer_mil_mean(self):
    info = info_lib.get_test_info()
    config = config_lib.get_test_config()
    config.model_class = 'MILTransformer'
    config.mil_pool = 'mean'
    validate_forward_pass(config, info)

  def test_transformer_mil_logsumexp(self):
    info = info_lib.get_test_info()
    config = config_lib.get_test_config()
    config.model_class = 'MILTransformer'
    config.mil_pool = 'logsumexp'
    validate_forward_pass(config, info)

  def test_lstm(self):
    info = info_lib.get_test_info()
    config = config_lib.get_test_config()
    config.model_class = 'LSTM'
    validate_forward_pass(config, info)


if __name__ == '__main__':
  unittest.main()
