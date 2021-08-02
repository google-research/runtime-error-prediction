# Copyright 2021 The Flax Authors.
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

"""Tests for flax.examples.imagenet.train."""

import pathlib
import tempfile

from absl.testing import absltest

import jax
from jax import random

import tensorflow as tf
import tensorflow_datasets as tfds

# Local imports.
import models
from trainers import default_trainer
from config import default as default_lib
from lib import setup
from core.data import error_kinds

jax.config.update('jax_disable_most_optimizations', True)
NUM_CLASSES = error_kinds.NUM_CLASSES

class TrainTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    tf.config.experimental.set_visible_devices([], 'GPU')

  def test_create_model(self):
    """Tests creating model."""
    config = default_lib.get_config()
    rng = jax.random.PRNGKey(config.seed_id)
    rng, init_rng = jax.random.split(rng)
    model = setup.create_model(config)
    fake_input =  setup.create_fake_inputs(config.dataset.batch_size, config.dataset.max_tokens, config.dataset.max_length, config.model.name)
    train_state = setup.create_train_state(init_rng, model, fake_input, config)
    model = setup.create_model(config)
    y = model.apply({'params': train_state.params}, fake_input, config)
    self.assertEqual(y.shape, (config.dataset.batch_size, NUM_CLASSES))



if __name__ == '__main__':
  absltest.main()
