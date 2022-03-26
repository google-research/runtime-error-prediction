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

"""Runner script."""

from absl import app
from absl import flags
import jax.numpy as jnp
from ml_collections.config_flags import config_flags

from core.data import codenet_paths
from core.data import info as info_lib
from core.lib import trainer

DEFAULT_DATASET_PATH = codenet_paths.DEFAULT_DATASET_PATH
DEFAULT_CONFIG_PATH = codenet_paths.DEFAULT_CONFIG_PATH


flags.DEFINE_string('dataset_path', DEFAULT_DATASET_PATH, 'Dataset path.')
flags.DEFINE_string('split', 'train', 'Split for training.')
flags.DEFINE_string('mode', 'train', 'Runner mode')
config_flags.DEFINE_config_file(
    name='config', default=DEFAULT_CONFIG_PATH, help_string='Config file.'
)
FLAGS = flags.FLAGS


def main(argv):
  del argv  # Unused.

  dataset_path = FLAGS.dataset_path
  split = FLAGS.split
  config = FLAGS.config
  jnp.set_printoptions(threshold=config.printoptions_threshold)
  info = info_lib.get_dataset_info(dataset_path, config)
  if FLAGS.mode == 'train':
    trainer.Trainer(config=config, info=info).run_train(
        dataset_path=dataset_path, split=split, steps=config.train_steps)
  elif FLAGS.mode == 'test':
    trainer.Trainer(config=config, info=info).run_test(
        dataset_path=dataset_path, split=split)
  else:
    raise ValueError('Unexpected mode', FLAGS.mode)


if __name__ == '__main__':
  app.run(main)
