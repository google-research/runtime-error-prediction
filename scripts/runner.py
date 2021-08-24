"""Runner script."""

import os

from absl import app, flags, logging
from ml_collections.config_flags import config_flags

from core.data import codenet_paths
from core.lib import trainer

DEFAULT_DATASET_PATH = codenet_paths.DEFAULT_DATASET_PATH
DEFAULT_CONFIG_PATH = codenet_paths.DEFAULT_CONFIG_PATH


flags.DEFINE_string('dataset_path', DEFAULT_DATASET_PATH, 'Dataset path.')
config_flags.DEFINE_config_file(
    name='config', default=DEFAULT_CONFIG_PATH, help_string='Config file.'
)
FLAGS = flags.FLAGS


def main(argv):
  del argv  # Unused.

  dataset_path = FLAGS.dataset_path
  config = FLAGS.config
  trainer.Trainer(config=config).run_train(dataset_path=dataset_path)


if __name__ == '__main__':
  app.run(main)
