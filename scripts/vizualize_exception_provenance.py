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
config_flags.DEFINE_config_file(
    name='config', default=DEFAULT_CONFIG_PATH, help_string='Config file.'
)
FLAGS = flags.FLAGS


def main(argv):
  del argv  # Unused.

  dataset_path = FLAGS.dataset_path
  config = FLAGS.config
  jnp.set_printoptions(threshold=config.printoptions_threshold)
  info = info_lib.get_dataset_info(dataset_path)
  t = trainer.Trainer(config=config, info=info)

  dataset = t.load_dataset(
      dataset_path=dataset_path, split='train', include_strings=True)

  for example in dataset:
    print(example)
    break


if __name__ == '__main__':
  app.run(main)
