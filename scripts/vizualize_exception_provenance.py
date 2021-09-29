"""Runner script."""

from absl import app
from absl import flags

from flax.training import common_utils
import jax
import jax.numpy as jnp
from ml_collections.config_flags import config_flags

from core.data import codenet_paths
from core.data import info as info_lib
from core.lib import trainer
import tensorflow_datasets as tfds

DEFAULT_DATASET_PATH = codenet_paths.DEFAULT_DATASET_PATH
DEFAULT_CONFIG_PATH = codenet_paths.DEFAULT_CONFIG_PATH


flags.DEFINE_string('dataset_path', DEFAULT_DATASET_PATH, 'Dataset path.')
config_flags.DEFINE_config_file(
    name='config', default=DEFAULT_CONFIG_PATH, help_string='Config file.'
)
FLAGS = flags.FLAGS


def exception_provenance(instruction_pointer, raise_index, raise_decisions):
  # instruction_pointer.shape: steps, num_nodes
  # raise_index.shape: scalar.
  # raise_decisions.shape: steps, num_nodes, 2
exception_provenance_batch = jax.vmap(exception_provenance)

def main(argv):
  del argv  # Unused.

  dataset_path = FLAGS.dataset_path
  config = FLAGS.config
  jnp.set_printoptions(threshold=config.printoptions_threshold)
  info = info_lib.get_dataset_info(dataset_path)
  t = trainer.Trainer(config=config, info=info)

  dataset = t.load_dataset(
      dataset_path=dataset_path, split='train', include_strings=True)

  rng = jax.random.PRNGKey(0)
  rng, init_rng = jax.random.split(rng)
  model = t.make_model(deterministic=False)

  state = t.create_train_state(init_rng, model)

  train_step = t.make_train_step()
  for batch in tfds.as_numpy(dataset):
    if config.multidevice:
      batch = common_utils.shard(batch)
    problem_id = batch.pop('problem_id')
    submission_id = batch.pop('submission_id')
    state, aux = train_step(state, batch)
    print(aux)
    print(aux.keys())
    instruction_pointer = aux['instruction_pointer']
    raise_index = batch['raise_index']
    raise_decisions = aux['raise_decisions']
    exception_provenance_batch(instruction_pointer, raise_index, raise_decisions)


    # TODO(dbieber): Figure out contributions of each node to the exception node.
    # Then load source.
    # And print everything.
    break



if __name__ == '__main__':
  app.run(main)
