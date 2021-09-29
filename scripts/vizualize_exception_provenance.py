"""Runner script."""

import os

from absl import app
from absl import flags

from flax.training import common_utils
import jax
import jax.numpy as jnp
from ml_collections.config_flags import config_flags

from core.data import codenet
from core.data import codenet_paths
from core.data import info as info_lib
from core.data import process
from core.lib import trainer
import tensorflow_datasets as tfds

DEFAULT_DATASET_PATH = codenet_paths.DEFAULT_DATASET_PATH
DEFAULT_CONFIG_PATH = codenet_paths.DEFAULT_CONFIG_PATH


flags.DEFINE_string('dataset_path', DEFAULT_DATASET_PATH, 'Dataset path.')
config_flags.DEFINE_config_file(
    name='config', default=DEFAULT_CONFIG_PATH, help_string='Config file.'
)
FLAGS = flags.FLAGS


def get_raise_contribution_at_step(instruction_pointer, raise_decisions):
  # instruction_pointer.shape: num_nodes
  # raise_decisions.shape: num_nodes, 2
  p_raise = raise_decisions[:, 0]
  print('p_raise.shape')
  print(p_raise.shape)
  print(instruction_pointer.shape)
  raise_contribution = p_raise * instruction_pointer
  # raise_contribution.shape: num_nodes
  return raise_contribution
get_raise_contribution_at_steps = jax.vmap(get_raise_contribution_at_step)


def get_raise_contribution(instruction_pointer, raise_decisions):
  # instruction_pointer.shape: steps, num_nodes
  # raise_decisions.shape: steps, num_nodes, 2
  raise_contributions = get_raise_contribution_at_steps(
      instruction_pointer, raise_decisions)
  # raise_contributions.shape: steps, num_nodes
  raise_contribution = jnp.sum(raise_contributions, axis=0)
  # raise_contribution.shape: num_nodes
  return raise_contribution
get_raise_contribution_batch = jax.vmap(get_raise_contribution)


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
    assert not config.multidevice

    if config.multidevice:
      batch = common_utils.shard(batch)
    problem_ids = batch.pop('problem_id')
    submission_ids = batch.pop('submission_id')
    state, aux = train_step(state, batch)
    print(aux)
    print(aux.keys())
    instruction_pointer = aux['instruction_pointer']
    exit_index = batch['exit_index']
    raise_index = exit_index + 1
    raise_decisions = aux['raise_decisions']
    print('instruction_pointer.shape')
    print(instruction_pointer.shape)
    contributions = get_raise_contribution_batch(instruction_pointer, raise_decisions)
    print(contributions)

    found = False
    for problem_id, submission_id, contribution in zip(problem_ids, submission_ids, contributions):
      problem_id = problem_id[0].decode('utf-8')
      submission_id = submission_id[0].decode('utf-8')
      python_path = codenet.get_python_path(problem_id, submission_id)
      if os.path.exists(python_path):
        found = True
        with open(python_path, 'r') as f:
          source = f.read()
        raw = process.make_rawruntimeerrorproblem(
            source, 'N/A', problem_id=problem_id, submission_id=submission_id)
        print(raw)
        print(source)
        print(contribution)
        print('---')
    if found:
      break

      # TODO(dbieber): Figure out contributions of each node to the exception node.
      # Then load source.
      # And print everything.



if __name__ == '__main__':
  app.run(main)
