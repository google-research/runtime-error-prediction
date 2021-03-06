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

"""Exception provenance visualization script.

Run the following to mount the Project CodeNet raw data and derived outputs to disk.
gcsfuse --implicit-dirs project-codenet-storage /mnt/project-codenet-storage/
"""

import os

from absl import app
from absl import flags

from flax.training import checkpoints
from flax.training import common_utils
import imageio
import jax
import jax.numpy as jnp
from ml_collections.config_flags import config_flags

from core.data import codenet
from core.data import codenet_paths
from core.data import error_kinds
from core.data import info as info_lib
from core.data import process
from core.lib import metrics
from core.lib import trainer
import tensorflow_datasets as tfds

DEFAULT_DATASET_PATH = codenet_paths.DEFAULT_DATASET_PATH
DEFAULT_CONFIG_PATH = codenet_paths.DEFAULT_CONFIG_PATH


flags.DEFINE_string('dataset_path', DEFAULT_DATASET_PATH, 'Dataset path.')
config_flags.DEFINE_config_file(
    name='config', default=DEFAULT_CONFIG_PATH, help_string='Config file.'
)
FLAGS = flags.FLAGS


def get_raise_contribution_at_step(instruction_pointer, raise_decisions, raise_index):
  # instruction_pointer.shape: num_nodes
  # raise_decisions.shape: num_nodes, 2
  # raise_index.shape: scalar.
  p_raise = raise_decisions[:, 0]
  raise_contribution = p_raise * instruction_pointer
  # raise_contribution.shape: num_nodes
  raise_contribution = raise_contribution.at[raise_index].set(0)
  return raise_contribution
get_raise_contribution_at_steps = jax.vmap(get_raise_contribution_at_step, in_axes=(0, 0, None))


def get_raise_contribution(instruction_pointer, raise_decisions, raise_index, step_limit):
  # instruction_pointer.shape: steps, num_nodes
  # raise_decisions.shape: steps, num_nodes, 2
  # raise_index.shape: scalar.
  # step_limit.shape: scalar.
  raise_contributions = get_raise_contribution_at_steps(
      instruction_pointer, raise_decisions, raise_index)
  # raise_contributions.shape: steps, num_nodes
  mask = jnp.arange(instruction_pointer.shape[0]) < step_limit
  # mask.shape: steps
  raise_contributions = jnp.where(mask[:, None], raise_contributions, 0)
  raise_contribution = jnp.sum(raise_contributions, axis=0)
  # raise_contribution.shape: num_nodes
  return raise_contribution
get_raise_contribution_batch = jax.vmap(get_raise_contribution)


def print_spans(raw):
  span_starts = raw.node_span_starts
  span_ends = raw.node_span_ends
  for i, (span_start, span_end) in enumerate(zip(span_starts, span_ends)):
    print(f'Span {i}: {raw.source[span_start:span_end]}')


def set_config(config):
  """This function is hard-coded to load a particular checkpoint.

  It also sets the model part of the config to match the config of that checkpoint.
  Everything related to parameter construction must match.
  """
  config.multidevice=False
  config.batch_size=2
  config.raise_in_ipagnn=True
  config.restore_checkpoint_dir=(
      '/mnt/runtime-error-problems-experiments/experiments/2021-09-24-pretrain-004-copy/6/'
      'I1466,o=sgd,bs=32,lr=0.3,gc=2,hs=256,span=max,'
      'tdr=0,tadr=0,pe=False,T=default/checkpoints/'
  )
  config.optimizer = 'sgd'
  config.hidden_size = 256
  config.span_encoding_method = 'max'
  config.permissive_node_embeddings = False
  config.transformer_emb_dim = 512
  config.transformer_num_heads = 8
  config.transformer_num_layers = 6
  config.transformer_qkv_dim = 512
  config.transformer_mlp_dim = 2048

  config.restore_checkpoint_dir=(
      '/mnt/runtime-error-problems-experiments/experiments/2021-10-11-finetune-006-copy/10/'
      'E122,o=sgd,bs=32,lr=0.3,gc=2,hs=256,span=max,'
      'tdr=0.1,tadr=0,pe=False,T=default/checkpoints'
  )
  config.span_encoding_method = 'max'

  config.restore_checkpoint_dir=(
      '/mnt/runtime-error-problems-experiments/experiments/2021-10-25-cfp-raise-001/32/'
      'I2292,o=sgd,bs=32,lr=0.3,gc=2,hs=256,span=sum,tdr=0,tadr=0,pe=False,T=tiny'
      '/checkpoints'
  )
  config.span_encoding_method = 'sum'
  return config


def main(argv):
  del argv  # Unused.

  dataset_path = FLAGS.dataset_path
  config = FLAGS.config
  config = set_config(config)

  jnp.set_printoptions(threshold=config.printoptions_threshold)
  info = info_lib.get_dataset_info(dataset_path, config)
  t = trainer.Trainer(config=config, info=info)

  split = 'valid'
  dataset = t.load_dataset(
      dataset_path=dataset_path, split=split, include_strings=True)

  # Initialize / Load the model state.
  rng = jax.random.PRNGKey(0)
  rng, init_rng = jax.random.split(rng)
  model = t.make_model(deterministic=False)
  state = t.create_train_state(init_rng, model)
  if config.restore_checkpoint_dir:
    state = checkpoints.restore_checkpoint(config.restore_checkpoint_dir, state)

  train_step = t.make_train_step()
  for batch in tfds.as_numpy(dataset):
    assert not config.multidevice
    # We do not allow multidevice in this script.
    # if config.multidevice:
    #   batch = common_utils.shard(batch)
    problem_ids = batch.pop('problem_id')
    submission_ids = batch.pop('submission_id')
    state, aux = train_step(state, batch)

    instruction_pointer = aux['instruction_pointer_orig']
    # instruction_pointer.shape: steps, batch_size, num_nodes
    instruction_pointer = jnp.transpose(instruction_pointer, [1, 0, 2])
    # instruction_pointer.shape: batch_size, steps, num_nodes
    exit_index = batch['exit_index']
    raise_index = exit_index + 1
    raise_decisions = aux['raise_decisions']
    # raise_decisions.shape: steps, batch_size, num_nodes, 2
    raise_decisions = jnp.transpose(raise_decisions, [1, 0, 2, 3])
    # raise_decisions.shape: batch_size, steps, num_nodes, 2
    contributions = get_raise_contribution_batch(instruction_pointer, raise_decisions, raise_index, batch['step_limit'])
    # contributions.shape: batch_size, num_nodes

    for index, (problem_id, submission_id, contribution) \
        in enumerate(zip(problem_ids, submission_ids, contributions)):
      problem_id = problem_id[0].decode('utf-8')
      submission_id = submission_id[0].decode('utf-8')
      python_path = codenet.get_python_path(problem_id, submission_id)
      r_index = int(raise_index[index])
      num_nodes = int(raise_index[index]) + 1
      target = int(batch['target'][index])
      target_error = error_kinds.to_error(target)
      prediction = int(jnp.argmax(aux['logits'][index]))
      prediction_error = error_kinds.to_error(prediction)
      step_limit = batch['step_limit'][index, 0]
      instruction_pointer_single = instruction_pointer[index]

      total_contribution = jnp.sum(contribution)
      actual_value = instruction_pointer[index, -1, r_index]
      max_contributor = int(jnp.argmax(contribution))
      max_contribution = contribution[max_contributor]

      # Not all submissions are in the copy of the dataset in gs://project-codenet-data.
      # So we only visualize those that are in the copy.
      if os.path.exists(python_path):
        found = True
        with open(python_path, 'r') as f:
          source = f.read()
        error_lineno = codenet.get_error_lineno(problem_id, submission_id)
        raw = process.make_rawruntimeerrorproblem(
            source, target,
            target_lineno=error_lineno, problem_id=problem_id, submission_id=submission_id)

        # Visualize the data.
        print('---')
        print(f'Problem: {problem_id} {submission_id} ({split})')
        print(f'Batch index: {index}')
        print(f'Target: {target} ({target_error})')
        print(f'Prediction: {prediction} ({prediction_error})')
        print()
        print(source.strip() + '\n')
        print_spans(raw)
        print(contribution[:num_nodes])
        print(f'Main contributor: Node {max_contributor} ({max_contribution})')
        print(f'Total contribution: {total_contribution} (Actual: {actual_value})')

        instruction_pointer_single_trim = instruction_pointer_single[:step_limit + 1, :num_nodes].T
        # instruction_pointer_single_trim.shape: num_nodes, timesteps
        image = metrics.instruction_pointer_to_image(instruction_pointer_single_trim)
        imageio.imwrite('viz-instruction-pointer.png', image, format='png')
        with open('viz-source.txt', 'w') as f:
          f.write(source)

        if error_lineno:
          nodes_at_error = process.get_nodes_at_lineno(raw, error_lineno)
          print(f'Error lineno: {error_lineno} (nodes {nodes_at_error})')
          print(source.split('\n')[error_lineno - 1])  # -1 for line index.

        # Wait for the user to press enter, then continue visualizing.
        input()


if __name__ == '__main__':
  app.run(main)
