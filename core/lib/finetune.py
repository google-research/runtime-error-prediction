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

from flax.core import frozen_dict
from flax.training import checkpoints


def finetune_from_lstm(state, restore_checkpoint_dir, config):
  """Updates the current state for fine-tuning from a pre-trained LSTM.

  It updates state using some of the values from the restore_checkpoint_dir.

  The Transformer node_span_encoder, as well as the IPA-GNN lstm weights.

  Branch decisions and raise decisions weights are not going to be loaded.
  """
  old_state = checkpoints.restore_checkpoint(config.restore_checkpoint_dir, None)
  old_params = old_state['params']

  state = state.replace(step=int(old_state['step']))
  state = state.replace(rng=old_state['rng'])

  params = state.params
  params_copy = params.unfreeze()
  params_copy['node_span_encoder'] = old_params['input_embedder']

  for n in range(config.rnn_layers):
    params_copy['ipagnn']['ipagnn_layer_scan'][f'lstm_{n}'] = old_params['encoder'][f'lstm_{n}']['OptimizedLSTMCell_0']

  state = state.replace(params=frozen_dict.FrozenDict(params_copy))
  return state


def finetune_from_ipagnn(state, restore_checkpoint_dir, config):
  """Updates the current state for fine-tuning from a pre-trained IPAGNN.

  It updates state using some of the values from the restore_checkpoint_dir.

  The Transformer node_span_encoder, as well as the IPA-GNN lstm weights and
  branch decision weights are loaded.

  Args:
    state: The current method's state.
    restore_checkpoint_dir: The directory to load the IPAGNN checkpoint from.
    config: The experiment config.
  Returns:
    A new version of state, with parameters taken from the pre-trained IPAGNN.
  """
  old_state = checkpoints.restore_checkpoint(config.restore_checkpoint_dir, None)
  state = state.replace(step=int(old_state['step']))
  # state = state.replace(opt_state=old_state['opt_state'])
  state = state.replace(rng=old_state['rng'])
  params = state.params
  old_params = old_state['params']
  key_paths = [
      # Note we omit loading the output layer weights.
      ('node_span_encoder',),
      ('ipagnn', 'ipagnn_layer_scan', 'branch_decide_dense',),
  ] + [
      ('ipagnn', 'ipagnn_layer_scan', f'lstm_{n}',)
      for n in range(config.rnn_layers)
  ]
  params_copy = params.unfreeze()
  for key_path in key_paths:
    params_component = params_copy
    old_params_component = old_params
    for key_path_component in key_path[:-1]:
      params_component = params_component[key_path_component]
      old_params_component = old_params_component[key_path_component]

    params_component[key_path[-1]] = old_params_component[key_path[-1]]
  state = state.replace(params=frozen_dict.FrozenDict(params_copy))
  return state
