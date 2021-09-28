from flax.core import frozen_dict
from flax.training import checkpoints


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
