import itertools
import random

from core.data import codenet_paths
from core.distributed import gcp


hparams = {
    'config.learning_rate': [
        1e-5, 3e-5, 1e-4, 3e-4, 0.001, 0.003,
        # 0.01, 0.03, 0.1, 0.3,
    ],
    # 'config.rnn_layers': [2, 4]
    'config.grad_clip_value': [0, 0.5, 1, 2],
    'config.hidden_size': [16, 32, 64, 128, 256, 512],
    'config.span_encoding_method': ['first', 'mean', 'max', 'sum'],
    'config.transformer_dropout_rate': [0, 0.1, 0.3],
    'config.transformer_attention_dropout_rate': [0, 0.1, 0.3],
    'transformer_size': ['tiny', 'small', 'default']
}

transformer_configs = {
    'default': {
        'config.transformer_emb_dim': 512,
        'config.transformer_num_heads': 8,
        'config.transformer_num_layers': 6,
        'config.transformer_qkv_dim': 512,
        'config.transformer_mlp_dim': 2048,
    },
    'small': {
        'config.transformer_emb_dim': 256,
        'config.transformer_num_layers': 2,
        'config.transformer_qkv_dim': 256,
        'config.transformer_num_heads': 4,
        'config.transformer_mlp_dim': 1024,
    },
    'tiny': {
        'config.transformer_emb_dim': 128,
        'config.transformer_num_layers': 2,
        'config.transformer_qkv_dim': 128,
        'config.transformer_num_heads': 4,
        'config.transformer_mlp_dim': 512,
    },
}


def dict_product(d):
  keys = d.keys()
  values = d.values()
  for specific_values in itertools.product(*values):
    yield dict(zip(keys, specific_values))


# Increment the global experiment id.
with open(codenet_paths.EXPERIMENT_ID_PATH, 'r') as f:
  experiment_id = int(f.read().strip()) + 1
with open(codenet_paths.EXPERIMENT_ID_PATH, 'w') as f:
  f.write(str(experiment_id))


def make_run_id(name, index, params):
  param_name_mapping = {
      'learning_rate': 'lr',
      'rnn_layers': 'L',
      'grad_clip_value': 'gc',
      'hidden_size': 'hs',
      'span_encoding_method': 'span',
      'transformer_size': 'T',
  }
  parts = []
  for key, value in params.items():
    # Strip 'config.' from the key.
    if key.startswith('config.'):
      key = key[len('config.'):]
    if key in param_name_mapping:
      key = param_name_mapping[key]
    else:
      key = ''.join(word[0] for word in key.split('_'))
    parts.append(f'{key}={value}')
  return f'{name}{index:03d},{",".join(parts)}'


def choose_commands(n, study_id, name, model_class, raise_in_ipagnn):
  commands = []
  for index, params in enumerate(dict_product(hparams)):
    run_id = make_run_id(name, index, params)
    if 'transformer_size' in params:
      transformer_size = params.pop('transformer_size')
      params.update(transformer_configs[transformer_size])

    flags = []
    for key, value in params.items():
      flags.append(f'--{key}={value}')
    command = (
        'cd compressive-ipagnn && '
        'python3 -m scripts.runner '
        f'--config.model_class={model_class} '
        f'--config.raise_in_ipagnn={raise_in_ipagnn} '
        '--config.batch_size=8 '
        '--dataset_path=/mnt/runtime-error-problems-experiments/datasets/project-codenet/full-noudf-ids '
        '--config.eval_freq=50000 '
        '--config.eval_subsample=1 '
        '--config.eval_max_batches=2500 '
        '--config.save_freq=25000 '
        f'--config.study_id={study_id} '
        f'--config.experiment_id={experiment_id} '
        f'--config.run_id={run_id} '
        + ' '.join(flags)
    )
    command = f'tmux new -d -s remote "{command}"'
    command = f'pgrep -l runner.py; if [ $? -ne 0 ]; then {command}; else echo "Skipping"; fi'
    commands.append(command)
  commands = random.sample(commands, n)
  return commands


def run_sweep(n, offset, study_id, name, model_class, raise_in_ipagnn):
  commands = choose_commands(n, study_id, name, model_class, raise_in_ipagnn)

  def make_run_command(index):
    return commands[index - offset]

  # Ensure TPUs are up and unused.
  print(f'Starting {n} TPUs')
  gcp.tpu_up_n(n, offset=offset)
  gcp.fix_firewall().wait()

  access_token = codenet_paths.get_personal_access_token()
  gcp.tpu_run_script(
      'scripts/setup-tpu.sh', n, {
          'PERSONAL_ACCESS_TOKEN': access_token
      }, offset=offset
  )

  gcp.fix_firewall().wait()
  gcp.tpu_run_commands(make_run_command, n, offset=offset)


def main():
  n = 1  # Machines per model
  study_id = '2021-09-10-transformer-size-dev'

  # Transformer
  offset = 1  # The machine index to start with.
  run_sweep(n, offset, study_id, 'T', 'Transformer', False)

  # IPAGNN
  offset = 2
  run_sweep(n, offset, study_id, 'I', 'IPAGNN', False)

  # Exception IPAGNN
  offset = 3
  run_sweep(n, offset, study_id, 'E', 'IPAGNN', True)  # Exception IPAGNN


# # To kill the runner processes:
# # python -m core.distributed.gcp tpu_run_command 'pkill runner.py && pkill tmux' --n=32 --offset=0
# gcp.fix_firewall().wait()
# gcp.tpu_run_command('pkill runner.py && pkill tmux', n, offset=offset)


if __name__ == '__main__':
  main()
