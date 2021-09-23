import itertools
import random

import fire

from core.data import codenet_paths
from core.distributed import gcp


hparams = {
    'config.batch_size': [32],
    'config.learning_rate': [
        # 1e-5, 3e-5,
        # 1e-4, 3e-4,
        0.001, 0.003,
        0.01, 0.03,
        0.1, 0.3,
    ],
    # 'config.rnn_layers': [2, 4]
    'config.grad_clip_value': [0, 0.5, 1, 2],
    'config.hidden_size': [16, 32, 64, 128, 256, 512],
    'config.span_encoding_method': ['first', 'mean', 'max', 'sum'],
    'config.transformer_dropout_rate': [0, 0.1, 0.3],
    'config.transformer_attention_dropout_rate': [0, 0.1, 0.3],
    'config.permissive_node_embeddings': [True, False],
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


def make_run_id(name, index, params):
  param_name_mapping = {
      'batch_size': 'bs',
      'learning_rate': 'lr',
      'rnn_layers': 'L',
      'grad_clip_value': 'gc',
      'hidden_size': 'hs',
      'span_encoding_method': 'span',
      'transformer_size': 'T',
      'permissive_node_embeddings': 'pe',
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


def choose_commands(n, experiment_id, study_id, name, model_class, raise_in_ipagnn, dataset_path):
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
        f'--dataset_path={dataset_path} '
        '--config.eval_freq=5000 '
        '--config.eval_subsample=1 '
        '--config.eval_max_batches=2500 '
        '--config.save_freq=5000 '
        f'--config.study_id={study_id} '
        f'--config.experiment_id={experiment_id} '
        f'--config.run_id={run_id} '
        + ' '.join(flags)
        + ' > out/stdout.txt 2> out/stderr.txt'
    )
    command = f'tmux new -d -s remote "{command}"'
    command = f'pgrep -l runner.py; if [ $? -ne 0 ]; then {command}; else echo "Skipping"; fi'
    commands.append(command)
  commands = random.sample(commands, n)
  return commands


def run_sweep(n, offset, experiment_id, study_id, name, model_class, raise_in_ipagnn, dataset_path):
  commands = choose_commands(n, experiment_id, study_id, name, model_class, raise_in_ipagnn, dataset_path)

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


def get_and_increment_global_experiment_id():
  # Increment the global experiment id.
  with open(codenet_paths.EXPERIMENT_ID_PATH, 'r') as f:
    experiment_id = int(f.read().strip()) + 1
  with open(codenet_paths.EXPERIMENT_ID_PATH, 'w') as f:
    f.write(str(experiment_id))
  return experiment_id


def main(experiment_id=None, study_id=None, pretrain=False):
  """Runs a sweep.

  To restart any failed jobs in an existing sweep, call this with the experiment_id
  of the sweep.
  This will rerun the start commands on each of the TPUs. If the job is already
  running (because it has not failed), the command will just print "Skipping".
  If the command had failed, this will restart it from where it left off.

  Args:
    experiment_id: If set, the sweep will use this experiment id. Otherwise the id
      will be chosen automatically. If the same experiment and study id as an existing
      study are used, the commands may be the same as those used previously.
      This can be used to resume failed training jobs.
    study_id: The study_id to use for the experiment sweep.
  """
  random.seed(0)

  if pretrain:
    dataset_path = codenet_paths.DEFAULT_CFP_DATASET_PATH
  else:
    dataset_path = codenet_paths.FULL_DATASET_PATH
  if experiment_id is None:
    experiment_id = get_and_increment_global_experiment_id()

  n = 20  # Machines per model

  # Exception IPAGNN
  # offset = 0
  # run_sweep(n, offset, experiment_id, study_id, 'E', 'IPAGNN', True, dataset_path)  # Exception IPAGNN

  # IPAGNN
  offset = 0
  run_sweep(n, offset, experiment_id, study_id, 'I', 'IPAGNN', False, dataset_path)

  # Transformer
  # offset = 40  # The machine index to start with.
  # run_sweep(n, offset, experiment_id, study_id, 'T', 'Transformer', False, dataset_path)


# # To kill the runner processes:
# # python -m core.distributed.gcp tpu_run_command 'pkill runner.py && pkill tmux' --n=60 --offset=0
# gcp.fix_firewall().wait()
# gcp.tpu_run_command('pkill runner.py && pkill tmux', n, offset=offset)


if __name__ == '__main__':
  fire.Fire(main)
