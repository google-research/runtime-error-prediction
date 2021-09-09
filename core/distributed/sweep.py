import itertools
import random

from core.data import codenet_paths
from core.distributed import gcp


hparams = {
    'config.learning_rate': [0.001, 0.003, 0.01, 0.03, 0.1, 0.3],
    # 'config.rnn_layers': [2, 4]
    'config.hidden_size': [16, 32, 64, 128, 256, 512],
    'config.span_encoding_method': ['first', 'mean', 'max', 'sum'],
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


def choose_commands(n, study_id, name, model_class, raise_in_ipagnn):
  commands = []
  for index, params in enumerate(dict_product(hparams)):
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
        f'--config.run_id={name}{index:03d} '
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
  n = 20  # Machines per model
  study_id = '2021-09-08-experiment-001'

  # Transformer
  offset = 0  # The machine index to start with.
  run_sweep(n, offset, study_id, 'T', 'Transformer', False)

  # IPAGNN
  offset = 20
  run_sweep(n, offset, study_id, 'I', 'IPAGNN', False)

  # IPAGNN
  offset = 40
  run_sweep(n, offset, study_id, 'E', 'IPAGNN', False)  # Exception IPAGNN


# # To kill the runner processes:
# # python -m core.distributed.gcp tpu_run_command 'pkill runner.py' --n=32 --offset=0
# gcp.fix_firewall().wait()
# gcp.tpu_run_command('pkill runner.py', n, offset=offset)


if __name__ == '__main__':
  main()
