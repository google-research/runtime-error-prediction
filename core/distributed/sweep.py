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

commands = []
for index, params in enumerate(dict_product(hparams)):
  flags = []
  for key, value in params.items():
    flags.append(f'--{key}={value}')
  command = (
      'cd compressive-ipagnn && '
      'python3 -m scripts.runner '
      '--config.model_class=IPAGNN '
      '--config.raise_in_ipagnn=True '
      '--config.batch_size=8 '
      '--dataset_path=/mnt/runtime-error-problems-experiments/datasets/project-codenet/full-noudf-ids '
      '--config.eval_freq=50000 '
      '--config.eval_subsample=1 '
      '--config.eval_max_batches=2500 '
      '--config.save_freq=25000 '
      '--config.study_id=2021-09-07-debugging-fleet '
      f'--config.experiment_id={experiment_id} '
      f'--config.run_id={index} '
      + ' '.join(flags)
  )
  commands.append(command)


# Calculate number of TPUs needed for sweep.
n = 4
commands = random.sample(commands, n)

def make_run_command(index):
  return commands[index]


# Ensure TPUs are up and unused.
print(f'Starting {n} TPUs')
gcp.tpu_up_n(n)
gcp.fix_firewall().wait()

access_token = codenet_paths.get_personal_access_token()
gcp.tpu_run_script(
    'scripts/setup-tpu.sh', n, {
        'PERSONAL_ACCESS_TOKEN': access_token
    }
)

gcp.fix_firewall().wait()
gcp.tpu_run_commands(make_run_command, n)
