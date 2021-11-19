from datetime import datetime

import os
import socket
import time

DEFAULT_CONFIG_PATH = 'config/default.py'
DEFAULT_DATASET_PATH = 'datasets/codenet/2021-11-01-f=0.01'
TEST_DATASET_PATH = 'datasets/codenet/2021-11-01-f=0.01'
DEFAULT_TOKENIZER_PATH = 'out/tokenizers/train-1000000.json'
DOCSTRING_TOKENIZER_PATH = 'out/tokenizers/train-docstrings-1000000.json'
DEFAULT_SPLITS_PATH = 'out/splits/default.json'
DEFAULT_EXPERIMENTS_DIR = 'out/experiments'
EXPERIMENT_ID_PATH = 'out/experiment_id.txt'

FULL_DATASET_PATH = '/mnt/runtime-error-problems-experiments/datasets/project-codenet/2021-10-07-full'
FULL_DATASET_PATH_WITH_DOCSTRINGS = '/mnt/runtime-error-problems-experiments/datasets/project-codenet/2021-11-17'
# Raw control_flow_programs data pattern:
DEFAULT_CFP_DATA_PATTERN = '/mnt/runtime-error-problems-experiments/datasets/control_flow_programs/decimal-large-state-L10/0.0.48/control_flow_programs-train.tfrecord-*'
# Processed control_flow_programs dataset path:
DEFAULT_CFP_DATASET_PATH = '/mnt/runtime-error-problems-experiments/datasets/control_flow_programs/processed/decimal-large-state-L10/0.0.48-002/'

RAW_CFP_RAISE_DATA_PATH = '/mnt/runtime-error-problems-experiments/datasets/control_flow_programs_raise/decimal-large-state-L30/2021-10-19-001/synthetic-20211018-001.tfrecord'
DEFAULT_CFP_RAISE_DATASET_PATH = '/mnt/runtime-error-problems-experiments/datasets/control_flow_programs_raise/processed/decimal-large-state-L30/2021-10-19-001/'

DATA_ROOT = '/mnt/disks/project-codenet-data/Project_CodeNet/'
OUT_ROOT = '/mnt/disks/project-codenet-data/out/'
EVALS_ROOT = '/mnt/disks/project-codenet-data/out/evals'
FILE_DIRNAME = os.path.dirname(__file__)
ERROR_CHECKER = os.path.join(FILE_DIRNAME, 'error-checker.py')

PERSONAL_ACCESS_TOKEN_PATH = ''

PYTHON3 = '/usr/bin/python3'
HOSTNAME = socket.gethostname()
SHORT_HOSTNAME = HOSTNAME
if HOSTNAME == 'dbieber-macbookpro.roam.corp.google.com':
  PYTHON3 = '/Users/dbieber/.virtualenvs/_3/bin/python'
  DATA_ROOT = '/Users/dbieber/code/github/googleprivate/compressive-ipagnn/data/Project_CodeNet'
  SHORT_HOSTNAME = 'dbieber-mac'
elif HOSTNAME == 'dbieber-macbookpro4.roam.corp.google.com':
  PYTHON3 = '/Users/dbieber/.virtualenvs/_3/bin/python'
  PERSONAL_ACCESS_TOKEN_PATH = '/Users/dbieber/secrets/colab_github_access_token'
  SHORT_HOSTNAME = 'dbieber-mac4'
elif HOSTNAME == 'code-executor-001':
  PYTHON3 = '/home/dbieber/_39/bin/python'
  DATA_ROOT = '/mnt/disks/project-codenet-data/Project_CodeNet/'
  EVALS_ROOT = '/mnt/disks/project-codenet-data/out/evals'
  OUT_ROOT = '/mnt/disks/project-codenet-data/out'
elif HOSTNAME == 'dev-000':
  PYTHON3 = '/home/dbieber/compressive-ipagnn/ipagnn/bin/python'
  DATA_ROOT = '/mnt/disks/project-codenet-data/Project_CodeNet/'
  EVALS_ROOT = '/mnt/disks/project-codenet-data/out/evals'
  OUT_ROOT = '/mnt/disks/project-codenet-data/out/'
elif HOSTNAME.startswith('t1v-'):  # TPU
  PYTHON3 = '/usr/bin/python3'
  FULL_DATASET_PATH = '/mnt/runtime-error-problems-experiments/datasets/project-codenet/2021-10-07-full'
  DATA_ROOT = '/mnt/project-codenet-storage/Project_CodeNet/'
  EVALS_ROOT = '/mnt/project-codenet-storage/out/evals'
  OUT_ROOT = '/mnt/project-codenet-storage/out'

# On TPUs, this we mount the GCS bucket "runtime-error-problems-experiments"
# at /mnt/runtime-error-problems-experiments.
GCS_EXPERIMENT_DIR = '/mnt/runtime-error-problems-experiments/experiments'
if os.path.exists(GCS_EXPERIMENT_DIR):
  DEFAULT_EXPERIMENTS_DIR = GCS_EXPERIMENT_DIR

CLOUD_DATA_ROOT = 'gs://project-codenet/'


def make_tfrecord_path(dataset_path, split):
  return os.path.join(dataset_path, f'{split}.tfrecord')


def make_ids_path(tfrecord_path):
  return tfrecord_path.replace('.tfrecord', '-ids.json')


def make_experiment_id():
  now = datetime.now()
  date_str = now.strftime('%Y%m%d')
  milliseconds = int(round(time.time() * 1000))
  return f'{date_str}-{milliseconds}'


def make_run_id():
  return ''


def make_run_dir(study_id, exp_id, run_id, experiments_dir=None):
  experiments_dir = experiments_dir or DEFAULT_EXPERIMENTS_DIR
  if study_id:
    return os.path.join(experiments_dir, study_id, exp_id, run_id)
  elif run_id:
    return os.path.join(experiments_dir, exp_id, run_id)
  else:
    return os.path.join(experiments_dir, exp_id)


def make_checkpoints_path(run_dir):
  return os.path.join(run_dir, 'checkpoints')


def make_top_checkpoints_path(run_dir):
  return os.path.join(run_dir, 'top-checkpoints')


def make_log_dir(run_dir, split='train'):
  return os.path.join(run_dir, split)


def make_metadata_path(run_dir):
  metadata_path = os.path.join(run_dir, 'metadata.json')
  if os.path.exists(metadata_path):
    # Don't overwrite existing metadata on restore attempts.
    metadata_path = os.path.join(run_dir, f'metadata-{int(time.time())}.json')
  return metadata_path


def get_personal_access_token():
  with open(PERSONAL_ACCESS_TOKEN_PATH, 'r') as f:
    return f.read().strip()


def get_problem_description_path(problem_id):
  return f'{DATA_ROOT}/problem_descriptions/{problem_id}.html'


def get_problem_docstring_path(problem_id):
  return f'{OUT_ROOT}/docstrings/{problem_id}.txt'
