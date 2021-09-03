from datetime import datetime

import os
import socket
import time

DEFAULT_CONFIG_PATH = 'config/default.py'
DEFAULT_DATASET_PATH = 'datasets/codenet/f=0.01-noudf'
DEFAULT_TOKENIZER_PATH = 'out/tokenizers/train-1000000.json'
DEFAULT_SPLITS_PATH = 'out/splits/default.json'
DEFAULT_EXPERIMENTS_DIR = 'out/experiments'

DATA_ROOT = '/mnt/disks/project-codenet-data/Project_CodeNet/'
EVALS_ROOT = '/mnt/disks/project-codenet-data/out/evals'
FILE_DIRNAME = os.path.dirname(__file__)
ERROR_CHECKER = os.path.join(FILE_DIRNAME, 'error-checker.py')

PYTHON3 = '/usr/bin/python3'
HOSTNAME = socket.gethostname()
if HOSTNAME == 'dbieber-macbookpro.roam.corp.google.com':
  PYTHON3 = '/Users/dbieber/.virtualenvs/_3/bin/python'
  DATA_ROOT = '/Users/dbieber/code/github/googleprivate/compressive-ipagnn/data/Project_CodeNet'
elif HOSTNAME == 'code-executor-001':
  PYTHON3 = '/home/dbieber/_39/bin/python'
  DATA_ROOT = '/mnt/disks/project-codenet-data/Project_CodeNet/'
  EVALS_ROOT = '/mnt/disks/project-codenet-data/out/evals'
elif HOSTNAME == 'dev-000':
  PYTHON3 = '/home/dbieber/compressive-ipagnn/ipagnn/bin/python'
  DATA_ROOT = '/home/veetee/Project_CodeNet/'
  EVALS_ROOT = '/home/veetee/out/evals'

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


def make_experiment_path(exp_id):
  return os.path.join(DEFAULT_EXPERIMENTS_DIR, exp_id)


def make_checkpoints_path(exp_id):
  experiment_path = make_experiment_path(exp_id)
  return os.path.join(experiment_path, 'checkpoints')


def make_log_dir(exp_id):
  experiment_path = make_experiment_path(exp_id)
  return os.path.join(experiment_path, 'logs')
