import os
import socket

DEFAULT_DATASET_PATH = 'out/data/default.tfrecord'
DEFAULT_TOKENIZER_PATH = 'out/tokenizers/default.json'
DEFAULT_SPLITS_PATH = 'out/splits/default.json'

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


def make_split_path(dataset_path, split):
  dataset_basename = os.path.basename(dataset_path)
  dataset_dir = os.path.dirname(dataset_path)
  split_dir = os.path.join(dataset_dir, split)
  os.makedirs(split_dir, exist_ok=True)
  return os.path.join(split_dir, dataset_basename)
