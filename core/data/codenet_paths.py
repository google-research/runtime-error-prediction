import os
import socket

DEFAULT_DATASET_PATH = 'out/data/default'
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
elif HOSTNAME == 'dev-000':
  PYTHON3 = '/home/dbieber/compressive-ipagnn/ipagnn/bin/python'
  DATA_ROOT = '/home/veetee/Project_CodeNet/'

CLOUD_DATA_ROOT = 'gs://project-codenet/'

def make_tfrecord_path(dataset_path, split):
  return os.path.join(dataset_path, f'{split}.tfrecord')


def make_ids_path(tfrecord_path):
  return tfrecord_path.replace('.tfrecord', '-ids.json')
