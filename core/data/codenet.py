import fire

import os
import shutil
import socket
import subprocess

DATA_ROOT = '/mnt/disks/project-codenet-data/Project_CodeNet/'
FILE_DIRNAME = os.path.dirname(__file__)
ERROR_CHECKER = os.path.join(FILE_DIRNAME, 'error-checker.py')

PIPE = subprocess.PIPE
PYTHON3 = '/usr/bin/python3'
HOSTNAME = socket.gethostname()
if HOSTNAME == 'dbieber-macbookpro.roam.corp.google.com':
  PYTHON3 = '/Users/dbieber/.virtualenvs/_3/bin/python'
  DATA_ROOT = '/Users/dbieber/code/github/googleprivate/compressive-ipagnn/data/Project_CodeNet'
elif HOSTNAME == 'code-executor-001':
  PYTHON3 = '/home/dbieber/_39/bin/python'
  DATA_ROOT = '/mnt/disks/project-codenet-data/Project_CodeNet/'


def get_all_problem_ids():
  problem_dir = os.path.join(DATA_ROOT, 'data')
  return os.listdir(problem_dir)


def get_all_submission_ids(problem_id):
  submission_dir = os.path.join(DATA_ROOT, 'data', problem_id, 'Python')
  if os.path.exists(submission_dir):
    submission_filenames = os.listdir(submission_dir)
    submission_ids = [filename.split('.')[0] for filename in submission_filenames]
    return submission_ids
  else:
    return []


def get_all_problem_and_submission_ids():
  for problem_id in get_all_problem_ids():
    for submission_id in get_all_submission_ids(problem_id):
      yield problem_id, submission_id


def get_python_path(problem_id, submission_id):
  return os.path.join(DATA_ROOT, 'data', problem_id, 'Python', f'{submission_id}.py')


def get_input_path(problem_id, submission_id):
  return os.path.join(DATA_ROOT, 'derived', 'input_output', 'data', problem_id, 'input.txt')


def get_output_path(problem_id, submission_id):
  return os.path.join(DATA_ROOT, 'derived', 'input_output', 'data', problem_id, 'input.txt')


def run_for_errors(problem_id, submission_id, skip_existing=True):
  """Runs the command in the error-checker subprocess."""
  out_dir = os.path.join('out', 'evals', problem_id, submission_id)
  if os.path.exists(out_dir):
    if skip_existing:
      return
    shutil.rmtree(out_dir)
  os.makedirs(out_dir)
  python_filepath = get_python_path(problem_id, submission_id)
  input_filepath = get_input_path(problem_id, submission_id)

  if not os.path.exists(input_filepath):
    return

  error_path = os.path.join(out_dir, 'error.txt')
  timeout_path = os.path.join(out_dir, 'timeout.txt')
  stdout_path = os.path.join(out_dir, 'stdout.txt')
  stderr_path = os.path.join(out_dir, 'stderr.txt')
  command = [PYTHON3, ERROR_CHECKER, 'run_for_errors', python_filepath, error_path]
  try:
    subprocess.run(
        command,
        input=open(input_filepath, 'rb').read(),
        stderr=open(stderr_path, 'wb'),
        stdout=open(stdout_path, 'wb'),
        timeout=1,
    )
  except subprocess.TimeoutExpired as e:
    with open(timeout_path, 'w') as f:
      f.write(str(e) + '\n')
  stdout = open(stdout_path, 'r').read()
  stderr = open(stderr_path, 'r').read()
  return stdout


if __name__ == '__main__':
  fire.Fire()
