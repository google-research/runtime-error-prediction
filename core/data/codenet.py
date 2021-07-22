from typing import Any, List, Text

import fire

import dataclasses
import os
import shutil
import socket
import subprocess

from core.data import error_kinds


DATA_ROOT = '/mnt/disks/project-codenet-data/Project_CodeNet/'
EVALS_ROOT = '/mnt/disks/project-codenet-data/out/evals'
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


def get_all_problem_ids_with_evals():
  return os.listdir(EVALS_ROOT)


def get_all_submission_ids_with_evals(problem_id):
  evals_dir = os.path.join(EVALS_ROOT, problem_id)
  if os.path.exists(evals_dir):
    return os.listdir(evals_dir)
  else:
    return []


def get_all_problem_and_submission_ids_with_evals():
  for problem_id in get_all_problem_ids_with_evals():
    for submission_id in get_all_submission_ids_with_evals(problem_id):
      yield problem_id, submission_id


def get_metadata_path(problem_id):
  return os.path.join(DATA_ROOT, 'metadata', f'{problem_id}.csv')


def get_python_path(problem_id, submission_id):
  return os.path.join(DATA_ROOT, 'data', problem_id, 'Python', f'{submission_id}.py')


def get_input_path(problem_id, submission_id):
  return os.path.join(DATA_ROOT, 'derived', 'input_output', 'data', problem_id, 'input.txt')


def get_output_path(problem_id, submission_id):
  return os.path.join(DATA_ROOT, 'derived', 'input_output', 'data', problem_id, 'input.txt')


def get_evals_dir(problem_id, submission_id):
  return os.path.join(EVALS_ROOT, problem_id, submission_id)


def get_evals_paths(problem_id, submission_id):
  evals_dir = get_evals_dir(problem_id, submission_id)
  error_path = os.path.join(evals_dir, 'error.txt')
  timeout_path = os.path.join(evals_dir, 'timeout.txt')
  stdout_path = os.path.join(evals_dir, 'stdout.txt')
  stderr_path = os.path.join(evals_dir, 'stderr.txt')
  return error_path, timeout_path, stdout_path, stderr_path


def get_problem_metadata(problem_id):
  metadata_path = get_metadata_path(problem_id)
  with open(metadata_path, 'r') as f:
    metadata_str = f.read()
  metadata_str_lines = metadata_str.split('\n')
  headers_str, body_lines = metadata_str_lines[0], metadata_str_lines[1:]
  assert headers_str == 'submission_id,problem_id,user_id,date,language,original_language,filename_ext,status,cpu_time,memory,code_size,accuracy'
  headers = headers_str.split(',')
  metadata = {}
  for line in body_lines:
    if not line:
      continue
    values = line.split(',')
    line_data = dict(zip(headers, values))
    submission_id = line_data['submission_id']
    if line_data['language'] == 'Python':
      metadata[submission_id] = line_data
  return metadata


def get_submission_metadata(problem_id, submission_id):
  metadata = get_problem_metadata(problem_id)
  return metadata.get(submission_id)


def read(path):
  if os.path.exists(path):
    with open(path, 'r') as f:
      return f.read()


def get_submission_error_kind(problem_id, submission_id):
  error_data, timeout_data, stdout_data, stderr_data = get_submission_eval_raw(
      problem_id, submission_id)
  if (error_data, timeout_data, stdout_data, stderr_data) == (None,) * 4:
    # The error-checker may not have been run for this submission yet.
    return error_kinds.NO_DATA
  if timeout_data:
    # The error-checker execution reached the timeout.
    return error_kinds.TIMEOUT
  if error_data:
    if stderr_data:
      for error_kind in error_kinds.ERROR_KINDS:
        if error_kind in stderr_data:
          # We've detected the error kind successfully.
          return error_kind
      # A new error kind occurred.
      other_error = stderr_data.strip().split('\n')[-1]
      return error_kinds.OTHER_ERROR
    else:
      # An error occurred but nothing is in stderr. This is unexpected.
      return error_kinds.SILENT_ERROR
  else:
    if stderr_data:
      # No error occurred, but the program used stderr anyway.
      return error_kinds.NO_ERROR_WITH_STDERR
    else:
      # No error occurred.
      return error_kinds.NO_ERROR


def get_submission_eval_raw(problem_id, submission_id):
  error_path, timeout_path, stdout_path, stderr_path = get_evals_paths(
      problem_id, submission_id)
  error_data = read(error_path)
  timeout_data = read(timeout_path)
  stdout_data = read(stdout_path)
  stderr_data = read(stderr_path)
  return error_data, timeout_data, stdout_data, stderr_data


def run_for_errors(problem_id, submission_id, skip_existing=True):
  """Runs the command in the error-checker subprocess."""
  evals_dir = get_evals_dir(problem_id, submission_id)
  if os.path.exists(evals_dir):
    if skip_existing:
      return
    shutil.rmtree(evals_dir)
  os.makedirs(evals_dir)
  python_filepath = get_python_path(problem_id, submission_id)
  input_filepath = get_input_path(problem_id, submission_id)

  if not os.path.exists(input_filepath):
    return

  error_path, timeout_path, stdout_path, stderr_path = get_evals_paths(
      problem_id, submission_id)
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
