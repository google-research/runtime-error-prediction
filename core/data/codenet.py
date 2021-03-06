# Copyright (C) 2021 Google Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import fire

import functools
import os
import re
import shutil
import subprocess
import tqdm

from core.data import codenet_paths
from core.data import error_kinds

DATA_ROOT = codenet_paths.DATA_ROOT
EVALS_ROOT = codenet_paths.EVALS_ROOT
PYTHON3 = codenet_paths.PYTHON3
ERROR_CHECKER = codenet_paths.ERROR_CHECKER
PIPE = subprocess.PIPE


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


def get_split_problem_and_submission_ids(problem_ids):
  for problem_id in problem_ids:
    for submission_id in get_all_submission_ids(problem_id):
      yield problem_id, submission_id


def get_all_problem_ids_with_evals():
  return os.listdir(EVALS_ROOT)


def get_all_submission_ids_with_evals(problem_id):
  evals_dir = os.path.join(EVALS_ROOT, problem_id)
  if os.path.exists(evals_dir):
    return [
        submission_id for submission_id in os.listdir(evals_dir)
        if os.listdir(os.path.join(evals_dir, submission_id))
    ]
  else:
    return []


def get_all_problem_and_submission_ids_with_evals():
  for problem_id in tqdm.tqdm(get_all_problem_ids_with_evals()):
    for submission_id in get_all_submission_ids_with_evals(problem_id):
      yield problem_id, submission_id


def get_split_problem_and_submission_ids_with_evals(problem_ids):
  for problem_id in tqdm.tqdm(problem_ids):
    for submission_id in get_all_submission_ids_with_evals(problem_id):
      yield problem_id, submission_id


def get_metadata_path(problem_id):
  return os.path.join(DATA_ROOT, 'metadata', f'{problem_id}.csv')


def get_python_path(problem_id, submission_id):
  return os.path.join(DATA_ROOT, 'data', problem_id, 'Python', f'{submission_id}.py')


def get_input_path(problem_id, submission_id):
  return os.path.join(DATA_ROOT, 'derived', 'input_output', 'data', problem_id, 'input.txt')


def get_output_path(problem_id, submission_id):
  return os.path.join(DATA_ROOT, 'derived', 'input_output', 'data', problem_id, 'output.txt')


def get_evals_dir(problem_id, submission_id):
  return os.path.join(EVALS_ROOT, problem_id, submission_id)


def get_evals_paths(problem_id, submission_id):
  evals_dir = get_evals_dir(problem_id, submission_id)
  error_path = os.path.join(evals_dir, 'error.txt')
  timeout_path = os.path.join(evals_dir, 'timeout.txt')
  stdout_path = os.path.join(evals_dir, 'stdout.txt')
  stderr_path = os.path.join(evals_dir, 'stderr.txt')
  return error_path, timeout_path, stdout_path, stderr_path


def get_error_lineno(problem_id, submission_id):
  error_data, timeout_data, stdout_data, stderr_data = get_submission_eval_raw(problem_id, submission_id)
  match = re.search(r'line (\d+), in main__errorchecker__', stderr_data)
  if match:
    # We subtract 1 from the reported line number because we injected 1 line
    # in error-checker.py
    return int(match.group(1)) - 1
  return 0  # Zero indicates no error.


@functools.lru_cache(maxsize=32)
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


def get_python_major_version(problem_id, submission_id):
  submission_metadata = get_submission_metadata(problem_id, submission_id)
  original_language = submission_metadata['original_language']
  if original_language in [
      'Python',
      'Python (2.7.3)',
      'Python (2.7.6)',
      'PyPy2 (5.6.0)',
      'PyPy2 (7.3.0)',
  ]:
    return 2
  elif original_language in [
      'Python3',
      'Python (3.4.2)',
      'Python (3.4.3)',
      'Python (3.8.2)',
      'PyPy3 (2.4.0)',
      'PyPy3 (7.3.0)',
  ]:
    return 3
  else:
    return None


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
