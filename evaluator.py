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

"""Runs codeforces submissions to check for runtime errors and correctness."""

import fire
import os
import pickle
import shutil
import subprocess

PIPE = subprocess.PIPE
PYTHON = '/Users/dbieber/.virtualenvs/_2/bin/python'
ERROR_CHECKER = '/Users/dbieber/code/playground/codeforces-data/error-checker.py'


def run_in_process_py2(python_filepath, input_filepath):
  import imp
  imp.load_source('user', python_filepath)


def run_in_process_py3(python_filepath, input_filepath):
  import importlib
  spec = importlib.util.spec_from_file_location('user', python_filepath)
  module = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(module)


def run(python_filepath, input_filepath):
  command = [PYTHON, python_filepath]
  return subprocess.run(
      command,
      input=open(input_filepath, 'rb').read(),
      stderr=subprocess.DEVNULL,
      stdout=PIPE,
  ).stdout.decode('utf-8')


def run_for_errors(python_filepath, input_filepath):
  print(input_filepath)
  command = [PYTHON, python_filepath]
  print(command)
  p = subprocess.run(
      command,
      input=open(input_filepath, 'rb').read(),
      stderr=PIPE,
      stdout=PIPE,
  )
  stdout = p.stderr.decode('utf-8')
  stderr = p.stderr.decode('utf-8')
  print('stderr: ', stderr)


def run_for_errors2(python_filepath, input_filepath):
  python_source = open(python_filepath, 'r').read()
  compiled = compile(python_source, python_filepath, 'exec')
  exec(compiled)


def run_for_errors3(python_filepath, input_filepath):
  """Runs the command in the error-checker subprocess."""
  solution_name = os.path.basename(python_filepath).split('.')[0]
  solutions_dir = os.path.dirname(python_filepath)
  task_dir = os.path.dirname(solutions_dir)
  task_name = os.path.basename(task_dir)
  sample_name = os.path.basename(input_filepath).split('.')[0]

  out_dir = os.path.join('out', task_name, solution_name, sample_name)
  if os.path.exists(out_dir):
    shutil.rmtree(out_dir)
  os.makedirs(out_dir)
  error_path = os.path.join(out_dir, 'error.txt')
  stdout_path = os.path.join(out_dir, 'stdout.txt')
  stderr_path = os.path.join(out_dir, 'stderr.txt')
  command = [PYTHON, ERROR_CHECKER, 'run_for_errors', python_filepath, error_path]
  p = subprocess.run(
      command,
      input=open(input_filepath, 'rb').read(),
      stderr=open(stderr_path, 'wb'),
      stdout=open(stdout_path, 'wb'),
  )
  stdout = open(stdout_path, 'r').read()
  stderr = open(stderr_path, 'r').read()
  return stdout


def check(python_filepath, input_filepath, output_filepath):
  result = run_for_errors3(python_filepath, input_filepath)
  with open(output_filepath, 'r') as f:
    target = f.read()
  return result.strip() == target.strip()


def check_solution(python_filepath):
  solutions_dir = os.path.dirname(python_filepath)
  task_dir = os.path.dirname(solutions_dir)
  samples_dir = os.path.join(task_dir, 'samples')
  i = 1
  all_ok = True
  score = 0
  while True:
    input_filepath = os.path.join(samples_dir, '{i}_input.txt'.format(i=i))
    output_filepath = os.path.join(samples_dir, '{i}_output.txt'.format(i=i))
    if not os.path.exists(input_filepath):
      break
    try:
      sample_ok = check(python_filepath, input_filepath, output_filepath)
    except:
      sample_ok = False
      raise
    all_ok &= sample_ok
    score += sample_ok
    i += 1
  return all_ok, score


def check_all_solutions(task_dir):
  solutions_dir = os.path.join(task_dir, 'solutions_python')
  for solution_filename in os.listdir(solutions_dir):
    python_filepath = os.path.join(solutions_dir, solution_filename)
    all_ok, score = check_solution(python_filepath)
    print('{solution_filename}: {score}'.format(solution_filename=solution_filename, score=score))


def check_all(codeforces_dir):
  for task_name in os.listdir(codeforces_dir):
    task_dir = os.path.join(codeforces_dir, task_name)
    print('Task: ' + task_dir)
    check_all_solutions(task_dir)


if __name__ == '__main__':
  fire.Fire()
