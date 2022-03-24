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

"""Runs a Python program using exec to check for output and errors."""

from apache_beam.io.gcp import gcsio

import fire


def run_for_errors(python_filepath, out_path):
  gcsio_client = gcsio.GcsIO()

  # Assumes the input is stdin when called.
  python_source = gcsio_client.open(python_filepath, 'r').read()
  python_source = python_source.replace('__name__ == "__main__"', 'True')
  python_source = python_source.replace("__name__ == '__main__'", 'True')
  python_source = (
      'def main__errorchecker__():\n'
      + '\n'.join('  ' + line for line in python_source.split('\n'))
      + '\n'
      + 'main__errorchecker__()\n'
  )
  compiled = compile(python_source, python_filepath, 'exec')
  try:
    exec(compiled, {}, {})
  except Exception as e:
    with gcsio_client.open(out_path, 'w') as f:
      # We can handle the exception systematically here.
      f.write(str(e) + '\n')
    raise


if __name__ == '__main__':
  fire.Fire()
