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

from datetime import datetime
import json
import subprocess
import sys

from core.data import codenet_paths


def get_commit():
  try:
    return subprocess.check_output([
        'git', 'rev-parse', 'HEAD'
    ]).decode('utf-8').strip()
  except subprocess.CalledProcessError:
    return None


def get_branch():
  try:
    return subprocess.check_output([
        'git', 'rev-parse', '--abbrev-ref', 'HEAD'
    ]).decode('utf-8').strip()
  except subprocess.CalledProcessError:
    return None


def get_whoami():
  try:
    return subprocess.check_output(['whoami']).decode('utf-8').strip()
  except subprocess.CalledProcessError:
    return None


def get_datetime():
  return datetime.now().strftime('%Y-%m-%d %H:%M:%S')


def get_metadata():
  return {
      'command': sys.argv,
      'hostname': codenet_paths.HOSTNAME,
      'short_hostname': codenet_paths.SHORT_HOSTNAME,
      'commit': get_commit(),
      'branch': get_branch(),
      'whoami': get_whoami(),
      'datetime': get_datetime(),
  }


def write_metadata(path):
  metadata = get_metadata()
  with open(path, 'w') as f:
    f.write(json.dumps(metadata))


def read_metadata(path):
  with open(path, 'r') as f:
    return json.loads(f.read())
