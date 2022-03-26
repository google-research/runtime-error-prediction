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

from core.distributed import gcp

call = gcp.call
wait = gcp.wait
parallel = gcp.parallel
_hostname = gcp._hostname
_zone = gcp._zone


repo_url = f'https://github.com/google-research/runtime-error-prediction.git'


def clone_args(index):
  branch = 'main'
  hostname = _hostname(index)
  zone = _zone(index)
  return ['gcloud', 'compute', 'ssh', hostname, '--command',
          f'sudo apt install git -y && git clone {repo_url} && cd runtime-error-prediction && git checkout {branch}',
          '--zone', zone]


def clone(index):
  args = clone_args(index)
  return call(args)


def setup_args(index):
  hostname = _hostname(index)
  zone = _zone(index)
  return ['gcloud', 'compute', 'ssh', hostname, '--command',
          f'runtime-error-prediction/core/distributed/setup.sh',
          '--zone', zone]


def setup(index):
  args = setup_args(index)
  return call(args)


n = 2
gcp.up_n(n)
gcp.fix_firewall().wait()

# wait(parallel(clone, n=n))
# # wait(parallel(setup, n=n))
# gcp.down_n(n)
