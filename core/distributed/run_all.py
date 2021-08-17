import getpass

from core.distributed import gcp

wait = gcp.wait
parallel = gcp.parallel
_hostname = gcp._hostname
_zone = gcp._zone


access_token = getpass.getpass('Personal access token: ')
repo_url = f'https://{access_token}@github.com/googleprivate/compressive-ipagnn.git'


def clone_args(index):
  branch = '2021-08-13-overfit'
  hostname = _hostname(index)
  zone = _zone(index)
  return ['gcloud', 'compute', 'ssh', hostname, '--command',
          f'git clone {repo_url} && cd compressive-ipagnn && git checkout {branch}',
          '--zone', zone]


def clone(index):
  args = clone_args(index)
  return call(args)


def setup_args(index):
  hostname = _hostname(index)
  zone = _zone(index)
  return ['gcloud', 'compute', 'ssh', hostname, '--command',
          f'compressive-ipagnn/core/distributed/setup.sh',
          '--zone', zone]


def setup(index):
  args = setup_args(index)
  return call(args)


n = 2
gcp.up_n(n)

wait(parallel(clone, n=n))
# wait(parallel(setup, n=n))
gcp.down_n(n)
