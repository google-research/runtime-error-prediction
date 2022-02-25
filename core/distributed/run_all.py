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
