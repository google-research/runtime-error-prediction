"""Issue commands to multiple GCE VMs in parallel."""

import pipes
import subprocess

from absl import logging
import fire
import termcolor


WORKER_PREFIX = 'worker'
DEFAULT_PROJECT = 'runtime-error-problems'


def call(args, stdin=None):
  """Uses subprocess to call the command given by the args."""
  shell_str = as_shell_string(args)
  logging.info(shell_str)
  print(termcolor.colored('RUNNING: ', 'green') + shell_str)
  return subprocess.Popen(args, stdin=stdin)


def parallel(f, n, offset=0):
  processes = []
  for i in range(offset, n + offset):
    p = f(i)
    processes.append(p)
  return processes


def wait(processes):
  for p in processes:
    p.wait()


def _hostname(index):
  return f'{WORKER_PREFIX}-{index:03d}'


def _zone(index):
  """Chooses a GCP zone based on the index."""
  if index < 6:
    return 'us-central1-a'
  elif index < 12:
    return 'us-east1-b'
  elif index < 18:
    return 'us-east4-c'
  elif index < 24:
    return 'us-west2-a'
  else:
    raise ValueError('Unhandled zone index')


def _tpu_hostname(index):
  return f'tpu-host-{index:03d}'


def _tpu_zone(index):
  """Chooses a GCP TPU zone based on the index."""
  if index < 70:
    return 'europe-west4-a'
  elif index < 80:  # 10
    return 'us-central1-b'
  elif index < 90:  # 10
    return 'us-central1-c'
  elif index < 168:  # 30
    return 'asia-east1-c'
  else:
    raise ValueError('Unhandled zone index')


def as_shell_string(args):
  """Turns the args representing a command into a string for running in a shell.

  The string returned can be copy/pasted and run from a shell. This uses
  pipes.quote to escape the arguments for shell use. Arguments with an equal
  sign, such as flag args, have the values on each side of the equal sign
  escaped separately in order to improve readability.

  Args:
    args: A list of the args used for running the process.
  Returns:
    A string for running the command from a shell.
  """
  quoted_args = []
  for arg in args:
    if '=' in arg:
      flag, value = arg.split('=', 1)
      quoted_args.append(pipes.quote(flag) + '=' + pipes.quote(value))
    else:
      quoted_args.append(pipes.quote(arg))
  return ' '.join(quoted_args)


def up_args(
    index,
    project=DEFAULT_PROJECT,
    machine_type='c2-standard-4',
):
  """Starts a single worker."""
  hostname = _hostname(index)
  zone = _zone(index)
  return f"""
gcloud beta compute --project={project} instances create {hostname} \
--zone={zone} \
--machine-type={machine_type} \
--subnet=default \
--network-tier=PREMIUM --maintenance-policy=MIGRATE \
--scopes=\
https://www.googleapis.com/auth/devstorage.read_write,\
https://www.googleapis.com/auth/logging.write,\
https://www.googleapis.com/auth/monitoring.write,\
https://www.googleapis.com/auth/servicecontrol,\
https://www.googleapis.com/auth/service.management.readonly,\
https://www.googleapis.com/auth/trace.append \
--image=debian-9-drawfork-v20200207 \
--image-project=eip-images \
--boot-disk-size=10GB \
--boot-disk-type=pd-standard \
--boot-disk-device-name={hostname} \
--reservation-affinity=any
""".split()


def up(index):
  args = up_args(index)
  return call(args)


def up_n(n, offset=0):
  return wait(parallel(up, n=n, offset=offset))


def create_instances(n):
  # Ensure N cpu VMs are started and set up.
  up_n(n)
  fix_firewall().wait()


def down_args(index):
  hostname = _hostname(index)
  zone = _zone(index)
  return (
      f'gcloud beta compute instances delete {hostname} --zone={zone} --quiet'.split()
  )


def down(index):
  args = down_args(index)
  return call(args)


def down_n(n, offset=0):
  return wait(parallel(down, n=n, offset=offset))


def fix_firewall_args():
  return (
      'gcloud compute firewall-rules create default-allow-ssh --allow tcp:22'
      .split())


def fix_firewall():
  args = fix_firewall_args()
  return call(args)


def tpu_up_args(
    index,
    project=DEFAULT_PROJECT,
):
  hostname = _tpu_hostname(index)
  zone = _tpu_zone(index)
  return f"""
gcloud alpha compute tpus tpu-vm create {hostname} \
--project={project} \
--zone={zone} \
--version=v2-alpha \
--accelerator-type=v2-8
  """.split()


def tpu_up(index):
  args = tpu_up_args(index)
  return call(args)


def tpu_down_args(
    index,
    project=DEFAULT_PROJECT,
):
  hostname = _tpu_hostname(index)
  zone = _tpu_zone(index)
  return f"""
gcloud alpha compute tpus tpu-vm delete {hostname} \
--project={project} \
--zone={zone} --quiet
  """.split()


def tpu_down(index):
  args = tpu_down_args(index)
  return call(args)


def tpu_up_n(n, offset=0):
  return wait(parallel(tpu_up, n=n, offset=offset))


def tpu_down_n(n, offset=0):
  return wait(parallel(tpu_down, n=n, offset=offset))


def _do_single_run(index, run_command_fn):
  command = run_command_fn(index)
  hostname = _hostname(index)
  zone = _zone(index)
  args = ['gcloud', 'compute', 'ssh', hostname, '--command', command,
          '--zone', zone]
  return call(args)


def _tpu_do_single_run(index, run_command_fn):
  command = run_command_fn(index)
  hostname = _tpu_hostname(index)
  zone = _tpu_zone(index)
  args = ['gcloud', 'alpha', 'compute', 'tpus', 'tpu-vm', 'ssh',
          hostname, '--command', command,
          '--zone', zone]
  return call(args)


def list_instances_args():
  return 'gcloud compute instances list'.split()


def list_instances():
  args = list_instances_args()
  return call(args)


def run_command(command, n, offset=0):
  calls = []
  for index in range(offset, n + offset):
    hostname = _hostname(index)
    zone = _zone(index)
    worker_command = f'echo -n "{hostname} " && {command}'
    calls.append(call(['gcloud', 'compute', 'ssh', hostname, '--command',
                       worker_command, '--zone', zone]))
  wait(calls)


def tpu_run_command(command, n, offset=0):
  calls = []
  for index in range(offset, n + offset):
    hostname = _tpu_hostname(index)
    zone = _tpu_zone(index)
    worker_command = f'echo -n {hostname} && {command}'
    calls.append(call([
        'gcloud', 'alpha', 'compute', 'tpus', 'tpu-vm', 'ssh',
        hostname, '--command', worker_command,
        '--zone', zone]))
  wait(calls)


def tpu_run_commands(run_command_fn, n, offset=0):
  calls = []
  for index in range(offset, n + offset):
    calls.append(_tpu_do_single_run(index, run_command_fn))
  wait(calls)


def tpu_run_script(filepath, n, environment, offset=0):
  calls = []
  for index in range(offset, n + offset):
    hostname = _tpu_hostname(index)
    zone = _tpu_zone(index)
    environment_tokens = [
        f'{key}={value}'
        for key, value in environment.items()
    ]
    command = [
        'gcloud', 'alpha', 'compute', 'tpus', 'tpu-vm', 'ssh',
        hostname, '--zone', zone, '--',
    ] + environment_tokens + [
        'bash', '-s'
    ]
    calls.append(call(command, stdin=open(filepath)))
  wait(calls)


if __name__ == '__main__':
  fire.Fire()
