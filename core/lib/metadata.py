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
    json.loads(f.read())
