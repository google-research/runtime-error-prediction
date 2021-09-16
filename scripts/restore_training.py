import os

from core.data import codenet_paths
from core.distributed import gcp
from core.lib import metadata as metadata_lib

import fire


def restore_training(
    study_id, experiment_id, run_id,
    experiments_dir=None
  ):
  experiment_id = str(experiment_id)

  run_dir = codenet_paths.make_run_dir(study_id, experiment_id, run_id, experiments_dir=experiments_dir)
  checkpoint_path = codenet_paths.make_checkpoints_path(run_dir)
  metadata_path = codenet_paths.make_metadata_path(run_dir)
  metadata = metadata_lib.read_metadata(metadata_path)
  command = metadata['command']

  machine = metadata['hostname']
  print(machine, gcp.as_shell_string(command))



if __name__ == '__main__':
  fire.Fire()
