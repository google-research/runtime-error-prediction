import dataclasses

from typing import Any

from core.data import codenet_paths
from core.data import error_kinds

DEFAULT_DATASET_PATH = codenet_paths.DEFAULT_DATASET_PATH


@dataclasses.dataclass
class Info:
  vocab_size: int
  num_classes: int
  all_error_kinds: Any


def get_dataset_info(dataset_path, config):
  if config.binary_targets:
    num_classes = 2
    all_error_kinds = list(range(2))
  elif 'control_flow_programs_raise' in dataset_path:
    num_classes = 1001
    all_error_kinds = list(range(1001))
  elif 'control_flow_programs' in dataset_path:
    num_classes = 1000
    all_error_kinds = list(range(1000))
  else:
    num_classes = error_kinds.NUM_CLASSES
    all_error_kinds = error_kinds.ALL_ERROR_KINDS
  return Info(
      vocab_size=30000,
      num_classes=num_classes,
      all_error_kinds=all_error_kinds,
  )


def get_test_info():
  return Info(
      vocab_size=500,
      num_classes=error_kinds.NUM_CLASSES,
      all_error_kinds=error_kinds.ALL_ERROR_KINDS,
  )
