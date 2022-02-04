import dataclasses

from typing import Any, Dict, List, Optional

from core.data import codenet_paths
from core.data import error_kinds

DEFAULT_DATASET_PATH = codenet_paths.DEFAULT_DATASET_PATH


@dataclasses.dataclass
class Info:
  vocab_size: int
  num_classes: int
  all_error_kinds: Any
  no_error_ids: List[int]
  error_ids: List[int]
  class_subsample_values: Optional[Dict]


def get_dataset_info(dataset_path, config):
  if config.binary_targets:
    num_classes = 2
    all_error_kinds = list(range(2))
    no_error_ids = [0]
    error_ids = [1]
    class_subsample_values = None
  elif 'control_flow_programs_raise' in dataset_path:
    num_classes = 1001
    all_error_kinds = list(range(1001))
    no_error_ids = list(range(1000))
    error_ids = [1000]
    class_subsample_values = None
  elif 'control_flow_programs' in dataset_path:
    num_classes = 1000
    all_error_kinds = list(range(1000))
    no_error_ids = all_error_kinds
    error_ids = []
    class_subsample_values = None
  elif 'errors-only' in dataset_path:
    num_classes = 3
    all_error_kinds = list(range(3))
    no_error_ids = [0]
    error_ids = [1, 2]
    class_subsample_values = None
  elif 'errors-L2E' in dataset_path:
    num_classes = 1002
    all_error_kinds = list(range(1002))
    no_error_ids = list(range(1000))
    error_ids = [1000, 1001]
    class_subsample_values = None
  else:  # Runtime Error Prediction
    num_classes = error_kinds.NUM_CLASSES
    all_error_kinds = error_kinds.ALL_ERROR_KINDS
    no_error_ids = [error_kinds.NO_ERROR_ID,]
    error_ids = list(set(range(num_classes)) - set(no_error_ids))
    class_subsample_values = {1: 0.0660801055}
  return Info(
      vocab_size=30000,
      num_classes=num_classes,
      all_error_kinds=all_error_kinds,
      no_error_ids=no_error_ids,
      error_ids=error_ids,
      class_subsample_values=class_subsample_values,
  )


def get_test_info():
  return Info(
      vocab_size=500,
      num_classes=error_kinds.NUM_CLASSES,
      all_error_kinds=error_kinds.ALL_ERROR_KINDS,
      no_error_ids=[error_kinds.NO_ERROR_ID,],
      error_ids=list(set(range(error_kinds.NUM_CLASSES)) - {error_kinds.NO_ERROR_ID}),
      class_subsample_values={1: 0.0660801055},
  )
