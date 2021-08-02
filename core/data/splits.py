import json
import random

import fire

from core.data import codenet
from core.data import codenet_paths


DEFAULT_SPLITS_PATH = codenet_paths.DEFAULT_SPLITS_PATH


def make_splits(valid=0.1, test=0.1):
  problem_ids = codenet.get_all_problem_ids()
  random.shuffle(problem_ids)
  num_problems = len(problem_ids)
  num_valid_problems = int(valid * num_problems)
  num_test_problems = int(test * num_problems)
  num_train_problems = num_problems - (num_valid_problems + num_test_problems)
  train_problem_ids = problem_ids[:num_train_problems]
  valid_problem_ids = problem_ids[num_train_problems:num_train_problems + num_valid_problems]
  test_problem_ids = problem_ids[num_train_problems + num_valid_problems:]
  return {
      'train': sorted(train_problem_ids),
      'valid': sorted(valid_problem_ids),
      'test': sorted(test_problem_ids),
  }


def save_splits(splits, path=DEFAULT_SPLITS_PATH):
  with open(path, 'w') as f:
    json.dump(splits, f, ensure_ascii=False, indent=2)


def load_splits(path=DEFAULT_SPLITS_PATH):
  with open(path, 'r') as f:
    return json.load(f)


def make_and_save_splits(valid=0.1, test=0.1, path=DEFAULT_SPLITS_PATH):
  splits = make_splits(valid=valid, test=test)
  save_splits(splits, path=path)


def get_all_problem_and_submission_ids(problem_ids):
  """Yields all problem and submission ids for the provided split."""
  for problem_id in problem_ids:
    for submission_id in codenet.get_all_submission_ids(problem_id):
      yield problem_id, submission_id


if __name__ == '__main__':
  fire.Fire()
