import collections

import fire
from tqdm import tqdm

from core.data import codenet


def get_eval_histogram(problem_id):
  error_counts = collections.defaultdict(int)
  for submission_id in codenet.get_all_submission_ids(problem_id):
    submission_eval = codenet.get_submission_eval(problem_id, submission_id)
    error_counts[submission_eval] += 1
  return error_counts


def get_full_eval_histogram():
  error_counts = collections.defaultdict(int)
  for problem_id, submission_id in codenet.get_all_problem_and_submission_ids():
    submission_eval = codenet.get_submission_eval(problem_id, submission_id)
    error_counts[submission_eval] += 1
  return error_counts


if __name__ == '__main__':
  fire.Fire()
