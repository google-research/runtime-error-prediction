import collections

import fire

from core.data import codenet


def get_eval_histogram(problem_id):
  error_counts = collections.defaultdict(int)
  for submission_id in codenet.get_all_submission_ids_with_evals(problem_id):
    error_kind = codenet.get_submission_error_kind(problem_id, submission_id)
    error_counts[error_kind] += 1
  return error_counts


def get_full_eval_histogram():
  error_counts = collections.defaultdict(int)
  last_problem_id = None
  last_total = 0
  total = 0
  for problem_id, submission_id in codenet.get_all_problem_and_submission_ids_with_evals():
    if problem_id != last_problem_id and last_total != total:
      last_problem_id = problem_id
      last_total = total
      print(dict(error_counts))
      print()
    error_kind = codenet.get_submission_error_kind(problem_id, submission_id)
    error_counts[error_kind] += 1
    total += 1
  return error_counts


if __name__ == '__main__':
  fire.Fire()
