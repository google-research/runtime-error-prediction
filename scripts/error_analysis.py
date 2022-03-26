# Copyright (C) 2021 Google Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

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
