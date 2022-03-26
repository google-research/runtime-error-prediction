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

"""Functions useful for exploring or inspecting the data."""

import fire

from core.data import codenet
from core.data import codenet_paths
from core.data import error_kinds
from core.data import process
from core.data import tokenization


DEFAULT_TOKENIZER_PATH = codenet_paths.DEFAULT_TOKENIZER_PATH


def get_source_and_target_for_submission(problem_id, submission_id):
  """Returns the source and target for the specified submission."""
  python_path = codenet.get_python_path(problem_id, submission_id)
  with open(python_path, 'r') as f:
    source = f.read()
    error_kind = codenet.get_submission_error_kind(problem_id, submission_id)
    if error_kind == error_kinds.NO_DATA:
      raise RuntimeError('No data available for python_path', python_path)
    target = error_kinds.to_index(error_kind)
  return source, target


def make_runtimeerrorproblem_for_submission(problem_id, submission_id, tokenizer=None):
  """Constructs a RuntimeErrorProblem from the provided problem_id and submission_id."""
  source, target = get_source_and_target_for_submission(problem_id, submission_id)
  target_lineno = codenet.get_error_lineno(problem_id, submission_id)
  return process.make_runtimeerrorproblem(
      source, target, target_lineno=target_lineno,
      tokenizer=tokenizer, problem_id=problem_id, submission_id=submission_id)


def make_rawruntimeerrorproblem_for_submission(problem_id, submission_id):
  """Constructs a RawRuntimeErrorProblem from the provided problem_id and submission_id."""
  source, target = get_source_and_target_for_submission(problem_id, submission_id)
  target_lineno = codenet.get_error_lineno(problem_id, submission_id)
  return process.make_rawruntimeerrorproblem(
      source, target, target_lineno=target_lineno,
      problem_id=problem_id, submission_id=submission_id)


def make_rawruntimeerrorproblem_for_file(filepath):
  """Constructs a RawRuntimeErrorProblem from the given filepath."""
  with open(filepath, 'r') as f:
    source = f.read()
  target = 0
  target_lineno = 0
  return process.make_rawruntimeerrorproblem(
      source, target, target_lineno=target_lineno)


def get_spans(problem_id, submission_id, tokenizer_path=DEFAULT_TOKENIZER_PATH):
  tokenizer = tokenization.load_tokenizer(path=tokenizer_path)
  source, target = get_source_and_target_for_submission(problem_id, submission_id)
  target_lineno = codenet.get_error_lineno(problem_id, submission_id)

  problem = process.make_runtimeerrorproblem(
      source, target, target_lineno=target_lineno,
      tokenizer=tokenizer, problem_id=problem_id, submission_id=submission_id)

  print(source)
  tokens = tokenizer.convert_ids_to_tokens(problem.tokens)
  for span_index, (span_start, span_end) in enumerate(zip(problem.node_token_span_starts, problem.node_token_span_ends)):
    print(f'Span {span_index}: {tokens[span_start:span_end + 1]}')


if __name__ == '__main__':
  fire.Fire()
