"""Parsing, tokenizing, and generating datasets from the CodeNet data."""

import collections
import functools
import itertools
import json
import os
import random

import fire
from python_graphs import control_flow
import tensorflow as tf

from core.data import codenet
from core.data import codenet_paths
from core.data import data_io
from core.data import descriptions
from core.data import error_kinds
from core.data import process
from core.data import splits
from core.data import tokenization


DEFAULT_DATASET_PATH = codenet_paths.DEFAULT_DATASET_PATH
DEFAULT_SPLITS_PATH = codenet_paths.DEFAULT_SPLITS_PATH
DEFAULT_TOKENIZER_PATH = codenet_paths.DEFAULT_TOKENIZER_PATH


def generate_docstrings():
  for index in range(4053):
    problem_id = f'p{index:05d}'
    problem_description_path = codenet_paths.get_problem_description_path(problem_id)

    # Load the description.
    if not os.path.exists(problem_description_path):
      continue
    with open(problem_description_path, 'r') as f:
      text = f.read()
    try:
      info = descriptions.extract_input_information(text)
    except:
      print(problem_id)

    docstring_path = codenet_paths.get_problem_docstring_path(problem_id)
    with open(docstring_path, 'w') as f:
      f.write(info)


def generate_tokenizer(
    path=DEFAULT_TOKENIZER_PATH,
    splits_path=DEFAULT_SPLITS_PATH,
    require_evals=True,
    include_docstrings=True,
    max_files=None):
  """Generates a tokenizer for the CodeNet data using only the train split.

  Args:
    path: The location to write the tokenizer data to.
    splits_path: The path to the split data. Only train problems will be used.
    require_evals: If True, only uses submissions for which the evals are available.
    include_docstrings: If True, includes the files with the synthetic problem docstrings
      when generating the vocab.
    max_files: (optional) The maximum number of submissions to use for
      generating the tokenizer.
  Returns:
    The generated Tokenizer.
  """
  if splits_path:
    splits_dict = splits.load_splits(path=splits_path)
    train_problem_ids = splits_dict['train']
    if require_evals:
      problem_and_submission_ids = codenet.get_split_problem_and_submission_ids_with_evals(
          train_problem_ids)
    else:
      problem_and_submission_ids = codenet.get_split_problem_and_submission_ids(
          train_problem_ids)
  else:
    if require_evals:
      problem_and_submission_ids = codenet.get_all_problem_and_submission_ids_with_evals()
    else:
      problem_and_submission_ids = codenet.get_all_problem_and_submission_ids()

  files = []
  for problem_id, submission_id in problem_and_submission_ids:
    python_path = codenet.get_python_path(problem_id, submission_id)
    files.append(python_path)
  random.shuffle(files)
  if max_files:
    files = files[:max_files]

  if include_docstrings:
    if splits_path:
      problem_ids = train_problem_ids
    else:
      problem_ids = codenet.get_all_problem_ids()
    for problem_id in problem_ids:
      docstring_path = codenet_paths.get_problem_docstring_path(problem_id)
      if os.path.exists(docstring_path):
        files.append(docstring_path)

  return tokenization.generate_tokenizer(path=path, files=files)


def keep_fraction(l, f):
  """Return fraction `f` of list `l`. Shuffles `l`."""
  random.shuffle(l)
  count = int(f * len(l))
  return l[:count]


def generate_codenet_dataset(
    tokenizer_path=DEFAULT_TOKENIZER_PATH,
    dataset_path=DEFAULT_DATASET_PATH,
    splits_path=DEFAULT_SPLITS_PATH,
    include_docstrings=True,
    fraction=1.0,
    max_files=None):
  """Generates a TFRecord dataset from the CodeNet data.

  Args:
    tokenizer_path: The tokenizer data to use when generating the dataset.
    dataset_path: The path to write the dataset to.
    splits_path: The path to the split data.
    include_docstrings: If True, adds a synthetic docstring at the start of
      each submission, generated from the problem statement.
    fraction: The fraction of submissions to include in the dataset.
    max_files: (optional) The maximum number of submissions to use for
      generating the tokenizer.
  """
  random.seed(0)
  splits_dict = splits.load_splits(path=splits_path)

  train_path = codenet_paths.make_tfrecord_path(dataset_path, 'train')
  valid_path = codenet_paths.make_tfrecord_path(dataset_path, 'valid')
  test_path = codenet_paths.make_tfrecord_path(dataset_path, 'test')

  train_problems_gen = process_codenet(
      tokenizer_path=tokenizer_path, problem_ids=splits_dict['train'],
      include_docstrings=include_docstrings, fraction=fraction)
  save_codenet_tfrecord(train_path, train_problems_gen, max_files=max_files)
  valid_problems_gen = process_codenet(
      tokenizer_path=tokenizer_path, problem_ids=splits_dict['valid'],
      include_docstrings=include_docstrings, fraction=fraction)
  save_codenet_tfrecord(valid_path, valid_problems_gen, max_files=max_files)
  test_problems_gen = process_codenet(
      tokenizer_path=tokenizer_path, problem_ids=splits_dict['test'],
      include_docstrings=include_docstrings, fraction=fraction)
  save_codenet_tfrecord(test_path, test_problems_gen, max_files=max_files)


def save_codenet_tfrecord(tfrecord_path, problems_gen, max_files=None):
  ids = []
  with tf.io.TFRecordWriter(tfrecord_path) as file_writer:
    for problem in itertools.islice(problems_gen, max_files):
      ids.append((problem.problem_id, problem.submission_id))
      record_bytes = data_io.to_tf_example(problem).SerializeToString()
      file_writer.write(record_bytes)

  ids_path = codenet_paths.make_ids_path(tfrecord_path)
  with open(ids_path, 'w') as f:
    json.dump(ids, f, ensure_ascii=False, indent=2)


@functools.lru_cache()
def get_problem_docstring(problem_id):
  docstring_path = codenet_paths.get_problem_docstring_path(problem_id)
  if os.path.exists(docstring_path):
    with open(docstring_path, 'r') as f:
      return f.read()


def process_codenet(
    tokenizer_path=DEFAULT_TOKENIZER_PATH,
    problem_ids=None,
    include_docstrings=True,
    fraction=1.0,
    class_subsample_values=None,
    start_at=0):
  """Makes RuntimeErrorProblem objects per submission using the tokenizer."""
  tokenizer = tokenization.load_tokenizer(path=tokenizer_path)

  if class_subsample_values == 'default':
    class_subsample_values = {1: 0.0660801055}

  if problem_ids:
    problem_and_submission_ids = codenet.get_split_problem_and_submission_ids_with_evals(
        problem_ids)
  else:
    print('Using all problem_ids')
    problem_and_submission_ids = codenet.get_all_problem_and_submission_ids_with_evals()

  count = 0
  yielded = 0
  runtime_error_count = 0
  sampled_out_count = 0
  udf_count = 0
  syntax_error_count = 0
  py2_skip = 0
  assertion_error_count = 0
  unexpected_error_count = 0
  for problem_id, submission_id in problem_and_submission_ids:
    if random.random() > fraction:
      # Only use a random `fraction` of the submissions.
      continue
    count += 1
    if count < start_at:
      continue
    if count % 100000 == 0 or count == 5000:
      print(count)

    python_major_version = codenet.get_python_major_version(
        problem_id, submission_id)
    if python_major_version != 3:
      py2_skip += 1
      continue

    python_path = codenet.get_python_path(problem_id, submission_id)
    with open(python_path, 'r') as f:
      source = f.read()

    docstring = get_problem_docstring(problem_id)
    if docstring:
      extended_source = f'''"""{docstring}
"""
{source}'''
    else:
      extended_source = source

    if include_docstrings:
      source = extended_source

    error_kind = codenet.get_submission_error_kind(problem_id, submission_id)
    if error_kind == error_kinds.NO_DATA:
      raise RuntimeError('No data available for python_path', python_path)
    target = error_kinds.to_index(error_kind)
    target_lineno = codenet.get_error_lineno(problem_id, submission_id)
    # target_lineno doesn't account for the docstring lines, so we add those in:
    if include_docstrings and docstring:
      if target_lineno:  # 0 indicates no error, and should remain 0.
        target_lineno += len(docstring.split('\n')) + 1

    if class_subsample_values:
      if target in class_subsample_values:
        subsample_value = class_subsample_values[target]
        if random.random() > subsample_value:
          sampled_out_count += 1
          continue

    try:
      problem = process.make_runtimeerrorproblem(
          source, target,
          target_lineno=target_lineno,
          docstring=docstring,
          extended_source=extended_source,
          tokenizer=tokenizer,
          problem_id=problem_id,
          submission_id=submission_id)
      yielded += 1
      yield problem
    except ValueError as e:
      if str(e) == 'UDF not currently supported.':
        udf_count += 1
        continue
      print(f'ValueError: {python_path}')
      raise
    except SyntaxError:
      # print(f'SyntaxError: {python_path}')
      syntax_error_count += 1
      if syntax_error_count % 1000 == 0:
        print(f'Syntax Error Count: {syntax_error_count}')
    except IndexError:
      print(f'IndexError: {python_path}')
      raise
    except RuntimeError as e:
      if str(e).startswith('maximum recursion depth exceeded while calling a Python object'):  # e.g. p03107/Python/s405509758.py
        runtime_error_count += 1
        continue
      if str(e) == 'return occurs outside of a function frame.':
        runtime_error_count += 1
        continue
      if str(e) == 'break occurs outside of a loop frame.':
        runtime_error_count += 1
        continue
      if str(e) == 'continue occurs outside of a loop frame.':
        runtime_error_count += 1
        continue
      print(f'RuntimeError: {python_path} - {e}')
      raise
    except AttributeError as e:
      print(f'AttributeError: {python_path} - {e}')
      raise
    except AssertionError as e:
      print(f'AssertionError: {python_path} - {e}')
      assertion_error_count += 1
    except:
      print(f'Unexpected error: {python_path}')
      unexpected_error_count += 1
      # raise

    # if runtime_error_count % 1000 == 5:
    #   print(f'Runtime Error Count: {runtime_error_count}')
    # if udf_count % 1000 == 5:
    #   print(f'udf_count: {udf_count}')

  print(f'Final Syntax Error Count: {syntax_error_count}')
  print(f'Final UDF Count: {udf_count}')
  print(f'Final Runtime Error Count: {runtime_error_count}')
  print(f'Final Sampled-out Count: {sampled_out_count}')
  print(f'Final Count: {count}')
  print(f'Final PY2 count: {py2_skip}')
  print(f'Final assertion_error_count: {assertion_error_count}')
  print(f'Final unexpected_error_count: {unexpected_error_count}')
  print(f'Yielded: {yielded}')

def investigate_udf_usage(problem_ids=None, start_at=0):
  if problem_ids:
    problem_and_submission_ids = codenet.get_split_problem_and_submission_ids_with_evals(
        problem_ids)
  else:
    print('Using all problem_ids')
    problem_and_submission_ids = codenet.get_all_problem_and_submission_ids_with_evals()

  count = 0
  udf_usages = collections.defaultdict(int)
  for problem_id, submission_id in problem_and_submission_ids:
    count += 1
    if count < start_at:
      continue
    if count % 50000 == 0:
      print(count)
      print(dict(udf_usages))

    python_path = codenet.get_python_path(problem_id, submission_id)
    with open(python_path, 'r') as f:
      source = f.read()
      error_kind = codenet.get_submission_error_kind(problem_id, submission_id)
      target = error_kinds.to_index(error_kind)
      if target == 0:
        print(python_path)
        raise RuntimeError()

    try:
      graph = control_flow.get_control_flow_graph(source)
      udf_usage = process.examine_udfs(graph, problem_id, submission_id)
      udf_usages[udf_usage] += 1
    except ValueError as e:
      print(f'ValueError: {python_path} - {e}')
    except SyntaxError:
      print(f'SyntaxError: {python_path}')
    except IndexError:
      print(f'IndexError: {python_path}')
    except RuntimeError:
      # Could be "return occurs outside of a function frame".
      print(f'RuntimeError: {python_path}')
    except AttributeError:
      print(f'AttributeError: {python_path}')
    except AssertionError:
      print(f'AssertionError: {python_path}')
    except:
      print(f'Unexpected error: {python_path}')
  print(dict(udf_usages))
  return dict(udf_usages)


def run_codenet_submissions(max_files=None):
  """Runs all CodeNet Python submissions, recording output and errors."""
  last_problem_id = None
  for problem_id, submission_id in codenet.get_all_problem_and_submission_ids():
    if problem_id != last_problem_id:
      print(problem_id)
      last_problem_id = problem_id
    codenet.run_for_errors(problem_id, submission_id)


if __name__ == '__main__':
  fire.Fire()
