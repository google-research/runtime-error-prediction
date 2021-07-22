"""Parsing, tokenizing, and generating datasets from the CodeNet data."""

import itertools

from core.data import codenet
from core.data import data_io
from core.data import error_kinds
from core.data import process
from core.data import tokenization

import tensorflow as tf

import fire

DEFAULT_TOKENIZER_PATH = 'out/tokenizers/full.json'
DEFAULT_DATASET_PATH = 'out/data/default.tfrecord'


def generate_tokenizer(path=DEFAULT_TOKENIZER_PATH, max_files=None):
  """Generates a tokenizer for the CodeNet data.

  Args:
    path: The location to write the tokenizer data to.
    max_files: (optional) The maximum number of submissions to use for
      generating the tokenizer.
  Returns:
    The generated Tokenizer.
  """
  files = []  
  for problem_id, submission_id in codenet.get_all_problem_and_submission_ids():
    python_path = codenet.get_python_path(problem_id, submission_id)
    files.append(python_path)
    if max_files and len(files) >= max_files:
      break
  return tokenization.generate_tokenizer(path=path, files=files)


def generate_codenet_dataset(
    tokenizer_path=DEFAULT_TOKENIZER_PATH,
    dataset_path=DEFAULT_DATASET_PATH,
    max_files=None):
  """Generates a TFRecord dataset from the CodeNet data.

  Args:
    tokenizer_path: The tokenizer data to use when generating the dataset.
    dataset_path: The path to write the dataset to.
  """
  with tf.io.TFRecordWriter(dataset_path) as file_writer:
    for problem in itertools.islice(process_codenet(tokenizer_path=tokenizer_path), max_files):
      record_bytes = data_io.to_tf_example(problem).SerializeToString()
      file_writer.write(record_bytes)


def process_codenet(tokenizer_path=DEFAULT_TOKENIZER_PATH, start_at=0):
  """Makes RuntimeErrorProblem objects per submission using the tokenizer."""
  tokenizer = tokenization.load_tokenizer(path=tokenizer_path)

  count = 0
  for problem_id, submission_id in codenet.get_all_problem_and_submission_ids_with_evals():
    count += 1
    if count < start_at:
      continue

    python_path = codenet.get_python_path(problem_id, submission_id)
    with open(python_path, 'r') as f:
      source = f.read()
      error_kind = codenet.get_submission_error_kind(problem_id, submission_id)
      target = error_kinds.to_index(error_kind)

    try:
      problem = process.make_runtimeerrorproblem(source, target, tokenizer=tokenizer)
      yield problem
    except SyntaxError:
      print(f'SyntaxError: {python_path}')
    except IndexError:
      print(f'IndexError: {python_path}')
      raise
    except RuntimeError:
      # Could be "return occurs outside of a function frame".
      print(f'RuntimeError: {python_path}')
    except AttributeError:
      print(f'AttributeError: {python_path}')
      raise
    except AssertionError:
      print(f'AssertionError: {python_path}')
    except:
      print(f'Unexpected error: {python_path}')
      raise

    if count % 1000 == 0:
      print(count)


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
