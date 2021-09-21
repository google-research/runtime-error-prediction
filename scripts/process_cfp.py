import glob
import os
import random

import fire
import tensorflow as tf

from core.data import data_io
from core.data import cfp_data_io
from core.data import codenet_paths
from core.data import process
from core.data import tokenization

DEFAULT_CFP_DATASET_PATH = codenet_paths.DEFAULT_CFP_DATASET_PATH
DEFAULT_CFP_DATA_PATTERN = codenet_paths.DEFAULT_CFP_DATA_PATTERN
DEFAULT_TOKENIZER_PATH = codenet_paths.DEFAULT_TOKENIZER_PATH


def generate_codenet_dataset(
    tfrecord_pattern=DEFAULT_CFP_DATA_PATTERN,
    tokenizer_path=DEFAULT_TOKENIZER_PATH,
    dataset_path=DEFAULT_CFP_DATASET_PATH,
    fraction=1.0):
  """Generates a TFRecord dataset from the control flow programs data.

  Args:
    tokenizer_path: The tokenizer data to use when generating the dataset.
    dataset_path: The path to write the dataset to.
  """
  tfrecord_paths = glob.glob(tfrecord_pattern)
  for tfrecord_path in tfrecord_paths:
    problems_gen = process_control_flow_programs(
        tfrecord_path=tfrecord_path,
        tokenizer_path=tokenizer_path,
        fraction=fraction)
    basename = os.path.basename(tfrecord_path)

    dataset_tfrecord_path = os.path.join(dataset_path, basename)
    with tf.io.TFRecordWriter(dataset_tfrecord_path) as file_writer:
      for problem in problems_gen:
        record_bytes = data_io.to_tf_example(problem).SerializeToString()
        file_writer.write(record_bytes)


def process_control_flow_programs(
    tfrecord_path,
    tokenizer_path=DEFAULT_TOKENIZER_PATH,
    fraction=1.0,
    start_at=0):
  """Makes RuntimeErrorProblem objects per program using the tokenizer."""
  tokenizer = tokenization.load_tokenizer(path=tokenizer_path)

  basename = os.path.basename(tfrecord_path)
  tfrecord_paths = [tfrecord_path]
  dataset = cfp_data_io.load_dataset(tfrecord_paths, include_strings=True)

  count = 0
  for index, example in enumerate(dataset):
    if random.random() > fraction:
      # Only use a random `fraction` of the submissions.
      continue
    count += 1
    if count < start_at:
      continue

    source = example['human_readable_code'][0].numpy().decode('utf-8')
    target = example['target_output'][0].numpy()
    original_step_limit = example['cfg_forward/steps'][0].numpy()

    problem = process.make_runtimeerrorproblem(
        source, target, tokenizer=tokenizer,
        problem_id=basename, submission_id=str(index))
    assert problem.step_limit == original_step_limit
    yield problem

    if count % 1000 == 0:
      print(count)


if __name__ == '__main__':
  fire.Fire()
