import glob
import os
import random

import fire
import tensorflow as tf

from core.data import data_io
from core.data import cfp_raise_data_io
from core.data import codenet_paths
from core.data import process
from core.data import tokenization

RAW_CFP_RAISE_DATA_PATH = codenet_paths.RAW_CFP_RAISE_DATA_PATH
DEFAULT_CFP_RAISE_DATASET_PATH = codenet_paths.DEFAULT_CFP_RAISE_DATASET_PATH
DEFAULT_TOKENIZER_PATH = codenet_paths.DEFAULT_TOKENIZER_PATH


def generate_dataset(
    tfrecord_pattern=RAW_CFP_RAISE_DATA_PATH,
    tokenizer_path=DEFAULT_TOKENIZER_PATH,
    dataset_path=DEFAULT_CFP_RAISE_DATASET_PATH,
    fraction=1.0):
  """Generates a TFRecord dataset from the control flow programs data.

  Args:
    tokenizer_path: The tokenizer data to use when generating the dataset.
    dataset_path: The path to write the dataset to.
  """
  random.seed(0)

  tfrecord_paths = glob.glob(tfrecord_pattern)
  for tfrecord_path in tfrecord_paths:
    problems_gen = process_programs(
        tfrecord_path=tfrecord_path,
        tokenizer_path=tokenizer_path,
        fraction=fraction)
    basename = os.path.basename(tfrecord_path)

    train_path = codenet_paths.make_tfrecord_path(dataset_path, 'train')
    valid_path = codenet_paths.make_tfrecord_path(dataset_path, 'valid')
    test_path = codenet_paths.make_tfrecord_path(dataset_path, 'test')
    with tf.io.TFRecordWriter(train_path) as train_file_writer:
      with tf.io.TFRecordWriter(valid_path) as valid_file_writer:
        with tf.io.TFRecordWriter(test_path) as test_file_writer:
          for index, problem in enumerate(problems_gen):
            record_bytes = data_io.to_tf_example(problem).SerializeToString()
            r = random.random()
            if r < 0.8:
              train_file_writer.write(record_bytes)
            elif r < 0.9:
              valid_file_writer.write(record_bytes)
            else:
              test_file_writer.write(record_bytes)


def process_programs(
    tfrecord_path,
    tokenizer_path=DEFAULT_TOKENIZER_PATH,
    fraction=1.0,
    start_at=0):
  """Makes RuntimeErrorProblem objects per program using the tokenizer."""
  tokenizer = tokenization.load_tokenizer(path=tokenizer_path)

  basename = os.path.basename(tfrecord_path)
  tfrecord_paths = [tfrecord_path]
  dataset = cfp_raise_data_io.load_dataset(tfrecord_paths)

  count = 0
  for index, example in enumerate(dataset):
    if random.random() > fraction:
      # Only use a random `fraction` of the submissions.
      continue
    count += 1
    if count < start_at:
      continue

    source = example['source'][0].numpy().decode('utf-8')
    target = example['target'][0].numpy().decode('utf-8')
    original_step_limit = example['steps'][0].numpy()

    if target == 'RuntimeError':
      target_index = 1000  # Error class.
    else:
      target_index = int(target)

    problem = process.make_runtimeerrorproblem(
        source, target_index, tokenizer=tokenizer,
        problem_id=basename, submission_id=str(index))
    assert problem.step_limit == original_step_limit
    yield problem

    if count % 1000 == 0:
      print(count)


if __name__ == '__main__':
  fire.Fire()
