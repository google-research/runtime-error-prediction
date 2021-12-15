"""Generates Control Flow Programs.

This file was introduced as part of the Exception IPA-GNN effort, for generating
a new dataset suitable for testing the vanilla IPA-GNN and Exception IPA-GNN.
"""

import collections
import dataclasses
import os
import random
from typing import Optional, Sequence, Text, Tuple

from absl import app
from python_graphs import control_flow
import tensorflow as tf
import tqdm

from core.data import codenet_paths
from core.data import process
from core.data.generation import program_generator
from core.data.generation import python_interpreter

TFRECORD_PATH = codenet_paths.RAW_CFP_RAISE_DATA_PATH
TFRECORD_PATH = 'tmp.tfrecord'
ASSERTION_ERROR_PROB = 0.5
ADD_ASSERTION_ERRO = True

DEFAULT_OPS = ("+=", "-=", "*=")


@dataclasses.dataclass
class ArithmeticIfRepeatsConfig:
  """Config for ArithmeticIfRepeats ProgramGenerator.

  Attributes:
    base: The base to represent the integers in.
    length: The number of statements in the generated programs.
    num_digits: The number of digits in the values used by the programs.
    max_repeat_statements: The maximum number of repeat statements allowed in
      a program.
    max_repetitions: The maximum number of repetitions a repeat statement may
      specify.
    repeat_probability: The probability that a given statement is a repeat
      statement, provided a repeat statement is possible at that location.
    max_if_statements: The maximum number of if statements allowed in a program.
    if_probability: The probability that a given statement is an if statement,
      provided an if statement is possible at that location.
    ifelse_probability: The probability that a given statement is an if-else
      statement, provided an if statement is possible at that location.
    max_nesting: The maximum depth of nesting permitted, or None if no limit.
    max_block_size: The maximum number of statements permitted in a block.
    ops: The ops allowed in the generated programs.
    encoder_name: The encoder name to use to encode the generated programs.
    mod: The value (if any) to mod the intermediate values of the program by
      after each step of execution.
    output_mod: The value (if any) to mod the final values of the program by.
  """
  base: int
  length: int
  num_digits: int = 1
  max_repeat_statements: Optional[int] = 2
  max_repetitions: int = 9
  repeat_probability: float = 0.1
  max_if_statements: Optional[int] = 2
  if_probability: float = 0.2
  ifelse_probability: float = 0.2
  max_nesting: Optional[int] = None
  max_block_size: Optional[int] = 9
  ops: Tuple[Text, ...] = DEFAULT_OPS
  encoder_name: Text = "simple"
  mod: Optional[int] = 10
  output_mod: Optional[int] = None



def int64_feature(value):
  """Constructs a tf.train.Feature for the given int64 value list."""
  return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def bytes_feature(values):
  """Constructs a tf.train.Feature for the given str value list."""
  values = [v.encode('utf-8') for v in values]
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=values))


def to_tf_example(source, target, steps):
  """Constructs a tf.train.Example for the source code."""
  return tf.train.Example(features=tf.train.Features(feature={
      'source': bytes_feature([source]),
      'target': bytes_feature([target]),
      'steps': int64_feature([steps]),
  }))


def decode_fn(record_bytes):
  features = {
      'source': tf.io.FixedLenFeature([1], dtype=tf.string),
      'target': tf.io.FixedLenFeature([1], dtype=tf.string),
      'steps': tf.io.FixedLenFeature([1], dtype=tf.int64),
  }
  return tf.io.parse_single_example(record_bytes, features)


def load_dataset(tfrecord_paths):
  return tf.data.TFRecordDataset(
      tfrecord_paths,
      compression_type=None, buffer_size=None, num_parallel_reads=32
  ).map(decode_fn)


def read():
  for example in load_dataset([TFRECORD_PATH]):
    source = example['source'].numpy()[0].decode('utf-8')
    target = example['target'].numpy()[0].decode('utf-8')
    print(source)
    print('---')
    # if 'raise' in source:
    #   print(target)


def generate_example_from_python_source(executor, base, python_source, mod, output_mod):
  """Generates an example dict from the given statements."""
  cfg = control_flow.get_control_flow_graph(python_source)
  python_source_lines = python_source.strip().split("\n")

  values = {"v0": 1}  # Assume v0 starts at 1.
  try:
    values = python_interpreter.evaluate_cfg(
        executor, cfg, mod=mod,
        initial_values=values,
        timeout=200)
    error_type = "NoError"
  except Exception as e:  # pylint: disable=broad-except
    error_type = type(e).__name__
  target_output = values["v0"]

  if output_mod is not None:
    try:
      target_output %= output_mod
    except TypeError:
      target_output = 1

  return {
      'human_readable_target_output': str(target_output),
      'error_type': error_type
  }


def add_assert_error(source, example):
  if example['error_type'] == 'RuntimeError':
    return source, example
  is_error = random.choices([0,1], [1-ASSERTION_ERROR_PROB, ASSERTION_ERROR_PROB])[0]
  add_val = random.randint(1,10)
  current_val = int(example['human_readable_target_output'])
  if is_error:
    source = f"{source}\nassert v0=={abs(current_val+add_val)%1000}"
    example['error_type'] = "AssertionError"
  else:
    source = f"{source}\nassert v0=={current_val}"
  return source, example


def main(argv: Sequence[str]) -> None:
  del argv  # Unused.

  # if os.path.exists(TFRECORD_PATH):
  #   return read()

  executor = python_interpreter.ExecExecutor()
  counts = collections.Counter()
  program_generator_config = ArithmeticIfRepeatsConfig(
      base=10,
      max_if_statements=5,
      length=30,
  )
  with tf.io.TFRecordWriter(TFRECORD_PATH) as file_writer:
    for _ in tqdm.tqdm(range(50)):
      source = program_generator.generate_python_source(
          30, program_generator_config)
      print(source)
      print()

      example = (
          generate_example_from_python_source(
              executor, program_generator_config.base, source,
              mod=1000,
              output_mod=1000,
          )
      )
      print(example)

      source, example = add_assert_error(source, example)

      target = example['human_readable_target_output']
      error_type = example['error_type']
      lines = source.split('\n')
      steps = process.get_step_limit(lines)
      counts[target] += 1

      if error_type != 'NoError':
        target = error_type
      record_bytes = to_tf_example(source, target, steps).SerializeToString()
      file_writer.write(record_bytes)
  print(dict(counts))


if __name__ == '__main__':
  app.run(main)
