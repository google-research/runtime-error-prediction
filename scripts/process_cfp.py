import glob
import random

import fire

from core.data import cfp_data_io
from core.data import codenet_paths
from core.data import process
from core.data import tokenization

DEFAULT_CFP_DATA_PATTERN = codenet_paths.DEFAULT_CFP_DATA_PATTERN
DEFAULT_TOKENIZER_PATH = codenet_paths.DEFAULT_TOKENIZER_PATH


def process_control_flow_programs(
    tfrecord_pattern=DEFAULT_CFP_DATA_PATTERN,
    tokenizer_path=DEFAULT_TOKENIZER_PATH,
    fraction=1.0,
    start_at=0):
  """Makes RuntimeErrorProblem objects per program using the tokenizer."""
  tokenizer = tokenization.load_tokenizer(path=tokenizer_path)

  tfrecord_paths = glob.glob(tfrecord_pattern)
  dataset = cfp_data_io.load_dataset(tfrecord_paths, include_strings=True)

  count = 0
  for index, example in enumerate(dataset):
    if random.random() > fraction:
      # Only use a random `fraction` of the submissions.
      continue
    count += 1
    if count < start_at:
      continue

    python_path = f'{tfrecord_pattern}:{index}'
    source = example['human_readable_code'].decode('utf-8')
    target = example['target_output']

    problem = process.make_runtimeerrorproblem(
        source, target, tokenizer=tokenizer,
        problem_id=index, submission_id=0)
    yield problem

    if count % 1000 == 0:
      print(count)


if __name__ == '__main__':
  fire.Fire()
