"""An example of reading the CodeNet processed datasets from disk."""

import fire
import tensorflow as tf

from core.data import data_io

DEFAULT_TOKENIZER_PATH = 'out/tokenizers/full.json'
DEFAULT_DATASET_PATH = 'out/data/default.tfrecord'


def load(
    dataset_path=DEFAULT_DATASET_PATH,
    tokenizer_path=DEFAULT_TOKENIZER_PATH):
  dataset = tf.data.TFRecordDataset(
      [dataset_path],
      compression_type=None, buffer_size=None, num_parallel_reads=None
  ).map(data_io.decode_fn).batch(8)
  for batch in dataset:
    print(batch)

if __name__ == '__main__':
  fire.Fire()
