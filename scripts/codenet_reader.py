"""An example of reading the CodeNet processed datasets from disk."""

import fire
import tensorflow as tf
import tensorflow_datasets as tfds

from core.data import codenet_paths
from core.data import data_io

DEFAULT_DATASET_PATH = codenet_paths.DEFAULT_DATASET_PATH


def load(dataset_path=DEFAULT_DATASET_PATH):
  dataset = tf.data.TFRecordDataset(
      [dataset_path],
      compression_type=None, buffer_size=None, num_parallel_reads=None
  ).map(data_io.decode_fn).padded_batch(8)
  for example in tfds.as_numpy(dataset):
    print(example)

if __name__ == '__main__':
  fire.Fire()
