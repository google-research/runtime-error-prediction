"""An example of reading the CodeNet processed datasets from disk."""

import fire
import tensorflow as tf
import tensorflow_datasets as tfds

from core.data import codenet_paths
from core.data import data_io

DEFAULT_DATASET_PATH = codenet_paths.DEFAULT_DATASET_PATH


def load(dataset_path=DEFAULT_DATASET_PATH):
  max_tokens = 1024
  max_num_nodes = 80
  max_num_edges = 160
  dataset = tf.data.TFRecordDataset(
      [dataset_path],
      compression_type=None, buffer_size=None, num_parallel_reads=None
  ).map(data_io.decode_fn).padded_batch(8, padded_shapes={
      'tokens': [max_tokens],
      'edge_sources': [max_num_edges],
      'edge_dests': [max_num_edges],
      'edge_types': [max_num_edges],
      'node_token_span_starts': [max_num_nodes],
      'node_token_span_ends': [max_num_nodes],
      'target': [1],
  })
  for example in tfds.as_numpy(dataset):
    print(example)

if __name__ == '__main__':
  fire.Fire()
