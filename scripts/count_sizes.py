"""Compute the distribution of program sizes."""

import dataclasses
import itertools
from typing import Any, List, Optional, Text

import fire
from flax import linen as nn
from flax.training import common_utils
from flax.training import train_state
import jax
import jax.numpy as jnp
import ml_collections
import optax
import tensorflow_datasets as tfds

from core.data import codenet_paths
from core.data import data_io
from core.data import error_kinds


DEFAULT_DATASET_PATH = codenet_paths.DEFAULT_DATASET_PATH




@dataclasses.dataclass
class Analyzer:

  max_tokens: int = 256
  max_num_nodes: int = 80
  max_num_edges: int = 160
  max_steps: int = 100
  hidden_size: int = 10
  allowlist: Optional[List[int]] = None

  def load_dataset(self, dataset_path=DEFAULT_DATASET_PATH, split='train'):
    allowlist = self.allowlist
    if allowlist == 'TIER1_ERROR_IDS':
      allowlist = error_kinds.TIER1_ERROR_IDS
    filter_fn = data_io.make_filter(
        self.max_tokens, self.max_num_nodes, self.max_num_edges,
        self.max_steps, allowlist=allowlist)

    # Return the requested dataset.
    return (
        data_io.load_dataset(dataset_path, split=split)
        .filter(filter_fn)
    )

  def run(self, dataset_path=DEFAULT_DATASET_PATH, split='train', steps=None):
    print(f'Training on data: {dataset_path}')
    dataset = self.load_dataset(dataset_path, split=split)
    for step, example in itertools.islice(enumerate(tfds.as_numpy(dataset)), steps):
      print(example['num_edges'])



if __name__ == '__main__':
  fire.Fire()
