"""Compute the distribution of program sizes."""

import dataclasses
import itertools
from typing import Any, List, Optional, Text

import fire
import jax
import jax.numpy as jnp
import tensorflow_datasets as tfds

from core.data import codenet_paths
from core.data import data_io
from core.data import error_kinds


DEFAULT_DATASET_PATH = codenet_paths.DEFAULT_DATASET_PATH


@dataclasses.dataclass
class Analyzer:

  max_tokens: int = 512
  max_num_nodes: int = 128
  max_num_edges: int = 128
  max_steps: int = 174
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

  def look_for_overlapping_spans(self, dataset_path=DEFAULT_DATASET_PATH, split='train', steps=None):
    dataset = self.load_dataset(dataset_path, split=split)
    for step, example in itertools.islice(enumerate(tfds.as_numpy(dataset)), steps):
      span_starts = example['node_token_span_starts']
      span_ends = example['node_token_span_ends']
      print(zip(span_starts, span_ends))

  def run(self, dataset_path=DEFAULT_DATASET_PATH, split='train', steps=None):
    print(f'Analyzing data: {dataset_path}')
    dataset = self.load_dataset(dataset_path, split=split)
    targets = []
    num_tokens = []
    num_edges = []
    num_nodes = []
    step_limits = []
    for step, example in itertools.islice(enumerate(tfds.as_numpy(dataset)), steps):
      targets.append(example['target'][0])
      num_tokens.append(example['num_tokens'][0])
      num_edges.append(example['num_edges'][0])
      num_nodes.append(example['num_nodes'][0])
      step_limits.append(example['step_limit'][0])

    return (targets, num_tokens, num_edges, num_nodes, step_limits)


if __name__ == '__main__':
  fire.Fire()
