"""Compute the distribution of program sizes."""

import dataclasses
import itertools
from typing import List, Optional

import fire
import numpy as np
import tensorflow_datasets as tfds

from core.data import codenet_paths
from core.data import data_io
from core.data import error_kinds
from core.data import explore
from core.data import tokenization


DEFAULT_DATASET_PATH = codenet_paths.DEFAULT_DATASET_PATH
DEFAULT_TOKENIZER_PATH = codenet_paths.DEFAULT_TOKENIZER_PATH


def pairwise(iterable):
  a, b = itertools.tee(iterable)
  next(b, None)
  return zip(a, b)


@dataclasses.dataclass
class Analyzer:

  filter_data: bool = True
  max_tokens: int = 512
  max_num_nodes: int = 128
  max_num_edges: int = 128
  max_steps: int = 174
  allowlist: Optional[List[int]] = None

  def load_dataset(self, dataset_path=DEFAULT_DATASET_PATH, split='train'):
    allowlist = self.allowlist
    if allowlist == 'TIER1_ERROR_IDS':
      allowlist = error_kinds.TIER1_ERROR_IDS
    if self.filter_data:
      filter_fn = data_io.make_filter(
          self.max_tokens, self.max_num_nodes, self.max_num_edges,
          self.max_steps, allowlist=allowlist)
    else:
      filter_fn = lambda example: True

    # Return the requested dataset.
    return (
        data_io.load_dataset(dataset_path, split=split, include_strings=True)
        .filter(filter_fn)
    )

  def look_for_overlapping_spans(
      self, dataset_path=DEFAULT_DATASET_PATH, tokenizer_path=DEFAULT_TOKENIZER_PATH,
      split='train', steps=None):
    tokenizer = tokenization.load_tokenizer(path=tokenizer_path)
    dataset = self.load_dataset(dataset_path, split=split)
    for step, example in itertools.islice(enumerate(tfds.as_numpy(dataset)), steps):
      span_starts = example['node_token_span_starts']
      span_ends = example['node_token_span_ends']
      # Recall, spans are inclusive.
      for (span_start, span_end), (next_span_start, next_span_end) in pairwise(zip(span_starts, span_ends)):
        if (span_start <= next_span_start <= span_end
            or span_start <= next_span_end <= span_end
            or next_span_start <= span_start <= next_span_end
            or next_span_start <= span_end <= next_span_end):
          print((span_start, span_end), (next_span_start, next_span_end))
          source = tokenizer.convert_ids_to_tokens(example['tokens'])
          print(f"""
Span 1: {source[span_start:span_end + 1]}
Span 2: {source[next_span_start:next_span_end + 1]}

Source: {' '.join(source)}

Submission ID: {example['problem_id'][0].decode('utf-8')} {example['submission_id'][0].decode('utf-8')}

""")

  def inspect_spans(
      self, dataset_path=DEFAULT_DATASET_PATH, tokenizer_path=DEFAULT_TOKENIZER_PATH,
      split='train', steps=None):
    tokenizer = tokenization.load_tokenizer(path=tokenizer_path)
    dataset = self.load_dataset(dataset_path, split=split)
    for step, example in itertools.islice(enumerate(tfds.as_numpy(dataset)), steps):
      span_starts = example['node_token_span_starts']
      span_ends = example['node_token_span_ends']
      true_branch_nodes = example['true_branch_nodes']
      false_branch_nodes = example['false_branch_nodes']
      raise_nodes = example['raise_nodes']
      # Recall, spans are inclusive.
      submission_id = example['submission_id'][0].decode('utf-8')
      problem_id = example['problem_id'][0].decode('utf-8')
      source, target = explore.get_source_and_target_for_submission(problem_id, submission_id)
      print(f"""Submission ID: {submission_id} {problem_id}
Source: {source}""")
      print(example['target'])
      print(example['target_lineno'])
      print(example['target_node_indexes'])
      print(example['num_target_nodes'])
      tokens = tokenizer.convert_ids_to_tokens(example['tokens'])
      for i, (span_start, span_end, true_node, false_node, raise_node) in enumerate(zip(span_starts, span_ends, true_branch_nodes, false_branch_nodes, raise_nodes)):
        print(f"""Span {i} (--> {true_node},{false_node},{raise_node}): {' '.join(tokens[span_start:span_end + 1])}""")

  def inspect_targets(
      self, dataset_path=DEFAULT_DATASET_PATH, tokenizer_path=DEFAULT_TOKENIZER_PATH,
      split='train', steps=None):
    tokenizer = tokenization.load_tokenizer(path=tokenizer_path)
    dataset = self.load_dataset(dataset_path, split=split)
    ok, nok = 0, 0
    for step, example in itertools.islice(enumerate(tfds.as_numpy(dataset)), steps):
      span_starts = example['node_token_span_starts']
      span_ends = example['node_token_span_ends']
      true_branch_nodes = example['true_branch_nodes']
      false_branch_nodes = example['false_branch_nodes']
      raise_nodes = example['raise_nodes']
      # Recall, spans are inclusive.
      submission_id = example['submission_id'][0].decode('utf-8')
      problem_id = example['problem_id'][0].decode('utf-8')
      if example['num_target_nodes'][0] == 0:
        continue
      print(f"""Submission ID: {submission_id} {problem_id}""")
      print(f"Target:  {example['target'][0]}")
      print(f"Lineno:  {example['target_lineno'][0]}")
      print(f"Indices: {example['target_node_indexes']}")
      if 0 in example['target_node_indexes']:
        ok += 1
      else:
        nok += 1
      print(f"#Index:  {example['num_target_nodes'][0]}")
      print(ok, nok, ok/(ok + nok) * 100)

  def run_counter(self, dataset_path=DEFAULT_DATASET_PATH, split='train', steps=None):
    print(f'Analyzing data: {dataset_path}')
    dataset = self.load_dataset(dataset_path, split=split)
    targets = []
    num_tokens = []
    num_edges = []
    num_nodes = []
    step_limits = []
    target_lineno = []
    num_target_nodes = []
    for step, example in itertools.islice(enumerate(tfds.as_numpy(dataset)), steps):
      targets.append(example['target'][0])
      num_tokens.append(example['num_tokens'][0])
      num_edges.append(example['num_edges'][0])
      num_nodes.append(example['num_nodes'][0])
      step_limits.append(example['step_limit'][0])
      target_lineno.append(example['target_lineno'][0])
      num_target_nodes.append(example['num_target_nodes'][0])
      if step % 1000 == 0:
        print(step)
        token_nums = np.array(num_tokens)
        print(np.sum(token_nums > 512), np.sum(token_nums <= 512), len(num_tokens))

    return (targets, num_tokens, num_edges, num_nodes, step_limits, target_lineno, num_target_nodes)


if __name__ == '__main__':
  fire.Fire()
