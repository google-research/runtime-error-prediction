"""Compute the distribution of program sizes."""

import dataclasses
import itertools
from typing import List, Optional

import fire
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

  max_tokens: int = 512
  max_num_nodes: int = 128
  max_steps: int = 174
  allowlist: Optional[List[int]] = None

  def load_dataset(self, dataset_path=DEFAULT_DATASET_PATH, split='train'):
    allowlist = self.allowlist
    if allowlist == 'TIER1_ERROR_IDS':
      allowlist = error_kinds.TIER1_ERROR_IDS
    filter_fn = data_io.make_filter(
        self.max_tokens, self.max_num_nodes, self.max_steps, allowlist=allowlist)

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
      tokens = tokenizer.convert_ids_to_tokens(example['tokens'])
      for i, (span_start, span_end, true_node, false_node, raise_node) in enumerate(zip(span_starts, span_ends, true_branch_nodes, false_branch_nodes, raise_nodes)):
        print(f"""Span {i} (--> {true_node},{false_node},{raise_node}): {' '.join(tokens[span_start:span_end + 1])}""")

  def run_counter(self, dataset_path=DEFAULT_DATASET_PATH, split='train', steps=None):
    print(f'Analyzing data: {dataset_path}')
    dataset = self.load_dataset(dataset_path, split=split)
    targets = []
    num_tokens = []
    num_nodes = []
    step_limits = []
    for step, example in itertools.islice(enumerate(tfds.as_numpy(dataset)), steps):
      targets.append(example['target'][0])
      num_tokens.append(example['num_tokens'][0])
      num_nodes.append(example['num_nodes'][0])
      step_limits.append(example['step_limit'][0])

    return (targets, num_tokens, num_nodes, step_limits)


if __name__ == '__main__':
  fire.Fire()
