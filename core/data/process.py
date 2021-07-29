"""Dataset preprocessing."""

from typing import List, Text

import bisect
import dataclasses

import fire
import gast as ast
from python_graphs import control_flow
from python_graphs import instruction as instruction_module

from core.data import tokenization


@dataclasses.dataclass
class RawRuntimeErrorProblem:
  """RawRuntimeErrorProblem."""
  source: Text
  edge_sources: List[int]
  edge_dests: List[int]
  edge_types: List[int]
  node_span_starts: List[int]
  node_span_ends: List[int]
  branch_list: List[List[int]]
  exit_index: int
  target: int


@dataclasses.dataclass
class RuntimeErrorProblem:
  """RuntimeErrorProblem for use on an accelerator."""
  tokens: List[int]
  edge_sources: List[int]
  edge_dests: List[int]
  edge_types: List[int]
  node_token_span_starts: List[int]
  node_token_span_ends: List[int]
  token_node_indexes: List[int]
  true_branch_nodes: List[int]
  false_branch_nodes: List[int]
  exit_index: int
  target: int


def get_character_index(source, lineno, col_offset):
  lines = source.split('\n')
  line_index = lineno - 1
  line_start = sum(len(line) + 1 for line in lines[:line_index])
  return line_start + col_offset


def get_span(instruction):
  ast_node = instruction.node
  if instruction.source == instruction_module.EXCEPTION:
    # Caution: Leaky abstraction.
    # The accesses of an exception node are defined in control_flow's handle_ExceptHandler.
    # TODO(dbieber): Add parent accessor to instruction module.
    parent = instruction.accesses[0][-1]  # An AST ExceptHandler node.
    lineno = parent.lineno
    col_offset = parent.col_offset
    end_lineno = parent.end_lineno
    end_col_offset = parent.end_col_offset
  elif instruction.source == instruction_module.ARGS:
    arg0 = instruction.accesses[0][1]
    argN = instruction.accesses[-1][1]
    lineno = arg0.lineno
    col_offset = arg0.col_offset
    end_lineno = argN.end_lineno
    end_col_offset = argN.end_col_offset
  else:
    lineno = ast_node.lineno
    col_offset = ast_node.col_offset
    end_lineno = ast_node.end_lineno
    end_col_offset = ast_node.end_col_offset
  return lineno, col_offset, end_lineno, end_col_offset


def make_rawruntimeerrorproblem(source, target):
  """Constructs a RawRuntimeErrorProblem from the provided source and target.

  Fields:
  - source: The text of a program
  - edge_sources: Together with edge_dests, forms an adjacency list of all edges in the program's graph representation
  - edge_dests: Together with edge_sources, forms an adjacency list of all edges in the program's graph representation
  - edge_types: A list the same length as edge_sources and edge_dests, contains the integer enum type of each edge in the program's graph representation.
  - node_span_starts: A list of the source span start for each node in the program's graph representation.
  - node_span_ends: A list of the source span ends for each node in the program's graph representation.
  """
  graph = control_flow.get_control_flow_graph(source)
  nodes = graph.nodes

  # cfg.nodes does not include an exit node, so we add 1.
  num_nodes = len(nodes) + 1
  exit_index = len(nodes)

  # node_span_starts and node_span_ends
  node_span_starts = []
  node_span_ends = []
  node_indexes = {}
  for node_index, node in enumerate(nodes):
    node_indexes[node.uuid] = node_index

    lineno, col_offset, end_lineno, end_col_offset = get_span(node.instruction)
    node_span_start = get_character_index(source, lineno, col_offset)
    node_span_end = get_character_index(source, end_lineno, end_col_offset)
    node_span_starts.append(node_span_start)
    node_span_ends.append(node_span_end)

  # edge_sources, edge_dests, and edge_types
  edge_sources = []
  edge_dests = []
  edge_types = []
  for node_index, node in enumerate(nodes):
    for next_node in node.next:
      edge_sources.append(node_index)
      edge_dests.append(node_indexes[next_node.uuid])
      edge_types.append(0)

  branch_list = get_branch_list(nodes, exit_index)

  return RawRuntimeErrorProblem(
      source=source,
      edge_sources=edge_sources,
      edge_dests=edge_dests,
      edge_types=edge_types,
      node_span_starts=node_span_starts,
      node_span_ends=node_span_ends,
      branch_list=branch_list,
      exit_index=exit_index,
      target=target,
  )


def get_branch_list(nodes, exit_index):
  """Computes the branch list for the control flow graph.

  Args:
    nodes: A list of control_flow.ControlFlowNodes.
    exit_index: The index of the exit node.
  Returns:
    A Python list representing the branch options available from each node. Each
    entry in the list corresponds to a node in the control flow graph, with the
    final entry corresponding to the exit node (not present in the cfg). Each
    entry is a 2-tuple indicating the next node reached by the True and False
    branch respectively (these may be the same.) The exit node leads to itself
    along both branches.
  """
  indexes_by_id = {
      id(node): index for index, node in enumerate(nodes)
  }
  indexes_by_id[id(None)] = exit_index
  branches = []
  for node in nodes:
    node_branches = node.branches
    if node_branches:
      branches.append([indexes_by_id[id(node_branches[True])],
                       indexes_by_id[id(node_branches[False])]])
    else:
      try:
        next_node = next(iter(node.next))
        next_index = indexes_by_id[id(next_node)]
      except StopIteration:
        next_index = exit_index
      branches.append([next_index, next_index])

  # Finally we add branches from the exit node to itself.
  # Omit this if running on BasicBlocks rather than ControlFlowNodes, because
  # ControlFlowGraphs have an exit BasicBlock, but no exit ControlFlowNodes.
  branches.append([exit_index, exit_index])
  return branches


def make_runtimeerrorproblem(source, target, tokenizer=None):
  raw = make_rawruntimeerrorproblem(source, target)
  tokenizer = tokenizer or tokenization.load_tokenizer()
  token_data = tokenize_raw_with_spans(tokenizer, raw)
  return RuntimeErrorProblem(
      tokens=token_data['tokens'],
      edge_sources=raw.edge_sources,
      edge_dests=raw.edge_dests,
      edge_types=raw.edge_types,
      node_token_span_starts=token_data['node_token_span_starts'],
      node_token_span_ends=token_data['node_token_span_ends'],
      token_node_indexes=token_data['token_node_indexes'],
      true_branch_nodes=raw.branch_list[:, 0],
      false_branch_nodes=raw.branch_list[:, 1],
      exit_index=raw.exit_index,
      target=raw.target,
  )


def tokenize_raw_with_spans(tokenizer, raw):
  return tokenize_with_spans(tokenizer, raw.source, raw.node_span_starts, raw.node_span_ends, raw.target)


def tokenize_with_spans(tokenizer, source, node_span_starts, node_span_ends, target):
  tokenized = tokenizer(source, return_offsets_mapping=True)
  tokens = tokenized['input_ids']
  offset_mapping = tokenized['offset_mapping']
  if offset_mapping:
    token_starts, token_ends = zip(*offset_mapping)
  else:  # No tokens.
    token_starts, token_ends = tuple(), tuple()

  node_token_span_starts = []
  node_token_span_ends = []
  token_node_indexes = [-1] * len(tokens)
  for i, (node_span_start, node_span_end) in enumerate(zip(node_span_starts, node_span_ends)):
    # First token starting before or at node_span_start:
    node_token_span_start = bisect.bisect_left(token_starts, node_span_start)
    while token_starts[node_token_span_start] > node_span_start:
      node_token_span_start -= 1
    # First token starting after or at node_span_end:
    node_token_span_end = bisect.bisect_left(token_ends, node_span_end)

    node_token_span_starts.append(node_token_span_start)
    node_token_span_ends.append(node_token_span_end)
    token_node_indexes[node_token_span_start:node_token_span_end] = (
        [i] * (node_token_span_end - node_token_span_start))

  return {
      'tokens': tokens,
      'node_token_span_starts': node_token_span_starts,
      'node_token_span_ends': node_token_span_ends,
      'token_node_indexes': token_node_indexes,
  }


def demo_parse_code():
  """Demonstration of making and processing a RuntimeErrorProblem."""
  source = """n = input()
print(any(set('47') >= set(str(i)) and n % i == 0 for i in range(1, n+1)) and 'YES' or 'NO')
"""
  raw = make_rawruntimeerrorproblem(source, 'n/a')
  tokenizer = tokenization.load_tokenizer()
  data = tokenize_raw_with_spans(tokenizer, raw)


if __name__ == '__main__':
  fire.Fire()
