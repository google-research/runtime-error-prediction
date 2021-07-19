"""Dataset preprocessing."""

from typing import List, Text

import bisect
import dataclasses

import fire
import gast as ast
from python_graphs import control_flow
from python_graphs import instruction as instruction_module

from core.data import tokenize


@dataclasses.dataclass
class RawRuntimeErrorProblem:
  """RawRuntimeErrorProblem."""
  source: Text
  edge_sources: List[int]
  edge_dests: List[int]
  edge_types: List[int]
  node_span_starts: List[int]
  node_span_ends: List[int]
  target: Text


@dataclasses.dataclass
class RuntimeErrorProblem:
  """RuntimeErrorProblem for use on an accelerator."""
  tokens: List[int]
  edge_sources: List[int]
  edge_dests: List[int]
  edge_types: List[int]
  node_token_span_starts: List[int]
  node_token_span_ends: List[int]
  target: Text


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
  elif isinstance(ast_node, ast.arguments):
    arg0 = ast_node.args[0]
    argN = ast_node.args[-1]
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

  # node_span_starts and node_span_ends
  node_span_starts = []
  node_span_ends = []
  node_indexes = {}
  for node_index, node in enumerate(graph.nodes):
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
  for node_index, node in enumerate(graph.nodes):
    for next_node in node.next:
      edge_sources.append(node_index)
      edge_dests.append(node_indexes[next_node.uuid])
      edge_types.append(0)

  return RawRuntimeErrorProblem(
      source=source,
      edge_sources=edge_sources,
      edge_dests=edge_dests,
      edge_types=edge_types,
      node_span_starts=node_span_starts,
      node_span_ends=node_span_ends,
      target=target,
  )


def make_runtimeerrorproblem(source, target, tokenizer=None):
  raw = make_rawruntimeerrorproblem(source, target)
  tokenizer = tokenizer or tokenize.load_tokenizer()
  token_data = tokenize_raw_with_spans(tokenizer, raw)
  return RuntimeErrorProblem(
      tokens=token_data['tokens'],
      edge_sources=raw.edge_sources,
      edge_dests=raw.edge_dests,
      edge_types=raw.edge_types,
      node_token_span_starts=token_data['node_token_span_starts'],
      node_token_span_ends=token_data['node_token_span_ends'],
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
  for i, (node_span_start, node_span_end) in enumerate(zip(node_span_starts, node_span_ends)):
    try:
      # Want first token starting before or at node_span_start
      node_token_span_start = bisect.bisect_left(token_starts, node_span_start)
      while token_starts[node_token_span_start] > node_span_start:
        node_token_span_start -= 1
      # Want first token starting after or at node_span_end
      node_token_span_end = bisect.bisect_left(token_ends, node_span_end)
    except ValueError:
      print('ValueError debug info')
      print(source)
      print(target)
      print(tokens)
      print(token_starts)
      print(node_span_start)
      print(token_ends)
      print(node_span_end)
      raise
    node_token_span_starts.append(node_token_span_start)
    node_token_span_ends.append(node_token_span_end)

  return {
      'tokens': tokens,
      'node_token_span_starts': node_token_span_starts,
      'node_token_span_ends': node_token_span_ends,
  }


def demo_parse_code():
  """Demonstration of making and processing a RuntimeErrorProblem."""
  source = """n = input()
print(any(set('47') >= set(str(i)) and n % i == 0 for i in range(1, n+1)) and 'YES' or 'NO')
"""
  raw = make_rawruntimeerrorproblem(source, 'n/a')
  tokenizer = tokenize.load_tokenizer()
  data = tokenize_raw_with_spans(tokenizer, raw)


if __name__ == '__main__':
  fire.Fire()
