"""Dataset preprocessing."""

import fire

from typing import List, Text

import dataclasses
from python_graphs import control_flow


@dataclasses.dataclass
class RuntimeErrorProblem:
  """RuntimeErrorProblem"""
  source: Text
  edge_sources: List[int]
  edge_dests: List[int]
  edge_types: List[int]
  node_span_starts: List[int]
  node_span_ends: List[int]


def get_character_index(source, lineno, col_offset):
  lines = source.split('\n')
  line_index = lineno - 1
  line_start = sum(len(line) for line in lines[:line_index])
  return line_start + col_offset


def process(source, target):
  """Constructs a RuntimeErrorProblem from the provided source and target.

  TODO(dbieber): Use target in RuntimeErrorProblem.

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

    ast_node = node.instruction.node
    lineno = ast_node.lineno
    col_offset = ast_node.col_offset
    end_lineno = ast_node.end_lineno
    end_col_offset = ast_node.end_col_offset
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

  return RuntimeErrorProblem(
      source=source,
      edge_sources=edge_sources,
      edge_dests=edge_dests,
      edge_types=edge_types,
      node_span_starts=node_span_starts,
      node_span_ends=node_span_ends,
  )

if __name__ == '__main__':
  fire.Fire()
