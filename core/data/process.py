"""Dataset preprocessing."""

import builtins
from typing import List, Optional, Text

import bisect
import collections
import dataclasses

import fire
import gast as ast
import numpy as np
from python_graphs import control_flow
from python_graphs import instruction as instruction_module

from core.data import tokenization


@dataclasses.dataclass
class RawRuntimeErrorProblem:
  """RawRuntimeErrorProblem."""
  source: Text
  problem_id: Optional[Text]
  submission_id: Optional[Text]
  edge_sources: List[int]
  edge_dests: List[int]
  edge_types: List[int]
  node_span_starts: List[int]
  node_span_ends: List[int]
  branch_list: List[List[int]]
  exit_index: int
  step_limit: int
  target: int
  target_lineno: Optional[int]


@dataclasses.dataclass
class RuntimeErrorProblem:
  """RuntimeErrorProblem for use on an accelerator."""
  tokens: List[int]
  problem_id: Text
  submission_id: Text
  edge_sources: List[int]
  edge_dests: List[int]
  edge_types: List[int]
  node_token_span_starts: List[int]
  node_token_span_ends: List[int]
  token_node_indexes: List[int]
  true_branch_nodes: List[int]
  false_branch_nodes: List[int]
  exit_index: int
  step_limit: int
  target: int
  target_lineno: Optional[int]
  target_node_indexes: List[int]


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


def examine_udfs(graph, problem_id, submission_id):
  nodes = graph.nodes
  ast_nodes = [n.instruction.node for n in nodes]

  # This doesn't consider the scope of a function, such as if
  # it is defined inside a class.
  nodes_by_function_name = {
      ast_node.name: ast_node
      for ast_node in ast_nodes
      if isinstance(ast_node, (ast.FunctionDef, ast.ClassDef))
  }

  # We're interested in splitting instructions that call user defined functions
  # into multiple nodes. We won't do this for FunctionDef, ClassDef.
  # If it's a class, we'll point to the init function.
  total_function_calls = 0
  calls_by_function_name = collections.defaultdict(int)
  for node in nodes:
    if isinstance(node.instruction.node, (ast.FunctionDef, ast.ClassDef)):
      continue

    num_func_calls = 0
    for ast_node in ast.walk(node.instruction.node):
      if isinstance(ast_node, ast.Call):
        if isinstance(ast_node.func, ast.Name):
          # e.g. "func_name()"
          function_name = ast_node.func.id
        elif isinstance(ast_node.func, ast.Attribute):
          # e.g. "o.func_name()"
          function_name = ast_node.func.attr
        else:
          # e.g. o[0]() (ast.Subscript)
          continue
        if function_name in nodes_by_function_name:
          num_func_calls += 1
          calls_by_function_name[function_name] += 1
          total_function_calls += num_func_calls
        elif function_name in dir(builtins):
          # Builtin function called.
          pass
        else:  # Unknown function called
          pass
  if calls_by_function_name.values() and max(calls_by_function_name.values()) > 1:
    n = max(calls_by_function_name.values())
    for f in calls_by_function_name:
      if calls_by_function_name[f] == n:
        break
    print(f'Calling function {f} {n} times')
    return 'Function called more than once'
  elif total_function_calls == 0:
    return 'No UDFs called'
  else:
    return 'UDFs called at most once'


def make_rawruntimeerrorproblem(
    source, target, target_lineno=None, problem_id=None, submission_id=None):
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
  lines = source.strip().split('\n')
  nodes = graph.nodes

  udf_usage = examine_udfs(graph, problem_id, submission_id)
  if udf_usage != 'No UDFs called':
    raise ValueError('UDF not currently supported.')

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
  step_limit = get_step_limit(lines)

  return RawRuntimeErrorProblem(
      source=source,
      problem_id=problem_id,
      submission_id=submission_id,
      edge_sources=edge_sources,
      edge_dests=edge_dests,
      edge_types=edge_types,
      node_span_starts=node_span_starts,
      node_span_ends=node_span_ends,
      branch_list=branch_list,
      exit_index=exit_index,
      step_limit=step_limit,
      target=target,
      target_lineno=target_lineno,
  )


def get_step_limit(lines):
  """Computes the maximum number of IPA-GNN steps allowed for a program."""
  step_limit = 1  # Start with one step for reaching exit.
  indents = []
  for line in lines:
    indent = len(line) - len(line.lstrip())
    while indents and indent <= indents[-1]:
      indents.pop()
    step_limit += 2 ** len(indents)
    if (line.lstrip().startswith('for') or line.lstrip().startswith('while')):
      indents.append(indent)
      # We add steps at both levels of indentation for loops.
      # Before for the initial condition check, after for subsequent condition
      # checks.
      step_limit += 2 ** len(indents)
  return step_limit


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


def get_nodes_at_lineno(raw, lineno):
  if lineno is None:
    return []

  # Compute the line boundaries.
  line_index = lineno - 1
  lines = raw.source.split('\n')
  line_starts = [0]
  current_line_start = 0
  for line in lines:
    current_line_start += len(line) + 1
    line_starts.append(current_line_start)

  line_start = line_starts[line_index]
  line_end = line_starts[line_index + 1]

  # Determine which nodes intersect the line.
  overlapping_nodes = []
  for node, (start, end) in enumerate(zip(raw.node_span_starts, raw.node_span_ends)):
    if (line_start <= start <= line_end
        or line_start <= end <= line_end
        or start <= line_start <= end
        or start <= line_end <= end):
      overlapping_nodes.append(node)
  return overlapping_nodes


def make_runtimeerrorproblem(source, target, target_lineno=None, tokenizer=None,
                             problem_id=None, submission_id=None):
  raw = make_rawruntimeerrorproblem(
        source, target, target_lineno=target_lineno,
        problem_id=problem_id, submission_id=submission_id)
  tokenizer = tokenizer or tokenization.load_tokenizer()
  token_data = tokenize_raw_with_spans(tokenizer, raw)
  branch_list = np.array(raw.branch_list)
  target_node_indexes = get_nodes_at_lineno(raw, target_lineno)
  return RuntimeErrorProblem(
      tokens=token_data['tokens'],
      problem_id=raw.problem_id,
      submission_id=raw.submission_id,
      edge_sources=raw.edge_sources,
      edge_dests=raw.edge_dests,
      edge_types=raw.edge_types,
      node_token_span_starts=token_data['node_token_span_starts'],
      node_token_span_ends=token_data['node_token_span_ends'],
      token_node_indexes=token_data['token_node_indexes'],
      true_branch_nodes=branch_list[:, 0],
      false_branch_nodes=branch_list[:, 1],
      exit_index=raw.exit_index,
      step_limit=raw.step_limit,
      target=raw.target,
      target_lineno=raw.target_lineno,
      target_node_indexes=target_node_indexes,
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
