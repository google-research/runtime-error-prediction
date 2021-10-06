"""Tests for process.py."""

import unittest

from core.data import process
from core.data import tokenization

def print_spans(raw):
  span_starts = raw.node_span_starts
  span_ends = raw.node_span_ends
  for i, (span_start, span_end) in enumerate(zip(span_starts, span_ends)):
    print(f'Span {i}: {raw.source[span_start:span_end]}')

class ProcessTest(unittest.TestCase):

  def setUp(self):
    tokenization.generate_tokenizer()

  def test_make_runtimeerrorproblem(self):
    tokenizer = tokenization.load_tokenizer()
    source = """n = input()
print(any(set('47') >= set(str(i)) and n % i == 0 for i in range(1, n+1)) and 'YES' or 'NO')
"""
    target_lineno = 2
    problem = process.make_runtimeerrorproblem(
        source, '1', target_lineno=target_lineno, tokenizer=tokenizer)
    target_tokens = [
        66, 31, 99, 84, 125, 10, 377, 10, 214, 223, 315, 222, 237, 214, 10,
        166, 10, 61, 95, 148, 66, 8, 61, 79, 18, 89, 61, 80, 104, 10, 19, 14,
        66, 13, 19, 95, 148, 9, 240, 9, 82, 9, 239, 222]
    target_node_token_span_starts = [0, 4]
    target_node_token_span_ends = [3, 43]
    target_token_node_indexes = [
        # Line 1:
        0, 0, 0, -1,
        # Line 2:
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, -1,
    ]
    self.assertEqual(problem.tokens, target_tokens)
    self.assertEqual(
        problem.node_token_span_starts, target_node_token_span_starts)
    self.assertEqual(problem.node_token_span_ends, target_node_token_span_ends)
    self.assertEqual(problem.token_node_indexes, target_token_node_indexes)


  def test_make_runtimeerrorproblem_try_except(self):
    tokenizer = tokenization.load_tokenizer()
    source = """while True:
    try:
        a,b = map(int,input().split())
        print(len(str(a+b)))
    except EOFError:
        break
"""
    target_lineno = 0
    problem = process.make_runtimeerrorproblem(
        source, '1', target_lineno=target_lineno, tokenizer=tokenizer)
    exit_index = problem.exit_index
    raise_index = exit_index + 1
    self.assertEqual(exit_index, 4)
    self.assertEqual(raise_index, 5)
    self.assertEqual(problem.raise_nodes, [5, 3, 3, 5, 5])

    # 0: True
    # 1: a,b = map(int,input().split())
    # 2: print(len(str(a+b)))
    # 3: EOFError
    # 4: <exit>
    # 5: <raise>

    self.assertEqual(problem.true_branch_nodes.tolist(), [1, 2, 0, 4, 4])
    self.assertEqual(problem.false_branch_nodes.tolist(), [4, 2, 0, 5, 4])

  def test_make_runtimeerrorproblem_raise(self):
    tokenizer = tokenization.load_tokenizer()
    source = """while True:
    try:
        a,b = map(int,input().split())
        raise ValueError()
    except EOFError:
        break
"""
    target_lineno = 0
    problem = process.make_runtimeerrorproblem(
        source, '1', target_lineno=target_lineno, tokenizer=tokenizer)
    exit_index = problem.exit_index
    raise_index = exit_index + 1
    self.assertEqual(exit_index, 4)
    self.assertEqual(raise_index, 5)
    self.assertEqual(problem.raise_nodes, [5, 3, 3, 5, 5])

    # 0: True
    # 1: a,b = map(int,input().split())
    # 2: raise ValueError()
    # 3: EOFError
    # 4: <exit>
    # 5: <raise>

    # Notice the raise node sends control flow to the EOFError node.
    self.assertEqual(problem.true_branch_nodes.tolist(), [1, 2, 3, 4, 4])
    self.assertEqual(problem.false_branch_nodes.tolist(), [4, 2, 3, 5, 4])

  def test_make_runtimeerrorproblem_straight_line_code(self):
    tokenizer = tokenization.load_tokenizer()
    source = """
x = 1
y = x + 2
z = y * 3
"""
    target_lineno = 0
    problem = process.make_runtimeerrorproblem(
        source, '1', target_lineno=target_lineno, tokenizer=tokenizer)
    start_index = problem.start_index
    exit_index = problem.exit_index
    raise_index = exit_index + 1

    # 0: x = 1
    # 1: y = x + 2
    # 2: z = y * 3
    # 3: <exit>
    # 4: <raise>

    self.assertEqual(start_index, 0)
    self.assertEqual(exit_index, 3)
    self.assertEqual(raise_index, 4)
    self.assertEqual(problem.raise_nodes, [4, 4, 4, 4])
    self.assertEqual(problem.true_branch_nodes.tolist(), [1, 2, 3, 3])
    self.assertEqual(problem.false_branch_nodes.tolist(), [1, 2, 3, 3])

  def test_make_runtimeerrorproblem_nested_while_loops(self):
    tokenizer = tokenization.load_tokenizer()
    source = """
x = 1
while x < 2:
  y = 3
  while y < 4:
    y += 5
  x += 6
"""
    target_lineno = 0
    problem = process.make_runtimeerrorproblem(
        source, '1', target_lineno=target_lineno, tokenizer=tokenizer)
    start_index = problem.start_index
    exit_index = problem.exit_index
    raise_index = exit_index + 1

    # 0: x = 1
    # 1: x < 2  # while
    # 2:   y = 3
    # 3:   y < 4  # while
    # 4:     y += 5
    # 5:   x += 6
    # 6: <exit>
    # 7: <raise>

    self.assertEqual(start_index, 0)
    self.assertEqual(exit_index, 6)
    self.assertEqual(raise_index, 7)
    self.assertEqual(problem.raise_nodes, [7, 7, 7, 7, 7, 7, 7])
    self.assertEqual(problem.true_branch_nodes.tolist(), [1, 2, 3, 4, 3, 1, 6])
    self.assertEqual(problem.false_branch_nodes.tolist(), [1, 6, 3, 5, 3, 1, 6])

  def test_make_runtimeerrorproblem_try_finally(self):
    tokenizer = tokenization.load_tokenizer()
    source = """
header0
try:
  try0
  try1
except Exception0 as value0:
  exception0_stmt0
finally:
  finally_stmt0
  finally_stmt1
after0
"""
    target_lineno = 0
    raw = process.make_rawruntimeerrorproblem(
        source, '1', target_lineno=target_lineno)
    print_spans(raw)
    raise
    
    problem = process.make_runtimeerrorproblem(
        source, '1', target_lineno=target_lineno, tokenizer=tokenizer)
    start_index = problem.start_index
    exit_index = problem.exit_index
    raise_index = exit_index + 1

    # 0: header0
    #    try:
    # 3:   try0
    # 4:   try1
    # 5:        Exception0  # (comparing the exception)
    # 6: except Exception0 as value0: exception0_stmt0  # (assigning to value0)
    # 7:   exception0_stmt0
    #    finally:
    # 1:   finally_stmt0
    # 2:   finally_stmt1
    # 8: after0
    # 9: <exit>
    # 10: <raise>

    self.assertEqual(start_index, 0)
    self.assertEqual(exit_index, 9)
    self.assertEqual(raise_index, 10)
    self.assertEqual(problem.raise_nodes, [10, 10, 10, 5, 5, 1, 1, 1, 10, 10])
    self.assertEqual(problem.true_branch_nodes.tolist(), [3, 2, 10, 4, 1, 6, 7, 1, 9, 9])
    self.assertEqual(problem.false_branch_nodes.tolist(), [3, 2, 8, 4, 1, 1, 7, 1, 9, 9])

  def test_make_runtimeerrorproblem_try_finally_in_try_finally(self):
    tokenizer = tokenization.load_tokenizer()
    source = """
try:
  try:
    try0
  except Exception0 as value0:
    exception0_stmt0
  finally:
    finally_stmt0
finally:
  exception1_stmt0
after0
"""
    target_lineno = 0
    raw = process.make_rawruntimeerrorproblem(
        source, '1', target_lineno=target_lineno)
    problem = process.make_runtimeerrorproblem(
        source, '1', target_lineno=target_lineno, tokenizer=tokenizer)
    start_index = problem.start_index
    exit_index = problem.exit_index
    raise_index = exit_index + 1

    # Note: index 0 is not the start statement!

    #    try:
    #      try:
    # 2:     try0
    # 3:          Exception0
    # 4:   except Exception0 as value0: exception0_stmt0
    # 5:     exception0_stmt0
    #      finally:
    # 1:     finally_stmt0
    #    finally:
    # 0:   exception1_stmt0
    # 6: after0
    # 7: <exit>
    # 8: <raise>

    self.assertEqual(start_index, 2)
    self.assertEqual(exit_index, 7)
    self.assertEqual(raise_index, 8)
    self.assertEqual(problem.raise_nodes, [8, 0, 3, 1, 1, 1, 8, 8])
    self.assertEqual(problem.true_branch_nodes.tolist(), [8, 0, 1, 4, 5, 1, 7, 7])
    self.assertEqual(problem.false_branch_nodes.tolist(), [6, 0, 1, 1, 5, 1, 7, 7])

    # Note how "except as" spans contain the exception matching span.

    # Note how even without using any raise edges, the instruction pointer can still reach the
    # raise node (via the true branch on a finally block)
    # Q: Can a false branch also lead to the raise node? Maybe from an except block?
    # A: An outermost "except A" statement's false branch goes to raise. And it can be reached via
    # a finally inside the try.
    # The next test test_make_runtimeerrorproblem_try_finally_in_try_except
    # exhibits this.

  def test_make_runtimeerrorproblem_try_finally_in_try_except(self):
    tokenizer = tokenization.load_tokenizer()
    source = """
try:
  try:
    try0
  finally:
    finally_stmt0
except A:
  exception1_stmt0
after0
"""
    target_lineno = 0
    raw = process.make_rawruntimeerrorproblem(
        source, '1', target_lineno=target_lineno)
    problem = process.make_runtimeerrorproblem(
        source, '1', target_lineno=target_lineno, tokenizer=tokenizer)
    start_index = problem.start_index
    exit_index = problem.exit_index
    raise_index = exit_index + 1
    print_spans(raw)
    print(exit_index)
    print(raise_index)
    print(problem.raise_nodes)
    print(problem.true_branch_nodes.tolist())
    print(problem.false_branch_nodes.tolist())

    # Note: index 0 is not the start statement!

    #    try:
    #      try:
    # 1:     try0
    #      finally:
    # 0:     finally_stmt0
    # 2:        A
    # 3:   exception1_stmt0
    # 4: after0
    # 5: <exit>
    # 6: <raise>

    self.assertEqual(start_index, 1)
    self.assertEqual(exit_index, 5)
    self.assertEqual(raise_index, 6)
    self.assertEqual(problem.raise_nodes, [2, 0, 6, 6, 6, 6])
    self.assertEqual(problem.true_branch_nodes.tolist(), [2, 0, 3, 4, 5, 5])
    self.assertEqual(problem.false_branch_nodes.tolist(), [4, 0, 6, 4, 5, 5])

    # An outermost "except A" statement's false branch goes to raise. And it can be reached via
    # a finally inside the try.
    # Can only get into "raising" territory via a finally block's true branch or via a raise edge.

if __name__ == '__main__':
  unittest.main()
