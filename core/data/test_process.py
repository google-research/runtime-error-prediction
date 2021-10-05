"""Tests for process.py."""

import unittest

from core.data import process
from core.data import tokenization


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

    # NOTE(dbieber): We are sending the true and false branches of a raise node
    # to itself. We may wish to change this behavior.
    self.assertEqual(problem.true_branch_nodes.tolist(), [1, 2, 2, 4, 4])
    self.assertEqual(problem.false_branch_nodes.tolist(), [4, 2, 2, 5, 4])

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
    exit_index = problem.exit_index
    raise_index = exit_index + 1

    # 0: x = 1
    # 1: y = x + 2
    # 2: z = y * 3
    # 3: <exit>
    # 4: <raise>

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

    self.assertEqual(exit_index, 6)
    self.assertEqual(raise_index, 7)
    self.assertEqual(problem.raise_nodes, [7, 7, 7, 7, 7, 7, 7])
    self.assertEqual(problem.true_branch_nodes.tolist(), [1, 2, 3, 4, 3, 1, 6])
    self.assertEqual(problem.false_branch_nodes.tolist(), [1, 6, 3, 5, 3, 1, 6])

if __name__ == '__main__':
  unittest.main()
