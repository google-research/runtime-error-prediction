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


if __name__ == '__main__':
  unittest.main()
