"""Tests for process.py."""

import unittest

from core.data import process
from core.data import tokenize


class ProcessTest(unittest.TestCase):

  def setUp(self):
    tokenize.generate_tokenizer()

  def test_make_runtimeerrorproblem(self):
    source = """n = input()
print(any(set('47') >= set(str(i)) and n % i == 0 for i in range(1, n+1)) and 'YES' or 'NO')
"""
    problem = process.make_runtimeerrorproblem(source, '1')
    target_tokens = [
        66, 31, 99, 84, 125, 10, 377, 10, 214, 223, 315, 222, 237, 214, 10,
        166, 10, 61, 95, 148, 66, 8, 61, 79, 18, 89, 61, 80, 104, 10, 19, 14,
        66, 13, 19, 95, 148, 9, 240, 9, 82, 9, 239, 222]
    self.assertEqual(problem.tokens, target_tokens)


if __name__ == '__main__':
  unittest.main()
