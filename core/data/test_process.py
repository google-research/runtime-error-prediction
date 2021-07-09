"""Tests for process.py."""

import unittest

from core.data import process


class ProcessTest(unittest.TestCase):

  def test_make_runtimeerrorproblem(self):
    source = """n = input()
print(any(set('47') >= set(str(i)) and n % i == 0 for i in range(1, n+1)) and 'YES' or 'NO')
"""
    problem = process.make_runtimeerrorproblem(source, '1')
    target_tokens = [
        67, 31, 100, 85, 126, 10, 382, 10, 212, 222, 322, 221, 236, 212, 10,
        167, 10, 62, 96, 149, 67, 8, 62, 80, 18, 90, 62, 81, 105, 10, 19, 14,
        67, 13, 19, 96, 149, 9, 240, 9, 83, 9, 238, 221]
    self.assertEqual(problem.tokens, target_tokens)


if __name__ == '__main__':
  unittest.main()
