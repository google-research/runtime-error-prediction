"""Tests for descriptions.py."""

import unittest

from core.data import descriptions
from core.data import example_problem_descriptions


class ProcessTest(unittest.TestCase):

  def test_extract_input_constraints(self):
    d = example_problem_descriptions.p00130
    c = descriptions.extract_input_constraints(d)
    self.assertEqual(c, 'No constraints')


if __name__ == '__main__':
  unittest.main()
