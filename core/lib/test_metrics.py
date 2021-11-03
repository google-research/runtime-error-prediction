"""Tests for metrics.py."""

import unittest

from core.data import info as info_lib
from core.lib import metrics


class MetricsTest(unittest.TestCase):

  def test_compute_binary_f1_score(self):
    info = info_lib.get_test_info()
    targets = np.array([
        [0, 1, 2, 3],
        [0, 1, 2, 3],
    ])
    metrics.compute_binary_f1_score(
        targets, predictions, logits, info)

if __name__ == '__main__':
  unittest.main()
