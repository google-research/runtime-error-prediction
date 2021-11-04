"""Tests for metrics.py."""

import unittest

import jax.numpy as jnp
import numpy as np

from core.data import info as info_lib
from core.lib import metrics


class MetricsTest(unittest.TestCase):

  def test_compute_binary_targets(self):
    info = info_lib.get_test_info()
    info.error_ids = [0]
    info.no_error_ids = [1, 2, 3]

    targets = np.array([3, 3, 0])
    predictions = np.array([0, 2, 0])
    logits = jnp.array([
        [5, 1, 2, 3],
        [0, 1, 3, 2],
        [5, 1, 1, 1],
    ])
    # logits.shape: num_eval_examples, num_classes
    binary_targets = metrics.compute_binary_targets(targets, info)
    np.testing.assert_array_equal(binary_targets, jnp.array([False, False, True]))

  def test_compute_binary_f1_score(self):
    info = info_lib.get_test_info()
    info.error_ids = [0]
    info.no_error_ids = [1, 2, 3]

    targets = np.array([0, 3, 0])
    predictions = np.array([0, 2, 0])
    logits = jnp.array([
        [5, 4, 4, 4],  # Multi-class prediction is correct, but binary prediction is wrong (predicts no error).
        [0, 1, 3, 2],  # Multi-class prediction is wrong, but binary prediction is right (no error).
        [5, 1, 1, 1],  # Both are correct (error).
    ])
    # logits.shape: num_eval_examples, num_classes

    f1_score = metrics.compute_binary_f1_score(targets, logits, info)
    self.assertEqual(f1_score, 2/3)


  def test_compute_binary_auc(self):
    info = info_lib.get_test_info()
    info.error_ids = [0]
    info.no_error_ids = [1, 2, 3]

    targets = np.array([0, 3, 0, 1])
    predictions = np.array([0, 2, 0, 0])
    logits = jnp.array([
        [5, 4, 4, 4],  # Multi-class prediction is correct, but binary prediction is wrong (predicts no-error).
        [0, 1, 3, 2],  # Multi-class prediction is wrong, but binary prediction is right (no error).
        [5, 1, 1, 1],  # Both are correct (error).
        [5, 1, 1, 1],  # Both are wrong (predicts no-error).
    ])
    # logits.shape: num_eval_examples, num_classes

    auc = metrics.compute_binary_auc(targets, logits, info)
    self.assertEqual(auc, 0.625)

  def test_compute_recall_at_precision(self):
    info = info_lib.get_test_info()
    info.error_ids = [0]
    info.no_error_ids = [1, 2, 3]

    targets = np.array([0, 3, 0, 1])
    predictions = np.array([0, 2, 0, 0])
    logits = jnp.array([
        [5, 4, 4, 4],  # Multi-class prediction is correct, but binary prediction is wrong (predicts no-error).
        [0, 1, 3, 2],  # Multi-class prediction is wrong, but binary prediction is right (no error).
        [5, 1, 1, 1],  # Both are correct (error).
        [5, 2, 1, 1],  # Both are wrong (predicts no-error).
    ])
    # logits.shape: num_eval_examples, num_classes

    recall_at_050 = metrics.compute_recall_at_precision(targets, logits, info, target_precision=0.50)
    self.assertEqual(recall_at_050, 1)
    recall_at_090 = metrics.compute_recall_at_precision(targets, logits, info, target_precision=0.90)
    self.assertEqual(recall_at_090, 0.5)

  def test_compute_precision_at_recall(self):
    info = info_lib.get_test_info()
    info.error_ids = [0]
    info.no_error_ids = [1, 2, 3]

    targets = np.array([0, 3, 0, 1])
    predictions = np.array([0, 2, 0, 0])
    logits = jnp.array([
        [5, 4, 4, 4],  # Multi-class prediction is correct, but binary prediction is wrong (predicts no-error).
        [0, 1, 3, 2],  # Multi-class prediction is wrong, but binary prediction is right (no error).
        [5, 1, 1, 1],  # Both are correct (error).
        [5, 2, 1, 1],  # Both are wrong (predicts no-error).
    ])
    # logits.shape: num_eval_examples, num_classes

    precision_at_050 = metrics.compute_precision_at_recall(targets, logits, info, target_recall=0.50)
    self.assertEqual(precision_at_050, 1)
    precision_at_090 = metrics.compute_precision_at_recall(targets, logits, info, target_recall=0.90)
    self.assertEqual(precision_at_090, 2/3)

if __name__ == '__main__':
  unittest.main()
