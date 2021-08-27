"""Tests for logit_math.py."""

import jax.numpy as jnp
import unittest

from core.modules.ipagnn import logit_math


class LogitMathTest(unittest.TestCase):

  def test_get_additional_logit(self):
    logits = jnp.array([2, 2, 2, 2])
    target_p = 0.2
    additional_logit = logit_math.get_additional_logit(target_p, logits)
    self.assertEqual(additional_logit, 2)

    logits = jnp.array([-jnp.inf, 3, 3, 3])
    target_p = 0.25
    additional_logit = logit_math.get_additional_logit(target_p, logits)
    self.assertEqual(additional_logit, 3)

    logits = jnp.array([-jnp.inf, 1, 2, 3])
    target_p = 1
    additional_logit = logit_math.get_additional_logit(target_p, logits)
    self.assertEqual(additional_logit, jnp.inf)

    logits = jnp.array([-jnp.inf, -1, -2, -3])
    target_p = 0
    additional_logit = logit_math.get_additional_logit(target_p, logits)
    self.assertEqual(additional_logit, -jnp.inf)

    logits = jnp.array([-jnp.inf, -jnp.inf, jnp.log(2), jnp.log(3), jnp.log(5)])
    target_p = 2/3
    additional_logit = logit_math.get_additional_logit(target_p, logits)
    self.assertEqual(additional_logit, jnp.log(20))


if __name__ == '__main__':
  unittest.main()
