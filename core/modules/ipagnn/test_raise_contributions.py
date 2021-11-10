"""Tests for logit_math.py."""

import jax
import jax.numpy as jnp
import numpy as np
import unittest

from core.modules.ipagnn import raise_contributions


class RaiseContributionTest(unittest.TestCase):

  def test_get_additional_logit(self):
    instruction_pointer = jnp.array([
        [1,  0,  0,  0,  0],
        [0, .8,  0,  0, .2],
        [0,  0, .8,  0, .2],
        [0,  0,  0, .4, .6],
    ])
    # instruction_pointer.shape: steps (4), num_nodes (5)
    branch_decisions = jnp.array([
        [[.5, .5], [.5, .5], [.5, .5], [.5, .5], [.5, .5]],
        [[.5, .5], [.5, .5], [.5, .5], [.5, .5], [.5, .5]],
        [[.5, .5], [.5, .5], [.5, .5], [.5, .5], [.5, .5]],
        [[.5, .5], [.5, .5], [.5, .5], [.5, .5], [.5, .5]],
    ])
    raise_decisions = jnp.array([
        [[0.2, 0.8], [0, 1], [0, 1], [0, 1], [0, 1]],
        [[0, 1], [0, 1], [0, 1], [0, 1], [0, 1]],
        [[0, 1], [0, 1], [0.5, 0.5], [0, 1], [0, 1]],
        [[0, 1], [0, 1], [0, 1], [0, 1], [0, 1]],
    ])
    true_indexes = jnp.array([1, 2, 3, 3, 4])
    false_indexes = jnp.array([1, 2, 3, 3, 4])
    raise_indexes = jnp.array([4, 4, 4, 4, 4])
    exit_index = jnp.array(3)
    raise_index = exit_index + 1
    step_limit = 3
    contributions = raise_contributions.get_raise_contributions(
        instruction_pointer,
        branch_decisions,
        raise_decisions,
        true_indexes,
        false_indexes,
        raise_indexes,
        raise_index,
        step_limit
    )
    np.testing.assert_array_equal(contributions,
        jnp.array([.1, 0, .2, 0, 0]))



if __name__ == '__main__':
  unittest.main()
