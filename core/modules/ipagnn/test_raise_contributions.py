# Copyright (C) 2021 Google Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for logit_math.py."""

import jax
import jax.numpy as jnp
import numpy as np
import unittest

from core.modules.ipagnn import raise_contributions as raise_contributions_lib


class RaiseContributionTest(unittest.TestCase):

  def test_get_raise_contribution_step_two(self):
    prev_raise_attributions = jnp.array([
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0.2, 0, 0, 0],
    ])
    instruction_pointer = jnp.array([0, .8, 0, 0, .2])
    branch_decisions = jnp.array([
        [.5, .5], [.5, .5], [.5, .5], [.5, .5], [.5, .5]
    ])
    raise_decisions = jnp.array([
        [0, 1], [0, 1], [0, 1], [0, 1], [0, 1],
    ])
    true_indexes = jnp.array([1, 2, 3, 3, 4])
    false_indexes = jnp.array([1, 2, 3, 3, 4])
    raise_indexes = jnp.array([4, 4, 4, 4, 4])
    exit_index = jnp.array(3)
    raise_index = exit_index + 1
    num_nodes = 5

    raise_contributions = raise_contributions_lib.get_raise_contribution_step(
        prev_raise_attributions,
        instruction_pointer,
        branch_decisions,
        raise_decisions,
        true_indexes,
        false_indexes,
        raise_indexes,
        num_nodes,
    )
    np.testing.assert_allclose(
        raise_contributions,
        jnp.array([
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0.2, 0, 0, 0],
        ])
    )

  def test_get_raise_contribution_step_one(self):
    prev_raise_attributions = jnp.array([
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
    ])
    instruction_pointer = jnp.array([1, 0, 0, 0, 0])
    branch_decisions = jnp.array([
        [.5, .5], [.5, .5], [.5, .5], [.5, .5], [.5, .5]
    ])
    raise_decisions = jnp.array([
        [.8, .2], [0, 1], [0, 1], [0, 1], [0, 1],
    ])
    true_indexes = jnp.array([1, 2, 3, 3, 4])
    false_indexes = jnp.array([1, 2, 3, 3, 4])
    raise_indexes = jnp.array([4, 4, 4, 4, 4])
    exit_index = jnp.array(3)
    raise_index = exit_index + 1
    num_nodes = 5

    raise_contributions = raise_contributions_lib.get_raise_contribution_step(
        prev_raise_attributions,
        instruction_pointer,
        branch_decisions,
        raise_decisions,
        true_indexes,
        false_indexes,
        raise_indexes,
        num_nodes,
    )
    np.testing.assert_allclose(
        raise_contributions,
        jnp.array([
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0.8, 0, 0, 0, 0],
        ])
    )


if __name__ == '__main__':
  unittest.main()
