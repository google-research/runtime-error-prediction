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

import jax
import jax.numpy as jnp


def get_raise_contribution_step(
    prev_raise_attributions,
    instruction_pointer,
    branch_decisions,
    raise_decisions,
    true_indexes,
    false_indexes,
    raise_indexes,
    num_nodes):
  r"""
  Define $v_{t,n,m}$ = Amount of "exception probability mass" at time step t
  at n attributable to an exception starting at m.

  We are interested in computing $v_{T,n_{raise},m}$
  for all m, where T is the final time step.

  (0) v_{t,n,m} =
    sum v from branch decisions + sum v from raise decisions

  (1) sum v from branch decisions
    = \sum_{n' \in \Nin(n)} v_{t,n',m} * b_{t, n', n} * p_{t, n'}

  b_{t, n', n} is the amount of branch decision from n' to n at step t.

  (2) sum v from raise decisions
    = (1 - \sum v_{t,m,:}) * r_{t, n', n} * p{t, n'}

  r_{t, n', n} is the amount of prob mass raised from n' to n at step t.
  """
  # instruction_pointer.shape: num_nodes
  # raise_decisions.shape: num_nodes, 2
  # branch_decisions.shape: num_nodes, 2
  p_raise = raise_decisions[:, 0]
  p_noraise = raise_decisions[:, 1]
  # p_raise.shape: num_nodes
  # p_noraise.shape: num_nodes
  p_true = p_noraise * branch_decisions[:, 0]
  p_false = p_noraise * branch_decisions[:, 1]
  # p_true.shape: num_nodes
  # p_false.shape: num_nodes

  def get_attribution(prev_raise_attributions_to_m):
    """Returns for each n how much to attribute to the passed in m.

    Args:
      prev_raise_attributions_to_m: For each n, how much to attribute to m as of
        the previous step.
    Returns:
      For each n, now how much to attribute to m.
    """
    # prev_raise_attributions_to_m.shape: num_nodes

    # (1) sum v from branch decisions
    denom = (
        jax.ops.segment_sum(
            p_true * instruction_pointer, true_indexes,
            num_segments=num_nodes)
        + jax.ops.segment_sum(
            p_false * instruction_pointer, false_indexes,
            num_segments=num_nodes)
    ) + 1e-12
    no_raise_contributions = (
        jax.ops.segment_sum(
            prev_raise_attributions_to_m * p_true * instruction_pointer, true_indexes,
            num_segments=num_nodes)
        + jax.ops.segment_sum(
            prev_raise_attributions_to_m * p_false * instruction_pointer, false_indexes,
            num_segments=num_nodes)
    ) / denom
    # no_raise_contributions.shape: num_nodes

    return (
        no_raise_contributions
        # + attributed_raise_contributions
        # + unattributed_raise_contributions
    )
  get_attribution_all_m = jax.vmap(get_attribution, in_axes=1, out_axes=1)

  # prev_raise_attributions.shape: num_nodes (n), num_nodes (m)
  attribution = get_attribution_all_m(prev_raise_attributions)
  # attribution.shape:
  #   num_nodes (n = current node), 
  #   num_nodes (m = node raise is attributed to)

  # (2) sum v from raise decisions
  # instruction_pointer[m] is total p from m
  # p_raise[m] indicates amount raising from m to n=raise_index[m]
  values_contributed = instruction_pointer * p_raise
  attribution = (
      attribution.at[raise_indexes, jnp.arange(num_nodes)]
      .add(values_contributed)
  )
  return attribution
get_raise_contribution_step_batch = jax.vmap(
    get_raise_contribution_step,
    in_axes=(0,) * 7 + (None,))
