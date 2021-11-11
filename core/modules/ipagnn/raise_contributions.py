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
  Define $v_{t,n,m}$
  = Amount of "exception mass"
  at n attributable to
  an exception starting at m at time step t.

  We are interested in computing $v_{T,n_{raise},m}$
  for all m, where T is the final time step.

  (0) v_{t,n,m} =
    sum v from branch decisions
    + sum v from raise decisions with attribution
    + sum v from raise decisions without attribution

  (1) sum v from branch decisions
    = \sum_{n' \in \Nin(n)} v_{t,n',m} * b_{t, n', n} * p_{t, n'}

  b_{t, n', n} is the amount of branch decision from n' to n at step t.

  (2) sum v from raise decisions with attribution
    = \sum_{n' \in \Nin(n)} v_{t,n',m} * r_{t, n', n} * p{t, n'}

  r_{t, n', n} is the amount of prob mass raised from n' to n at step t.

  (3) sum v from raise decisions without attribution
    = (1 - \sum v_{t,m,:}) * r_{t, n', n} * p{t, n'}
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
    # print('(1) no_raise_contributions')
    # print(no_raise_contributions)

    # # (2) sum v from raise decisions with attribution
    # attributed_raise_contributions = jax.ops.segment_sum(
    #     prev_raise_attributions_to_m * p_raise * instruction_pointer, raise_indexes,
    #     num_segments=num_nodes)
    # # attributed_raise_contributions.shape: num_nodes

    # print('(2) attributed_raise_contributions')
    # print(attributed_raise_contributions)

    # # (3) sum v from raise decisions without attribution
    # unattributed_value = 1 - jnp.sum(prev_raise_attributions_to_m)
    # print('(3) unattributed_value')
    # print(unattributed_value)
    # # unattributed_value.shape: scalar.
    # unattributed_raise_contributions = jax.ops.segment_sum(
    #     unattributed_value * p_raise * instruction_pointer, raise_indexes,
    #     num_segments=num_nodes)
    # # unattributed_raise_contributions.shape: num_nodes

    # print('(3) unattributed_raise_contributions')
    # print(unattributed_raise_contributions)

    return (
        no_raise_contributions
        # + attributed_raise_contributions
        # + unattributed_raise_contributions
    )
  get_attribution_all_m = jax.vmap(get_attribution, in_axes=1, out_axes=1)

  # prev_raise_attributions.shape: num_nodes (n), num_nodes (m)
  # print('prev_raise_attributions')
  # print(prev_raise_attributions)
  attribution = get_attribution_all_m(prev_raise_attributions)
  # attribution.shape:
  #   num_nodes (n = current node), 
  #   num_nodes (m = node raise is attributed to)

  # instruction_pointer[m] is total p from m
  # p_raise[m] indicates amount raising from m to n=raise_index[m]
  unattributed_amounts = 1  # 1 - jnp.sum(prev_raise_attributions, axis=1)
  values_contributed = unattributed_amounts * instruction_pointer * p_raise
  attribution = (
      attribution.at[raise_indexes, jnp.arange(num_nodes)]
      .add(values_contributed)
  )

  return attribution
get_raise_contribution_step_batch = jax.vmap(
    get_raise_contribution_step,
    in_axes=(0,) * 7 + (None,))
