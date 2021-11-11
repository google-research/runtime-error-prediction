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
    # print('prev_raise_attributions_to_m.shape')
    # print(prev_raise_attributions_to_m.shape)
    # print('instruction_pointer')
    # print(instruction_pointer)

    # (1) sum v from branch decisions
    # Goal: By 4pm, calculate (1).
    denom = (
        jax.ops.segment_sum(
            p_true * instruction_pointer, true_indexes,
            num_segments=num_nodes)
        + jax.ops.segment_sum(
            p_false * instruction_pointer, false_indexes,
            num_segments=num_nodes)
    )
    no_raise_contributions = (
        jax.ops.segment_sum(
            prev_raise_attributions_to_m * p_true * instruction_pointer, true_indexes,
            num_segments=num_nodes)
        + jax.ops.segment_sum(
            prev_raise_attributions_to_m * p_false * instruction_pointer, false_indexes,
            num_segments=num_nodes)
    ) / (denom + 1e-12)
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
  # print('(5) attribution')
  # print(attribution)

  return attribution


def get_raise_contributions(
    instruction_pointer,
    branch_decisions,
    raise_decisions,
    true_indexes,
    false_indexes,
    raise_indexes,
    raise_index,
    step_limit):
  # instruction_pointer.shape: steps, num_nodes
  # branch_decisions.shape: steps, num_nodes, 2
  # raise_decisions.shape: steps, num_nodes, 2
  # true_indexes.shape: num_nodes
  _, num_nodes = instruction_pointer.shape
  raise_contributions = jnp.zeros((num_nodes, num_nodes))
  for t in range(step_limit):
    raise_contributions = get_raise_contribution_step(
        raise_contributions,
        instruction_pointer[t],  # per-step
        branch_decisions[t],  # per-step
        raise_decisions[t],  # per-step
        true_indexes,
        false_indexes,
        raise_indexes,
        num_nodes
    )
  # raise_contributions.shape:
  #   num_nodes (n = current node), 
  #   num_nodes (m = node raise is attributed to)
  print('raise_contributions')
  print(raise_contributions)
  raise_contributions = raise_contributions[raise_index, :]
  # raise_contributions.shape: num_nodes (m)
  return raise_contributions
get_raise_contribution_batch = jax.vmap(get_raise_contributions)


def get_raise_contribution_from_batch_and_aux(batch, aux):
  instruction_pointer = aux['instruction_pointer_orig']
  # instruction_pointer.shape: steps, batch_size, num_nodes
  instruction_pointer = jnp.transpose(instruction_pointer, [1, 0, 2])
  # instruction_pointer.shape: batch_size, steps, num_nodes
  exit_index = batch['exit_index']
  raise_index = exit_index + 1
  raise_decisions = aux['raise_decisions']
  # raise_decisions.shape: steps, batch_size, num_nodes, 2
  raise_decisions = jnp.transpose(raise_decisions, [1, 0, 2, 3])
  # raise_decisions.shape: batch_size, steps, num_nodes, 2

  branch_decisions = aux['branch_decisions']
  # branch_decisions.shape: steps, batch_size, num_nodes, 2
  branch_decisions = jnp.transpose(branch_decisions, [1, 0, 2, 3])
  # branch_decisions.shape: batch_size, steps, num_nodes, 2
  print('branch_decisions.shape')
  print(branch_decisions.shape)
  true_indexes = batch['true_branch_nodes']
  false_indexes = batch['false_branch_nodes']
  raise_indexes = batch['raise_nodes']
  print('true_indexes.shape')
  print(true_indexes.shape)
  step_limit = batch['step_limit']

  contributions = get_raise_contribution_batch(
      instruction_pointer,
      branch_decisions,
      raise_decisions,
      true_indexes,
      false_indexes,
      raise_indexes,
      raise_index,
      step_limit)

  # contributions.shape: batch_size, num_nodes
  return contributions


#  OLD try/except unaware implementation follows:


def get_raise_contribution_at_step_old(instruction_pointer, raise_decisions, raise_index):
  # instruction_pointer.shape: num_nodes
  # raise_decisions.shape: num_nodes, 2
  # raise_index.shape: scalar.
  p_raise = raise_decisions[:, 0]
  raise_contribution = p_raise * instruction_pointer
  # raise_contribution.shape: num_nodes
  raise_contribution = raise_contribution.at[raise_index].set(0)
  return raise_contribution
get_raise_contribution_at_steps_old = jax.vmap(get_raise_contribution_at_step_old, in_axes=(0, 0, None))


def get_raise_contribution_old(instruction_pointer, raise_decisions, raise_index, step_limit):
  # instruction_pointer.shape: steps, num_nodes
  # raise_decisions.shape: steps, num_nodes, 2
  # raise_index.shape: scalar.
  # step_limit.shape: scalar.
  raise_contributions = get_raise_contribution_at_steps_old(
      instruction_pointer, raise_decisions, raise_index)
  # raise_contributions.shape: steps, num_nodes
  mask = jnp.arange(instruction_pointer.shape[0]) < step_limit
  # mask.shape: steps
  raise_contributions = jnp.where(mask[:, None], raise_contributions, 0)
  raise_contribution = jnp.sum(raise_contributions, axis=0)
  # raise_contribution.shape: num_nodes
  return raise_contribution
get_raise_contribution_batch_old = jax.vmap(get_raise_contribution_old)

