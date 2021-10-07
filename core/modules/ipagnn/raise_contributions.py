import jax
import jax.numpy as jnp


def get_raise_contribution_at_step(instruction_pointer, raise_decisions, raise_index):
  # instruction_pointer.shape: num_nodes
  # raise_decisions.shape: num_nodes, 2
  # raise_index.shape: scalar.
  p_raise = raise_decisions[:, 0]
  raise_contribution = p_raise * instruction_pointer
  # raise_contribution.shape: num_nodes
  raise_contribution = raise_contribution.at[raise_index].set(0)
  return raise_contribution
get_raise_contribution_at_steps = jax.vmap(get_raise_contribution_at_step, in_axes=(0, 0, None))


def get_raise_contribution(instruction_pointer, raise_decisions, raise_index, step_limit):
  # instruction_pointer.shape: steps, num_nodes
  # raise_decisions.shape: steps, num_nodes, 2
  # raise_index.shape: scalar.
  # step_limit.shape: scalar.
  raise_contributions = get_raise_contribution_at_steps(
      instruction_pointer, raise_decisions, raise_index)
  # raise_contributions.shape: steps, num_nodes
  mask = jnp.arange(instruction_pointer.shape[0]) < step_limit
  # mask.shape: steps
  raise_contributions = jnp.where(mask[:, None], raise_contributions, 0)
  raise_contribution = jnp.sum(raise_contributions, axis=0)
  # raise_contribution.shape: num_nodes
  return raise_contribution
get_raise_contribution_batch = jax.vmap(get_raise_contribution)


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
  contributions = get_raise_contribution_batch(instruction_pointer, raise_decisions, raise_index, batch['step_limit'])
  # contributions.shape: batch_size, num_nodes
  return contributions
