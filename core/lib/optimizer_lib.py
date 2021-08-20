"""Optimizer utility functions."""

from flax import optim
import jax
import jax.numpy as jnp


def clip_grad(grad, clip_by, clip_value):
  """Clips the gradient using the method clip_by."""
  if clip_by == 'global_norm':
    global_norm = jnp.sqrt(
        sum([jnp.sum(jnp.square(x)) for x in jax.tree_util.tree_leaves(grad)]))
    should_clip = global_norm > clip_value
    grad = jax.tree_map(
        lambda g: jnp.where(should_clip, g * clip_value / global_norm, g),
        grad
    )
  else:
    raise ValueError('Unexpected value for clip_by', clip_by)
  return grad
