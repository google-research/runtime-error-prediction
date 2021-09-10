"""Optimizer utility functions."""

from flax import optim
import jax
import jax.numpy as jnp


def compute_global_norm(grads):
  return jnp.sqrt(
      sum([
          jnp.sum(jnp.square(x))
          for x in jax.tree_util.tree_leaves(grads)
      ])
  )


def clip_grads(grads, clip_by, clip_value):
  """Clips the gradient using the method clip_by."""
  if clip_by == 'global_norm':
    global_norm = compute_global_norm(grads)
    should_clip = global_norm > clip_value
    grads = jax.tree_map(
        lambda g: jnp.where(should_clip, g * clip_value / global_norm, g),
        grads
    )
  else:
    raise ValueError('Unexpected value for clip_by', clip_by)
  return grads
