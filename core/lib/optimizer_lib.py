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

"""Optimizer utility functions."""

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
