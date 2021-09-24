"""Toy MLP model."""

from typing import Any

from flax import linen as nn
import jax.numpy as jnp


class MlpModel(nn.Module):

  info: Any

  @nn.compact
  def __call__(self, x):
    x = x['tokens']
    # x.shape: batch_size, length
    batch_size = x.shape[0]
    x = nn.Embed(num_embeddings=30000, features=128)(x[:, :30])
    # x.shape: batch_size, 30, 128
    x = nn.Dense(features=30)(x)
    # x.shape: batch_size, 30, 30
    x = jnp.reshape(x, (batch_size, -1))
    # x.shape: batch_size, 900
    x = nn.relu(x)
    x = nn.Dense(features=30)(x)
    # x.shape: batch_size, 30
    x = nn.relu(x)
    x = nn.Dense(features=self.info.num_classes)(x)
    # x.shape: batch_size, num_classes
    return x, None
