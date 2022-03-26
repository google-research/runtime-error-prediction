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

from typing import Any, List

from flax import linen as nn
import jax


def create_lstm_cells(n):
  """Creates a list of n LSTM cells."""
  cells = []
  for i in range(n):
    cell = nn.LSTMCell(
        gate_fn=nn.sigmoid,
        activation_fn=nn.tanh,
        kernel_init=nn.initializers.xavier_uniform(),
        recurrent_kernel_init=nn.initializers.orthogonal(),
        bias_init=nn.initializers.zeros,
        name=f'lstm_{i}',
    )
    cells.append(cell)
  return cells


class StackedRNNCell(nn.Module):
  """Stacked RNN Cell."""

  cells: List[Any]

  def __call__(self, carry, inputs):
    new_carry = []
    for c, cell in zip(carry, self.cells):
      new_c, inputs = cell(c, inputs)
      new_carry.append(new_c)
    return new_carry, inputs

  def initialize_carry(self, rng, batch_dims, size,
                       init_fn=nn.initializers.zeros):
    cells = self.cells
    keys = jax.random.split(rng, len(cells))
    return [
        cell.initialize_carry(key, batch_dims, size, init_fn=init_fn)
        for key, cell in zip(keys, cells)
    ]
