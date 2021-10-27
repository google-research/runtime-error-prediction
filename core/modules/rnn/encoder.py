from typing import Callable, Any, Optional

import jax.numpy as jnp
from flax import linen as nn

from core.modules.rnn import lstm


class LSTMEncoder(nn.Module):
  """LSTM Model Encoder for sequence to sequence translation.

  This Encoder does not encode the input
  tokens itself. It assumes the tokens have already been encoded, and any
  desired positional embeddings have already been aded.
  """
  
  dropout_rate: float
  num_layers: int
  hidden_dim: int
  deterministic: bool = True
  dtype: Any = jnp.float32

  @nn.compact
  def __call__(self,
               encoded_inputs,):
    """Applies Transformer model on the encoded inputs.

    Args:
      encoded_inputs: pre-encoded input data.

    Returns:
      output of a lstm encoder.
    """
    batch_size, max_tokens, _ = encoded_inputs.shape

    x = encoded_inputs
    x = nn.Dropout(rate=self.dropout_rate)(x, deterministic=self.deterministic)
    x = x.astype(self.dtype)

    # Input Encoder
    for layer_num in range(self.num_layers):
      initial_state = lstm.SimpleLSTM.initialize_carry((batch_size,), self.hidden_dim)
      _, x = lstm.SimpleLSTM(name=f"lstm_{layer_num}")(initial_state, x) # TODO(rgoel): Add layer norm to each layer

    encoded = nn.LayerNorm(dtype=self.dtype, name='encoder_norm')(x)

    return encoded
