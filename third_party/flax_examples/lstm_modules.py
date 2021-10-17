from typing import Callable, Any, Optional

from flax import linen as nn
from flax import struct
from jax import lax
import jax.numpy as jnp
import numpy as np

@struct.dataclass
class LSTMConfig:
  """Global hyperparameters used to minimize obnoxious kwarg plumbing."""
  vocab_size: int
  output_vocab_size: int
  share_embeddings: bool = False
  logits_via_embedding: bool = False
  dtype: Any = jnp.float32
  emb_dim: int = 512
  num_layers: int = 1
  hidden_dim: int = 2048
  max_len: int = 2048
  dropout_rate: float = 0.1
  deterministic: bool = False
  decode: bool = False
  kernel_init: Callable = nn.initializers.xavier_uniform()
  bias_init: Callable = nn.initializers.normal(stddev=1e-6)
  posemb_init: Optional[Callable] = None