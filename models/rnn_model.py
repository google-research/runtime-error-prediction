import functools
from flax import linen as nn
import jax
import jax.numpy as jnp

from core.data import error_kinds

NUM_CLASSES = error_kinds.NUM_CLASSES
PAD_ID = -1

class SimpleLSTM(nn.Module):
  """A simple unidirectional LSTM."""

  @functools.partial(
      nn.transforms.scan,
      variable_broadcast='params',
      in_axes=1, out_axes=1,
      split_rngs={'params': False})
  @nn.compact
  def __call__(self, carry, x):
    return nn.OptimizedLSTMCell()(carry, x)

  @staticmethod
  def initialize_carry(batch_dims, hidden_size):
    # Use fixed random key since default state init fn is just zeros.
    return nn.OptimizedLSTMCell.initialize_carry(
        jax.random.PRNGKey(0), batch_dims, hidden_size)

class StackedLSTMModel(nn.Module):
  """Applies an LSTM to form statement embeddings, and again for program embs.
    Following tokenization, each single example in the dataset can be a dict with the following keys:
    - tokens: A list of ints representing the source of the program
    - edge_sources, edge_dests, edge_types: As above
    - node_token_span_starts: A list of the token span start for each node in the program's graph representation.
    - node_token_span_ends: A list of the token span ends for each node in the program's graph representation.
  """
  vocab_size: int = 32
  emb_dim: int = 32
  def setup(self):
    # Embedding setup
    self.embed = nn.Embed(num_embeddings=self.vocab_size, features=self.emb_dim, name='embed')
    # LSTM setup
    self.lstm = SimpleLSTM()
    self.logits = nn.Dense(features=NUM_CLASSES)


  @nn.compact
  def __call__(self, example_inputs,
            config,):

    inputs = example_inputs['tokens']
    node_token_span_starts = example_inputs['node_token_span_starts']
    node_token_span_ends = example_inputs['node_token_span_ends']
    
    emb_dim = config.model.hidden_size
    hidden_size = config.model.hidden_size
    batch_size = config.dataset.batch_size
    assert inputs.ndim == 2  # (batch_size, length)

    x = inputs.astype('int32')

    program_line_embeddings = self.embed(x)

    def get_program_lens(node_token_span_ends):
      # Take the first occurence of PAD_ID which is the
      # length of the program.
      return jnp.argwhere(node_token_span_ends==PAD_ID, size=1)[0][0]-1

    # There is a bug in the following line.
    initial_state = SimpleLSTM.initialize_carry((batch_size,), hidden_size)
    (c,h), forward_outputs = self.lstm(initial_state, program_line_embeddings)    
    output = self.logits(forward_outputs[:, -1, :])
    return output
