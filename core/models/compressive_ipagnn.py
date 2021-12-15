"""IPA-GNN models."""

from typing import Any

from flax import linen as nn
import jax
import jax.numpy as jnp
from jax import lax

from core.data import error_kinds
from core.modules.ipagnn import encoder
from core.modules.ipagnn import logit_math
from core.modules.ipagnn import spans
from core.modules.ipagnn import raise_contributions as raise_contributions_lib
from core.modules.ipagnn import rnn
from third_party.flax_examples import transformer_modules

StackedRNNCell = rnn.StackedRNNCell


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


def make_concat(h):
  """Creates concat and unconcat functions for the hidden state.

  This function assumes that the components of h are all the same shape.

  Args:
    h: The RNN state to create the concat and unconcat functions for.
  Returns:
    concat: Accepts a hidden state (a pytree of ndarrays) and returns a single
      ndarray representing the whole state.
    unconcat: The inverse of concat. Accepts a single ndarray and splits it into
      the structure used by the RNN hidden states.
  """
  h_leaves, h_treedef = jax.tree_flatten(h)
  h_num_components = len(h_leaves)
  del h_leaves  # Unused.

  def concat(h):
    return jnp.concatenate(jax.tree_leaves(h), axis=-1)

  def unconcat(h_concat):
    h_leaves = jnp.split(h_concat, h_num_components, axis=-1)
    return jax.tree_unflatten(h_treedef, h_leaves)

  return concat, unconcat


class SkipEncoder(nn.Module):
  """Encoder layer for the compressed edges.
  
  TODO(dbieber): Rename to CompressedEdgesEncoder.
  """

  config: Any

  @nn.compact
  def __call__(self, statement_embeddings_nh):
    """Creates skip embeddings for a single example.

    The skip embedding from node i to node j is an RNN run over the code from
    statement i to statement j-1 inclusive.

    Args:
      statement_embeddings_nh: A single example's statement embeddings. Shape
        is (num_statements, hidden_size).
      config: The experiment's Config object.
    Returns:
      The skip embeddings for all pairs of statements in the example. Shape is
        (num_statements, num_statements, hidden_size). Axis order is
        val[from, to, d].
    """
    config = self.config
    hidden_size = config.model.hidden_size
    skip_embedder_layer_norm = nn.LayerNorm(
        name='skip_embedder_layer_norm')
    num_statements = statement_embeddings_nh.shape[0]
    num_nodes = num_statements + 1
    cells = create_lstm_cells(config.model.rnn_cell.layers)
    lstm = StackedRNNCell(cells=cells, name='skip_encoder_rnn')

    initial_state = lstm.initialize_carry(
        jax.random.PRNGKey(0), cells, (), hidden_size)
    default_result = jnp.zeros((hidden_size,))

    concat, unconcat = make_concat(initial_state)

    def create_skip_embeddings_from(start_index):
      """Creates skip embeddings for one start-statement for a single example.
      Args:
        start_index: (int) The index of the start statement to produce the
          embeddings from.
      Returns:
        The skip-embeddings going from the specified start statement index to
        all other statements in the code. Shape is
        (num_statements, hidden_size).
      """
      initial_carry = initial_state, 0

      def f(carry, statement_embedding):
        # statement_embedding.shape: hidden_size
        state, index = carry
        state, result = lstm(state, statement_embedding)
        state = unconcat(skip_embedder_layer_norm(concat(state)))
        state, result = jax.tree_multimap(
            lambda v, default: jnp.where(index >= start_index, v, default),
            (state, result),
            (initial_state, default_result)
        )
        # result.shape: hidden_size
        carry = state, index + 1
        return carry, result

      unused_carry, results = lax.scan(
          f, initial_carry, statement_embeddings_nh, length=num_statements)
      # results.shape: num_statements, hidden_size
      results = jnp.concatenate(
          (jnp.expand_dims(default_result, axis=0), results),
          axis=0)
      # results.shape: num_nodes, hidden_size
      return results

    skip_embedder = jax.vmap(create_skip_embeddings_from)
    results = skip_embedder(jnp.arange(num_nodes))
    # results.shape: num_nodes, num_nodes, hidden_size
    return results



class SkipEncoderModel(nn.Module):
  """Skip Encoder model."""

  config: Any
  info: Any

  @nn.compact
  def __call__(
      self,
      node_embeddings,
      docstring_embeddings,
      docstring_mask,
      edge_sources,
      edge_dests,
      edge_types,
      true_indexes,
      false_indexes,
      raise_indexes,
      start_node_indexes,
      exit_node_indexes,
      step_limits,
  ):
    """Applies Transformer model on the inputs.

    Args:
      node_embeddings: TODO
      docstring_embeddings: TODO
      docstring_mask: TODO
      edge_sources: TODO
      edge_dests: TODO
      edge_types: TODO
      true_indexes: TODO
      false_indexes: TODO
      raise_indexes: TODO
      start_node_indexes: TODO
      exit_node_indexes: TODO
      step_limits: TODO
    Returns:
      output of a transformer decoder.
    """
    config = self.config
    info = self.info

    to_log = {}
    def log(value, label):
      to_log[label] = value
      return value
    # Inputs and configs.
    vocab_size = info.vocab_size
    output_token_vocabulary_size = info.vocab_size

    num_layers = config.max_steps
    assert num_layers >= 2, 'At least two steps are required.'
    num_layers = log(num_layers, label='num_layers')

    hidden_size = config.model.hidden_size

    # Initialize modules.
    embed = nn.Embed(
        num_embeddings=vocab_size,
        features=hidden_size,
        name='embed')
    cells = create_lstm_cells(config.model.rnn_cell.layers)
    lstm = StackedRNNCell(cells=cells, name='statement_embedder')

    # Embed individual tokens.
    assert inputs.ndim == 2  # (batch, len)
    inputs_i32 = inputs.astype('int32')
    token_embeddings = embed(inputs_i32)
    # token_embeddings.shape: batch_size, length, hidden_size

    batch_size = token_embeddings.shape[0]
    token_embeddings_bnlh = node_embeddings[:, :, None, :]  # length == 1
    # token_embeddings_bnlh.shape:
    #     batch_size, num_statements, length, hidden_size
    num_statements = token_embeddings_bnlh.shape[1]
    num_nodes = num_statements + 1

    # Create the statement embeddings.
    # We embed each statement individually by running the LSTM over the tokens.
    statement_initial_state = lstm.initialize_carry(
        jax.random.PRNGKey(0), cells, (), hidden_size)
    def embed_statement(token_embeddings):
      # token_embeddings.shape: tokens_per_statement, hidden_size
      _, results = lax.scan(lstm, statement_initial_state, token_embeddings)
      return results[-1]
    embed_all_statements = jax.vmap(embed_statement)  # single example
    statement_embeddings = jax.vmap(embed_all_statements)(token_embeddings_bnlh)
    # statement_embeddings.shape: batch_size, num_statements, hidden_size
    statement_embeddings = log(
        statement_embeddings, label='statement_embeddings')

    skip_encoder = SkipEncoder(config=config)
    skip_embeddings = jax.vmap(skip_encoder)(statement_embeddings)
    # skip_embeddings.shape: num_nodes, num_nodes, hidden_size
    skip_embeddings = log(skip_embeddings, label='skip_embeddings')

    # Create a hidden state for every node in every example.
    cells = create_lstm_cells(config.model.rnn_cell.layers)
    lstm = StackedRNNCell(cells=cells, name='execution_rnn')

    h = lstm.initialize_carry(
        jax.random.PRNGKey(0), cells,
        (batch_size, num_nodes), hidden_size)
    if config.model.rnn_cell.learn_start_state:
      # Add a learned state to the execution start states.
      def get_layer_initial_state(layer):
        stddev = 1
        return (
            self.param(
                f'initial_state_{layer}a',
                (hidden_size,),
                nn.initializers.normal(stddev=stddev)),
            self.param(
                f'initial_state_{layer}b',
                (hidden_size,),
                nn.initializers.normal(stddev=stddev)),
        )
      learned_initial_state = [
          get_layer_initial_state(layer)
          for layer in range(config.model.rnn_cell.layers)
      ]
      h = jax.tree_multimap(
          # h_part.shape: batch_size, num_nodes, hidden_size
          lambda h_part, initial_state_part: h_part + initial_state_part,
          h, learned_initial_state
      )

    # Create a soft instruction pointer for each example.
    p = jnp.zeros((batch_size, num_nodes,)).at[:, 0].set(1)

    h_key_dense = nn.Dense(
        name='h_key',
        features=hidden_size,
        kernel_init=nn.initializers.xavier_uniform(),
        bias_init=nn.initializers.normal(stddev=1e-6))
    skip_dense = nn.Dense(
        name='skip_dense',
        features=hidden_size,
        kernel_init=nn.initializers.xavier_uniform(),
        bias_init=nn.initializers.normal(stddev=1e-6))
    dense1 = nn.Dense(
        name='dense1',
        features=1,
        kernel_init=nn.initializers.xavier_uniform(),
        bias_init=nn.initializers.normal(stddev=1e-6))
    output_layer = nn.Dense(
        features=output_token_vocabulary_size,
        kernel_init=nn.initializers.xavier_uniform(),
        bias_init=nn.initializers.normal(stddev=1e-6),
        name='output_layer')

    concat, _ = make_concat(h)

    def _create_initial_mask(num_nodes, unused_length):
      """Creates a mask indicating which nodes can skip to which nodes."""
      # The first step must be from node 0->1.
      # The only entry that matters is the 0->1 entry.
      # The rest are to avoid nan.
      return jnp.zeros((num_nodes, num_nodes)).at[:, 1].set(1)

    def _create_mask(num_nodes, length, max_skip=0):
      """Creates a mask indicating which nodes can skip to which nodes.
      Args:
        num_nodes: The total number of nodes in the graph. One more than the max
          number of statements.
        length: The number of statements. Skipping beyond this point will not
          be permitted by the mask.
        max_skip: (optional) If non-zero, this is the max number of statements
          that can be skipped in a single step.
      Returns:
        A 0/1 mask indicating which nodes can be skipped to from which other
        nodes.
      """
      if max_skip:
        mask = jnp.tri(num_nodes, k=max_skip) - jnp.tri(num_nodes)
      else:
        mask = 1 - jnp.tri(num_nodes)
      mask = jnp.where(jnp.arange(num_nodes) <= length, mask, 0)
      # The exit node (and beyond) points only to the exit node:
      mask = mask.at[:, length].set(1)
      if max_skip:
        mask *= jnp.tri(num_nodes, k=max_skip)
      return mask

    def _create_final_mask(num_nodes, length):
      """Forces skipping to the exit node (length)."""
      return jnp.zeros((num_nodes, num_nodes)).at[:, length].set(1)

    def step(h, p, skip_embeddings_nnh, length, layer):
      """For a single example, execute every statement according to p.
      Args:
        h: The hidden state at every node at the start of the step.
        p: The instruction pointer's probability of being at each node at the
          start of the step.
        skip_embeddings_nnh: The skip embeddings for skipping from any node i to
          any node j. Shape: num_nodes, num_nodes, hidden_size.
        length: The number of statements in the example. Here this is
          (num_nodes - 1). With variable length programs, this should indicate
          the exit node.
        layer: The step number. If layer == num_layers - 1 (the last layer),
          set up the mask such that skipping to the exit node is required.
      Returns:
        The new hidden states for every node, the new instruction pointer,
        and additional data about the computation.
      """
      to_log_inner = {}
      def log(value, label):
        to_log_inner[label] = value
        return value

      # leaves(h).shape: num_nodes, hidden_size
      # p.shape: num_nodes,
      mask = _create_mask(num_nodes, length,
                          max_skip=config.model.skip_encoder.max_skip)
      mask = jnp.where(layer < num_layers - 1,
                       mask,
                       _create_final_mask(num_nodes, length))
      mask = jnp.where(layer == 0,
                       _create_initial_mask(num_nodes, length),
                       mask)
      if config.model.skip_encoder.skip_attention:
        skip_embeddings_keys = skip_dense(skip_embeddings_nnh)
        # skip_embeddings_keys.shape: num_nodes, num_nodes, hidden_size
        h_key = h_key_dense(concat(h))
        # h_key.shape: num_nodes, hidden_size

        h_skip_concat = jnp.concatenate([
            jnp.broadcast_to(h_key, (num_nodes, num_nodes, hidden_size)),
            skip_embeddings_keys
        ], axis=-1)
        h_skip_activations = nn.relu(h_skip_concat)

        skip_logits = jnp.squeeze(dense1(h_skip_activations), axis=-1)
        # skip_logits.shape: num_nodes, num_nodes

        # Mask skip_logits so that only forward execution is permitted,
        # and so that execution beyond `length` is not permitted.
        skip_logits = jnp.where(mask, skip_logits, -jnp.inf)
      else:
        skip_logits = jnp.where(mask, 1, -jnp.inf)
      skip_logits = log(skip_logits, label='skip_logits')
      # skip_logits.shape: num_nodes, num_nodes
      skip_p = nn.softmax(skip_logits, axis=-1)
      skip_p = log(skip_p, label='skip_p')
      # skip_p.shape: num_nodes, num_nodes

      def skip_execute(h, i, j):
        """Performs a single step of skip execution.
        Executes from node i to j, assuming a start state of h at node i.
        Args:
          h: The state at the start of skip execution.
          i: The index of the node we're skipping from.
          j: The index of the node we're skipping to.
        Returns:
          An updated state representing the RNN state at j if the models
          skip-executes from i to j.
        """
        state, _ = lstm(h, skip_embeddings_nnh[i, j, :])
        state = jax.tree_multimap(
            lambda part, default: jnp.where(j > i, part, default),
            state,
            h,
        )
        return state

      skip_from = jax.vmap(skip_execute, in_axes=(0, 0, None), out_axes=0)
      skip_from_to = jax.vmap(skip_from, in_axes=(None, None, 0), out_axes=1)
      state_proposals = skip_from_to(
          h, jnp.arange(num_nodes), jnp.arange(num_nodes))
      # This next line isn't necessary. It removes unused entries.
      state_proposals = jax.tree_map(
          lambda h: jnp.where(jnp.expand_dims(mask, -1), h, 0),
          state_proposals)
      # This next line isn't necessary. It removes unused entries.
      state_proposals = jax.tree_map(
          lambda h: jnp.where(jnp.expand_dims(p > 0, axis=(1, 2)), h, 0),
          state_proposals)

      proposed_logits = output_layer(concat(state_proposals))
      # proposed_logits.shape:
      #     num_nodes, num_nodes, output_token_vocabulary_size
      proposed_outputs = jnp.argmax(proposed_logits, axis=-1)
      proposed_outputs = log(proposed_outputs, label='proposed_outputs')

      hs = state_proposals
      # leaves(hs).shape: num_nodes, num_nodes, hidden_size

      denom = jnp.expand_dims(
          jnp.einsum('ij,i->j', skip_p, p) + 1e-7,
          1)
      h_new = jax.tree_map(
          lambda h_part: jnp.einsum('ijh,ij,i->jh', h_part, skip_p, p) / denom,
          hs)
      p_new = jnp.einsum('j,ji->i', p, skip_p)

      all_logits = output_layer(concat(h_new))
      # all_logits.shape: num_nodes, output_token_vocabulary_size

      all_outputs = jnp.expand_dims(jnp.argmax(all_logits, axis=-1), axis=-1)
      all_outputs = log(all_outputs, label='all_outputs')
      return h_new, p_new, to_log_inner

    step = jax.vmap(step, in_axes=(0, 0, 0, 0, None))

    for layer in range(num_layers):
      h = log(h, label=f'h_{layer}')
      p = log(p, label=f'p_{layer}')
      h, p, to_log_inner = step(h, p, skip_embeddings, statement_lengths, layer)
      for key, value in to_log_inner.items():
        log(value, label=f'{key}_{layer}')
    h = log(h, label=f'h_{num_layers}')
    p = log(p, label=f'p_{num_layers}')
    # leaves(h).shape: batch_size, num_nodes, hidden_size

    def get_final_state(h, length):
      # leaves(h).shape: num_nodes, hidden_size
      return jax.tree_map(lambda h_part: h_part[length, :], h)

    final_state = jax.vmap(get_final_state)(h, statement_lengths)
    # leaves(final_state).shape: batch_size, hidden_size
    final_embeddings = concat(final_state)
    # final_embeddings.shape: batch_size, k * hidden_size (k=cell_depth*layers)

    logits = output_layer(final_embeddings)
    # logits.shape: batch_size, base
    logits = jnp.expand_dims(logits, axis=1)
    # logits.shape: batch_size, 1, base
    logits = log(logits, label='logits')

    # TODO(dbieber): hcb temporarily disabled since causing errors in colab.
    # tap_func = functools.partial(log_lib.log_value, label='logs')
    # to_log = hcb.id_tap(tap_func, to_log)
    # for value in jax.tree_leaves(to_log):
    #   logits = lax.tie_in(value, logits)
    return logits