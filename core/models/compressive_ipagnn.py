"""Skip Encoder Model supporting control flow graphs."""

import functools
from typing import Any

from absl import logging  # pylint: disable=unused-import
import flax
from flax.linen import nn
import jax
from jax import lax
import jax.numpy as jnp
from core.modules.ipagnn import rnn

Embed = nn.Embed
StackedRNNCell = rnn.StackedRNNCell


@flax.struct.dataclass
class InterpreterState:
  step: int
  hidden_states: Any
  instruction_pointer: Any


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


def create_instruction_pointer(start, num_nodes):
  """Creates a soft instruction pointer initialized at `start`."""
  return jnp.zeros((num_nodes,)).at[start].set(1.0)


class NodeEmbedder(nn.Module):
  """Embeds the statement at each node."""

  config: Any
  info: Any

  @nn.compact
  def __call__(self, data):
    config = self.config
    info = self.info
    hidden_size = config.model.hidden_size
    vocab_size = info.vocab_size

    def emb_init(key, shape, dtype=jnp.float32):
      return jax.random.uniform(
          key, shape, dtype,
          -config.initialization.maxval,
          config.initialization.maxval)

    token_embed = Embed(num_embeddings=vocab_size,
                        features=hidden_size,
                        emb_init=emb_init,
                        name='token_embed')
    # TODO(dbieber): Consider concat and MLP in place of LSTM embedding.
    cells = create_lstm_cells(config.model.rnn_cell.layers)
    embed_lstm = StackedRNNCell(cells=cells, name='embed_lstm')
    def embed_single_node(token_embedding):
      # token_embedding.shape: statement_length, hidden_size
      initial_hidden_state = embed_lstm.initialize_carry(
          jax.random.PRNGKey(0), cells, (), hidden_size)
      _, result = lax.scan(embed_lstm, initial_hidden_state, token_embedding)
      return result[-1]
    node_embed = jax.vmap(embed_single_node)

    token_embeddings = token_embed(data)
    # token_embeddings.shape: num_nodes, statement_length, hidden_size
    node_embeddings = node_embed(token_embeddings)
    # node_embeddings.shape: num_nodes, hidden_size
    return node_embeddings


class SkipEmbedder(nn.Module):
  """Module that creates skip embeddings."""

  config: Any

  @nn.compact
  def __call__(
      self, node_embeddings, max_steps,
      num_nodes, true_indexes, false_indexes, exit_index):
    config = self.config
    embedder = SkipEmbedderSingleSource(config=config)
    embedder = functools.partial(
        embedder,
        node_embeddings=node_embeddings,
        max_steps=max_steps,
        num_nodes=num_nodes,
        true_indexes=true_indexes,
        false_indexes=false_indexes,
        exit_index=exit_index)
    from_node_indexes = jnp.arange(num_nodes)
    skip_embeddings = jax.vmap(embedder)(from_node_indexes)
    # You cannot skip from a node to itself.
    # We place the node embedding on the diagonal, so that "skipping in place"
    # represents normal non-skip execution of the node.
    embeddings = skip_embeddings.at[jnp.diag_indices(num_nodes)].set(
        node_embeddings)
    return embeddings


class SkipEmbedderSingleSource(nn.Module):
  """Module that creates skip embeddings from a single start node i."""

  config: Any

  @nn.compact
  def __call__(
      self, from_node_index, node_embeddings, max_steps,
      num_nodes, true_indexes, false_indexes, exit_index):
    """Creates skip embeddings representing the possible paths from i to j.

    Args:
      from_node_index: The node i to start at. This function creates skip
        embeddings starting only at this node.
      node_embeddings: Tensor (num_nodes, hidden_size) with embedding for each
        node.
      max_steps: The maximum number of execution steps permitted in a single
        skip.
      num_nodes: The number of nodes in the graph.
      true_indexes: For each node, index of the next node if the true branch is
        taken. Shape: num_nodes.
      false_indexes: For each node, index of the next node if the false branch
        is taken. If a node is not a branch, this is the same as the true index.
        Shape is (num_nodes,).
      exit_index: The index of the exit node.
    Returns:
      A single embedding for each destination node. Shape:
      (num_nodes, hidden_size)
    """
    config = self.config
    execute_cells = create_lstm_cells(config.model.rnn_cell.layers)
    execute_lstm = StackedRNNCell(cells=execute_cells,
                                         name='skip_execute_lstm')
    def execute_single_node(hidden_state, node_embedding):
      # node_embedding.shape: hidden_size
      result, _ = execute_lstm(hidden_state, node_embedding)
      return result
    execute = jax.vmap(execute_single_node)

    branch_decide_dense = nn.Dense(
        name='branch_decide_dense',
        features=2,
        kernel_init=nn.initializers.xavier_uniform(),
        bias_init=nn.initializers.normal(stddev=1e-6))

    def branch_decide_single_node(hidden_state):
      # leaves(hidden_state).shape: hidden_size
      hidden_state_concat = jnp.concatenate(
          jax.tree_leaves(hidden_state), axis=0)
      return branch_decide_dense(hidden_state_concat)
    branch_decide = jax.vmap(branch_decide_single_node)

    def update_instruction_pointer(
        instruction_pointer, branch_decisions, true_indexes, false_indexes):
      # instruction_pointer.shape: num_nodes,
      # branch_decisions: num_nodes, 2,
      # true_indexes: num_nodes,
      # false_indexes: num_nodes
      p_true = branch_decisions[:, 0]
      p_false = branch_decisions[:, 1]
      true_contributions = jax.ops.segment_sum(
          p_true * instruction_pointer, true_indexes,
          num_segments=num_nodes)
      false_contributions = jax.ops.segment_sum(
          p_false * instruction_pointer, false_indexes,
          num_segments=num_nodes)
      return true_contributions + false_contributions

    def aggregate(
        hidden_states, instruction_pointer, branch_decisions,
        true_indexes, false_indexes):
      # leaves(hidden_states).shape: num_nodes, hidden_size
      # instruction_pointer.shape: num_nodes,
      # branch_decisions: num_nodes, 2,
      # true_indexes: num_nodes,
      # false_indexes: num_nodes,
      p_true = branch_decisions[:, 0]
      p_false = branch_decisions[:, 1]
      denominators = update_instruction_pointer(
          instruction_pointer, branch_decisions, true_indexes, false_indexes)
      denominators += 1e-7
      # denominator.shape: num_nodes,

      def aggregate_component(h):
        # h.shape: num_nodes
        # p_true.shape: num_nodes
        # instruction_pointer.shape: num_nodes
        true_contributions = jax.ops.segment_sum(
            h * p_true * instruction_pointer, true_indexes,
            num_segments=num_nodes)
        false_contributions = jax.ops.segment_sum(
            h * p_false * instruction_pointer, false_indexes,
            num_segments=num_nodes)
        # *_contributions.shape: num_nodes, hidden_size
        return (true_contributions + false_contributions) / denominators
      aggregate_component = jax.vmap(aggregate_component, in_axes=1, out_axes=1)

      return jax.tree_map(aggregate_component, hidden_states)

    def step_single_example(hidden_states, instruction_pointer,
                            node_embeddings, true_indexes, false_indexes,
                            exit_index):
      """Computes new values of p_{s,i,t} and h^(skip)_{s,i,t}."""
      # Execution (e.g. apply RNN)
      # leaves(hidden_states).shape: num_nodes, hidden_size
      # instruction_pointer.shape: num_nodes,
      # node_embeddings.shape: num_nodes, hidden_size
      hidden_state_contributions = execute(hidden_states, node_embeddings)
      # leaves(hidden_state_contributions).shape: num_nodes, hidden_size

      # Use the exit node's hidden state as it's hidden state contribution
      # to avoid "executing" the exit node.
      def mask_h(h_contribution, h):
        return h_contribution.at[exit_index, :].set(h[exit_index, :])
      hidden_state_contributions = jax.tree_multimap(
          mask_h, hidden_state_contributions, hidden_states)

      # Branch decisions (e.g. Dense layer)
      branch_decision_logits = branch_decide(hidden_state_contributions)
      branch_decisions = nn.softmax(branch_decision_logits, axis=-1)

      # Update state
      instruction_pointer_new = update_instruction_pointer(
          instruction_pointer, branch_decisions, true_indexes, false_indexes)
      hidden_states_new = aggregate(
          hidden_state_contributions, instruction_pointer, branch_decisions,
          true_indexes, false_indexes)
      return hidden_states_new, instruction_pointer_new

    def step_(carry, _):
      hidden_states, instruction_pointer = carry
      hidden_states_new, instruction_pointer_new = (
          step_single_example(
              hidden_states, instruction_pointer,
              node_embeddings, true_indexes, false_indexes,
              exit_index)
      )
      carry = hidden_states_new, instruction_pointer_new
      return carry, carry
    if config.model.ipagnn2.checkpoint and not self.is_initializing():
      step_ = jax.checkpoint(step_)

    instruction_pointer = create_instruction_pointer(start=from_node_index,
                                                     num_nodes=num_nodes)
    # instruction_pointer.shape: num_nodes,
    hidden_states = StackedRNNCell.initialize_carry(
        jax.random.PRNGKey(0), execute_cells, (num_nodes,),
        config.model.hidden_size)
    # hidden_states.shape: num_nodes, hidden_size

    carry = hidden_states, instruction_pointer
    _, carries = lax.scan(step_, carry, None, length=max_steps)
    # We want to aggregate the hidden states across time, averaging according
    # to probability.
    hidden_states, instruction_pointer = carries
    # leaves(hidden_states).shape: max_steps, num_nodes, hidden_size
    # instruction_pointer.shape: max_steps, num_nodes
    hidden_states = jax.tree_map(
        lambda h: jnp.sum(h * jnp.expand_dims(instruction_pointer, -1), axis=0),
        hidden_states)
    # leaves(hidden_states): num_nodes, hidden_size
    # TODO(dbieber): get result from hidden state in more principled way
    result = jax.tree_leaves(hidden_states)[-1]
    return nn.LayerNorm(result, name='skip_layer_norm')


class SkipEncoderLineByLine(nn.Module):
  """Skip encoder layer (line by line RNN) for a single example."""

  config: Any

  @nn.compact
  def __call__(self, node_embeddings):
    """Creates skip embeddings for a single example.

    The skip embedding from node i to node j is an RNN run over the code from
    statement i to statement j-1 inclusive.

    Args:
      node_embeddings: A single example's node embeddings. Shape
        is (num_nodes, hidden_size).
    Returns:
      The skip embeddings for all pairs of statements in the example. Shape is
        (num_statements, num_statements, hidden_size). Axis order is
        val[from, to, d].
    """
    config = self.config
    hidden_size = config.model.hidden_size
    skip_embedder_layer_norm = nn.LayerNorm(
        name='skip_embedder_layer_norm')
    statement_embeddings_nh = node_embeddings[:-1]
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


class MaskMaker(nn.Module):
  """Determines which locations are OK to skip to."""

  config: Any

  @nn.compact
  def __call__(self, step, max_steps, exit_index, post_domination_matrix, length, num_nodes):
    """Creates the skip mask.

    Args:
      step: The current step number [0, max_steps).
      max_steps: The maximum number of steps for the example. At the final step,
        the mask enforces skipping to the exit node.
      exit_index: The index of the exit node.
      post_domination_matrix: A 0/1 matrix indicating which nodes post-dominate
        which other nodes.
      length: The number of nodes in the example, including the exit node, but
        not including unused nodes (after the exit node).
      num_nodes: The number of nodes including unused padding nodes.
    Returns:
      The mask indicating which locations are OK to skip to.
    """
    config = self.config
    # Initial mask: read the first statement (the inputs)
    initial_mask = jnp.zeros((num_nodes, num_nodes)).at[:, 1].set(1)

    # Default mask: only skips to post-dominating nodes are permitted
    default_mask = post_domination_matrix
    # The exit node (and beyond) points only to the exit node:
    # Note that every node is post-dominated by the exit node.
    default_mask = default_mask.at[:, exit_index].set(1)

    # Final mask: forced skip to exit
    final_mask = jnp.zeros((num_nodes, num_nodes)).at[:, exit_index].set(1)

    # Select the mask using the step number.
    mask = jnp.where(step < max_steps - 1, default_mask, final_mask)
    mask = jnp.where(step == 0, initial_mask, mask)
    return mask


class MaskMakerLineByLine(nn.Module):
  """Determines which locations are OK to skip to; only permits line-by-line."""

  config: Any

  @nn.compact
  def __call__(
      self, step, max_steps, exit_index, post_domination_matrix,
      length, num_nodes):
    """Creates the skip mask.

    Args:
      step: The current step number [0, max_steps).
      max_steps: The maximum number of steps for the example. At the final step,
        the mask enforces skipping to the exit node.
      exit_index: The index of the exit node.
      post_domination_matrix: A 0/1 matrix indicating which nodes post-dominate
        which other nodes.
      length: The number of nodes in the example, including the exit node, but
        not including unused nodes (after the exit node).
      num_nodes: The number of nodes including unused padding nodes.
    Returns:
      The mask indicating which locations are OK to skip to.
    """
    config = self.config
    # Initial mask: read the first statement (the inputs)
    initial_mask = jnp.zeros((num_nodes, num_nodes)).at[:, 1].set(1)

    # Default mask: only skip to next-line node is permitted
    off_diagonal = jnp.tri(num_nodes, k=1) - jnp.tri(num_nodes)
    # The exit node (and beyond) points only to the exit node:
    default_mask = jnp.where(jnp.tri(num_nodes, k=1),
                             off_diagonal.at[:, exit_index].set(1),
                             0)
    # Never advance beyond the exit node.
    default_mask = jnp.where(jnp.arange(num_nodes) > exit_index,
                             0, default_mask)

    # Final mask: forced skip to exit
    final_mask = jnp.zeros((num_nodes, num_nodes)).at[:, exit_index].set(1)

    # Select the mask using the step number.
    mask = jnp.where(step < max_steps - 1, default_mask, final_mask)
    mask = jnp.where(step == 0, initial_mask, mask)
    return mask


class MaskMakerNoSkip(nn.Module):
  """Determines which locations are OK to skip to; no skipping is permitted."""

  config: Any

  @nn.compact
  def __call__(
      self, step, max_steps, exit_index, post_domination_matrix,
      length, num_nodes):
    """Creates the skip mask. Disallows skipping.

    Args:
      step: The current step number [0, max_steps).
      max_steps: The maximum number of steps for the example. At the final step,
        the mask enforces skipping to the exit node.
      exit_index: The index of the exit node.
      post_domination_matrix: A 0/1 matrix indicating which nodes post-dominate
        which other nodes.
      length: The number of nodes in the example, including the exit node, but
        not including unused nodes (after the exit node).
      num_nodes: The number of nodes including unused padding nodes.
    Returns:
      The mask indicating which locations are OK to skip to.
    """
    config = self.config
    # The diagonal represents normal (non-skip) execution.
    default_mask = jnp.diagonal(num_nodes)

    # Final mask: forced skip to exit
    final_mask = jnp.zeros((num_nodes, num_nodes)).at[:, exit_index].set(1)

    # Select the mask using the step number.
    mask = jnp.where(step < max_steps - 1, default_mask, final_mask)
    return mask


def make_mask_maker(config):
  mask_maker_kind = config.model.skip_encoder.mask_maker
  if mask_maker_kind == 'default':
    return MaskMaker(config=config, name='mask_maker')
  elif mask_maker_kind == 'no-skip':
    return MaskMakerNoSkip(config=config, name='mask_maker')
  elif mask_maker_kind == 'line-by-line':
    return MaskMakerLineByLine(config=config, name='mask_maker')
  else:
    raise ValueError('Unexpected mask maker kind.', mask_maker_kind)


class SkipDecider(nn.Module):
  """Decides how much to skip to each of the valid skip destinations."""

  config: Any

  @nn.compact
  def __call__(self, hidden_states, skip_embeddings, skip_mask):
    config = self.config
    decider = SkipDeciderSingleSource(config=config)
    # leaves(hidden_states).shape: num_nodes, hidden_size
    # skip_embeddings.shape: num_nodes, num_nodes, hidden_size
    # skip_mask.shape: num_nodes, num_nodes
    decisions = jax.vmap(decider)(hidden_states, skip_embeddings, skip_mask)
    return decisions


class SkipDeciderSingleSource(nn.Module):
  """Decides how much to skip to each of the valid skip destinations."""

  config: Any

  @nn.compact
  def __call__(self, hidden_states, skip_embeddings, skip_mask):
    config = self.config
    num_nodes = skip_embeddings.shape[0]
    hidden_size = config.model.hidden_size
    key_dense = nn.Dense(
        name='key_dense',
        features=hidden_size,
        kernel_init=nn.initializers.xavier_uniform(),
        bias_init=nn.initializers.normal(stddev=1e-6))
    query_dense = nn.Dense(
        name='query_dense',
        features=hidden_size,
        kernel_init=nn.initializers.xavier_uniform(),
        bias_init=nn.initializers.normal(stddev=1e-6))
    logit_dense = nn.Dense(
        name='logit_dense',
        features=1,
        kernel_init=nn.initializers.xavier_uniform(),
        bias_init=nn.initializers.normal(stddev=1e-6))

    # A fixed start node is assumed.
    # leaves(hidden_states).shape: hidden_size
    # skip_embeddings: All skip embeddings from the same start node.
    # skip_embeddings.shape: num_nodes, hidden_size
    keys = key_dense(skip_embeddings)
    # keys.shape: num_nodes, hidden_size
    hidden_concat = jnp.concatenate(jax.tree_leaves(hidden_states), axis=-1)
    queries = query_dense(hidden_concat)
    # queries.shape: hidden_size
    kq = jnp.concatenate([keys,
                          jnp.broadcast_to(queries, (num_nodes, hidden_size))],
                         axis=-1)
    kq_activations = nn.relu(kq)
    # kq_activations.shape: num_nodes, (2*hidden_size)
    logits = jnp.squeeze(logit_dense(kq_activations), axis=-1)
    # logits.shape: num_nodes
    masked_logits = jnp.where(skip_mask, logits, -jnp.inf)
    # masked_logits.shape: num_nodes
    return nn.softmax(masked_logits, axis=-1)


class SkipExecutor(nn.Module):
  """For each start node, executes skipping to each destination node.

  Recall that the diagonal of skip_embeddings has the node embeddings. So
  skip-executing from i->i actually represents regular (non-skip) execution of
  node i. This requires making a branch decision to determine the new node after
  execution.
  """

  config: Any

  @nn.compact
  def __call__(self, hidden_states, skip_embeddings, execute_cells):
    config = self.config
    # leaves(hidden_states): num_nodes, hidden_size
    # skip_embeddings.shape: num_nodes, num_nodes, hidden_size
    execute_lstm = StackedRNNCell(cells=execute_cells,
                                  name='execute_lstm')
    def execute_i_to_j(hidden_state_i, embedding_ij):
      # leaves(hidden_state_i).shape: hidden_size
      # leaves(embedding_ij).shape: hidden_size
      new_state_ij, _ = execute_lstm(hidden_state_i, embedding_ij)
      # leaves(new_state_ij).shape: hidden_size
      return new_state_ij
    execute_all_to_j = jax.vmap(execute_i_to_j, in_axes=0, out_axes=0)
    execute_all_to_all = jax.vmap(execute_all_to_j,
                                  in_axes=(None, 1), out_axes=1)

    hidden_state_proposals = execute_all_to_all(hidden_states, skip_embeddings)
    # leaves(hidden_state_proposals).shape: num_nodes, num_nodes, hidden_size
    return hidden_state_proposals


class BranchDecider(nn.Module):
  """Assuming no skipping, decides how much to take the True and False branches.
  """

  config: Any

  @nn.compact
  def __call__(self, hidden_state_proposals):
    branch_dense = nn.Dense(
        name='branch_dense',
        features=2,
        kernel_init=nn.initializers.xavier_uniform(),
        bias_init=nn.initializers.normal(stddev=1e-6))

    # leaves(hidden_state_proposal).shape: num_nodes, hidden_size
    embeddings = jnp.concatenate(jax.tree_leaves(hidden_state_proposals),
                                 axis=-1)
    # embeddings.shape: num_nodes, (k*hidden_size)
    branch_logits = branch_dense(embeddings)
    # branch_logits.shape: num_nodes, 2
    branch_decisions = nn.softmax(branch_logits, axis=-1)
    return branch_decisions


class Aggregator(nn.Module):
  """Applies IPA aggregation to the proposed states."""

  config: Any

  @nn.compact
  def __call__(
      self, interpreter_state, hidden_state_proposals,
      hidden_state_skip_proposals, skip_decisions,
      branch_decisions, node_embeddings, true_indexes, false_indexes):
    config = self.config
    instruction_pointer = interpreter_state.instruction_pointer
    # instruction_pointer.shape: num_nodes
    num_nodes = instruction_pointer.shape[0]

    # skip_decisions.shape: num_nodes, num_nodes
    yes_skip_decisions = skip_decisions.at[jnp.diag_indices(num_nodes)].set(0)
    # yes_skip_decisions.shape: num_nodes, num_nodes
    no_skip_decisions = jnp.diagonal(skip_decisions)
    # no_skip_decisions.shape: num_nodes
    # branch_decisions.shape: num_nodes, 2
    p_true = branch_decisions[:, 0]
    p_false = branch_decisions[:, 1]

    # instruction_pointer[j] = (
    #   instruction_pointer[i] * skip_decisions[i, j]
    #   + skip_decisions[i, i] * branch_decisions[True] if true_indexes[i]==j
    #   + skip_decisions[i, i] * branch_decisions[False] if false_indexes[i]==j)
    p_branch_true = instruction_pointer * no_skip_decisions * p_true
    p_branch_false = instruction_pointer * no_skip_decisions * p_false
    true_branch_contributions = jax.ops.segment_sum(
        p_branch_true, true_indexes, num_segments=num_nodes)
    false_branch_contributions = jax.ops.segment_sum(
        p_branch_false, false_indexes, num_segments=num_nodes)
    skip_contributions = jnp.einsum(
        'i,ij->j', instruction_pointer, yes_skip_decisions)
    # *_contributions.shape: num_nodes
    new_instruction_pointer = (
        true_branch_contributions
        + false_branch_contributions
        + skip_contributions)
    # new_instruction_pointer.shape: num_nodes

    def aggregate_component(h, h_skip):
      # h.shape: num_nodes, hidden_size
      # h_skip.shape: num_nodes, num_nodes, hidden_size
      # p_true.shape == p_false.shape: num_nodes
      # yes_skip_decisions.shape: num_nodes, num_nodes
      # no_skip_decisions.shape: num_nodes
      # instruction_pointer.shape: num_nodes
      true_branch_contributions = jax.ops.segment_sum(
          h * p_branch_true[:, None], true_indexes,
          num_segments=num_nodes)
      false_branch_contributions = jax.ops.segment_sum(
          h * p_branch_false[:, None], false_indexes,
          num_segments=num_nodes)
      skip_contributions = jnp.einsum(
          'ijh,i,ij->jh', h_skip, instruction_pointer, yes_skip_decisions)
      # *_contributions.shape: num_nodes, hidden_size
      return (
          (true_branch_contributions
           + false_branch_contributions
           + skip_contributions)
          / (new_instruction_pointer[:, None] + 1e-7)
      )

    new_hidden_states = jax.tree_multimap(aggregate_component,
                                          hidden_state_proposals,
                                          hidden_state_skip_proposals)
    return InterpreterState(
        step=interpreter_state.step + 1,
        instruction_pointer=new_instruction_pointer,
        hidden_states=new_hidden_states,
    )


class Decoder(nn.Module):
  """Decodes final hidden states into logits."""

  @nn.compact
  def __call__(self, hidden_states, exit_index, vocab_size):
    logits_dense = nn.Dense(
        name='logits_dense',
        features=vocab_size,
        kernel_init=nn.initializers.xavier_uniform(),
        bias_init=nn.initializers.normal(stddev=1e-6))

    exit_hidden_states = jax.tree_map(lambda h: h[exit_index], hidden_states)
    exit_concat = jnp.concatenate(jax.tree_leaves(exit_hidden_states), axis=-1)
    logits = logits_dense(exit_concat)
    return logits


class SkipIPAGNNSingle(nn.Module):
  """Skip-IPAGNN model for a single example."""

  config: Any
  info: Any
  max_steps: int

  @nn.compact
  def __call__(self, 
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
      step_limits):
    config = self.config
    info = self.info

    # Get inputs:
    # true_indexes.shape: num_nodes
    # false_indexes.shape: num_nodes
    start_indexes = start_node_indexes  # TODO(dbieber): Rename singular.
    exit_index = exit_node_indexes
    steps_all = step_limits
    # steps_all.shape: scalar.
    num_nodes = node_embeddings.shape[0]  # Includes padding.
    # TODO(dbieber): Get the actual post_domination_matrix.
    post_domination_matrix = jnp.ones((num_nodes, num_nodes))
    # length is the number of nodes, excluding padding.
    length = exit_node_indexes
    # length.shape: scalar.
    output_token_vocabulary_size = info.vocab_size
    max_steps = self.max_steps

    # Create modules:
    node_embedder = NodeEmbedder(info=info, config=config,
                                 name='node_embedder')
    # Make skip embedder.
    max_skip_steps = (config.model.skip_encoder.max_skip
                      or max_steps)
    skip_embedder = SkipEmbedder(
        max_steps=max_skip_steps,
        num_nodes=num_nodes,
        true_indexes=true_indexes,
        false_indexes=false_indexes,
        exit_index=exit_index,
        config=config,
        name='skip_embedder')
    # skip_embedder = SkipEncoderLineByLine(config=config,
    #                                       name='skip_embedder')
    mask_maker = make_mask_maker(config)
    skip_decider = SkipDecider(config=config, name='skip_decider')
    execute_cells = create_lstm_cells(config.model.rnn_cell.layers)
    skip_executor = SkipExecutor(
        execute_cells=execute_cells, config=config,
        name='skip_executor')
    branch_decider = BranchDecider(name='branch_decider')
    aggregator = Aggregator(
        true_indexes=true_indexes, false_indexes=false_indexes, config=config,
        name='aggregator')
    decoder = Decoder(vocab_size=output_token_vocabulary_size,
                      name='decoder')

    # Pre-execution computation:
    # node_embeddings.shape: num_nodes, hidden_size
    skip_embeddings = skip_embedder(node_embeddings)
    # skip_embeddings.shape: num_nodes, num_nodes, hidden_size
    # skip_mask.shape: num_nodes, num_nodes

    # Execution Definition (Skip, Execute, Branch, Aggregate):
    def step(interpreter_state, unused_input):
      hidden_states = interpreter_state.hidden_states

      # Determine which nodes are valid to skip to.
      skip_mask = mask_maker(interpreter_state.step, steps_all, exit_index,
                             post_domination_matrix, length, num_nodes)
      # SkipDecider: For each node, choose how much to skip to each of the
      # allowed skip destinations.
      skip_decisions = skip_decider(hidden_states, skip_embeddings, skip_mask)
      # skip_decisions.shape: num_nodes, num_nodes

      # SkipExecutor: For each destination for each node, run the RNN to
      # determine what the hidden state would be if we went to that destination.
      hidden_state_skip_proposals = skip_executor(
          hidden_states, skip_embeddings)
      # leaves(hidden_state_proposals).shape: num_nodes, num_nodes, hidden_size
      # Prevent executing the exit node.
      hidden_state_skip_proposals = jax.tree_multimap(
          lambda hp, h: hp.at[exit_index, exit_index].set(h[exit_index]),
          hidden_state_skip_proposals, hidden_states
      )
      # The diagonal of hidden_state_proposals represents the hidden that
      # results from regular (non-skip) execution of the node.
      hidden_state_proposals = jax.tree_map(lambda hp: jnp.diagonal(hp).T,
                                            hidden_state_skip_proposals)
      # leaves(hidden_state_proposals).shape: num_nodes, hidden_size

      # BranchDecider: For each node, given that we've chosen not to skip (and
      # hence are just executing the single statement at that note), decide
      # whether to take the True or False branch. This decision only matters if
      # the statement is an if/while.
      branch_decisions = branch_decider(hidden_state_proposals)
      # branch_decisions.shape: num_nodes, 2

      # Aggregate: Compute the new soft instruction pointer using the skip and
      # branch decisions. Aggregate the hidden state proposals accordingly.
      new_interpreter_state = aggregator(
          interpreter_state,
          hidden_state_proposals, hidden_state_skip_proposals,
          skip_decisions, branch_decisions, node_embeddings)
      # instruction_pointer.shape: num_nodes,
      # leaves(hidden_states): num_nodes, hidden_size

      # Only perform num_steps steps of computation.
      new_interpreter_state = jax.tree_multimap(
          lambda a, b: jnp.where(interpreter_state.step[0] < steps_all, a, b),
          new_interpreter_state,
          interpreter_state)

      return new_interpreter_state, None
    if config.model.ipagnn2.checkpoint and not self.is_initializing():
      step = jax.remat(step)

    # Initialization:
    initial_instruction_pointer = create_instruction_pointer(0, num_nodes)
    initial_hidden_states = StackedRNNCell.initialize_carry(
        jax.random.PRNGKey(0), execute_cells, (num_nodes,),
        config.model.hidden_size)

    # Execution
    initial_interpreter_state = InterpreterState(
        step=jnp.array([0]),
        instruction_pointer=initial_instruction_pointer,
        hidden_states=initial_hidden_states)
    final_interpreter_state, _ = lax.scan(
        step, initial_interpreter_state, None, length=max_steps)
    final_hidden_states = final_interpreter_state.hidden_states
    # leaves(final_hidden_states): num_nodes, hidden_size

    # Decode
    logits = decoder(final_hidden_states, exit_index)
    # logits.shape: vocab_size
    return logits


class SkipIPAGNN(nn.Module):
  """Skip-IPAGNN model with batch dimension (not graph batching)."""

  config: Any
  info: Any
  max_steps: int

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
      step_limits):
    config = self.config
    info = self.info
    ipagnn = SkipIPAGNNSingle(
        config=config, info=info, max_steps=self.max_steps,
        name='ipagnn')
    ipagnn_batch = jax.vmap(ipagnn)
    logits = ipagnn_batch(
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
        step_limits)
    logits = logits[:, None, :]
    return logits
