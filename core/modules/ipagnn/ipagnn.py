from typing import Any

from flax import linen as nn
import numpy as np
import jax
import jax.numpy as jnp

from core.modules.ipagnn import rnn


def _rnn_state_to_embedding(hidden_state):
  return jnp.concatenate(
      jax.tree_leaves(hidden_state), axis=-1)


class IPAGNNLayer(nn.Module):
  """IPAGNN single-layer with batch dimension (not graph batching)."""
  info: Any
  config: Any

  def setup(self):
    info = self.info
    config = self.config

    self.raise_decide_dense = nn.Dense(
        name='raise_decide_dense',
        features=2,  # raise or don't raise.
        kernel_init=nn.initializers.xavier_uniform(),
        bias_init=nn.initializers.normal(stddev=1e-6))
    self.branch_decide_dense = nn.Dense(
        name='branch_decide_dense',
        features=2,  # true branch or false branch.
        kernel_init=nn.initializers.xavier_uniform(),
        bias_init=nn.initializers.normal(stddev=1e-6))

    cells = rnn.create_lstm_cells(config.rnn_layers)
    self.lstm = rnn.StackedRNNCell(cells)

  def __call__(
      self,
      # State. Varies from step to step.
      carry,
      # Inputs. Shared across all steps.
      node_embeddings,
      edge_sources,
      edge_dests,
      edge_types,
      true_indexes,
      false_indexes,
      exit_indexes,
      raise_indexes,
      step_limits,
  ):
    info = self.info
    config = self.config

    # State. Varies from step to step.
    hidden_states, instruction_pointer, current_step = carry

    # Inputs.
    vocab_size = info.vocab_size
    hidden_size = config.hidden_size
    batch_size, num_nodes, unused_hidden_size = node_embeddings.shape
    # true_indexes.shape: batch_size, num_nodes
    # faise_indexes.shape: batch_size, num_nodes
    # exit_indexes.shape: batch_size
    # raise_indexes.shape: batch_size

    # State.
    # current_step.shape: batch_size
    # leaves(hidden_states).shape: batch_size, num_nodes, hidden_size
    # instruction_pointer.shape: batch_size, num_nodes

    def raise_decide_single_node(hidden_state):
      # leaves(hidden_state).shape: hidden_size
      hidden_state_embedding = _rnn_state_to_embedding(hidden_state)
      return self.raise_decide_dense(hidden_state_embedding)
    raise_decide_single_example = jax.vmap(raise_decide_single_node)
    raise_decide = jax.vmap(raise_decide_single_example)

    def branch_decide_single_node(hidden_state):
      # leaves(hidden_state).shape: hidden_size
      hidden_state_embedding = _rnn_state_to_embedding(hidden_state)
      return self.branch_decide_dense(hidden_state_embedding)
    branch_decide_single_example = jax.vmap(branch_decide_single_node)
    branch_decide = jax.vmap(branch_decide_single_example)

    def update_instruction_pointer_single_example(
        instruction_pointer, raise_decisions, branch_decisions,
        raise_index, true_indexes, false_indexes):
      # instruction_pointer.shape: num_nodes
      # raise_decisions.shape: num_nodes, 2
      # branch_decisions.shape: num_nodes, 2
      # raise_index.shape: scalar.
      # true_indexes.shape: num_nodes
      # false_indexes.shape: num_nodes
      p_raise = raise_decisions[:, 0]
      p_noraise = raise_decisions[:, 1]
      # assert p_raise + p_noraise == 1
      p_true = p_noraise * branch_decisions[:, 0]
      p_false = p_noraise * branch_decisions[:, 1]
      raise_contribution = jnp.sum(p_raise * instruction_pointer)
      # raise_contribution.shape: scalar.
      raise_contributions = (
          jnp.zeros((num_nodes,)).at[raise_index].set(raise_contribution)
      )
      # raise_contributions.shape: num_nodes
      max_contribution = jnp.max(raise_contributions)
      true_contributions = jax.ops.segment_sum(
          p_true * instruction_pointer, true_indexes,
          num_segments=num_nodes)
      false_contributions = jax.ops.segment_sum(
          p_false * instruction_pointer, false_indexes,
          num_segments=num_nodes)
      aux = {
          'p_raise': p_raise,
          'p_noraise': p_noraise,
          'p_true': p_true,
          'p_false': p_false,
          'raise_index': raise_index,
          'max_contribution': max_contribution,
          'raise_contribution': raise_contribution,
          'raise_contributions': raise_contributions,
          'true_contributions': true_contributions,
          'false_contributions': false_contributions,
      }
      instruction_pointer_new = (
          raise_contributions + true_contributions + false_contributions)
      return instruction_pointer_new, aux
    update_instruction_pointer = jax.vmap(update_instruction_pointer_single_example)

    def aggregate_single_example(
        hidden_states, instruction_pointer, raise_decisions, branch_decisions,
        raise_index, true_indexes, false_indexes):
      # leaves(hidden_states).shape: num_nodes, hidden_size
      # instruction_pointer.shape: num_nodes
      # raise_decisions.shape: num_nodes, 2
      # branch_decisions.shape: num_nodes, 2
      # raise_index.shape: scalar.
      # true_indexes.shape: num_nodes
      # false_indexes.shape: num_nodes
      p_raise = raise_decisions[:, 0]
      p_noraise = raise_decisions[:, 1]
      p_true = p_noraise * branch_decisions[:, 0]
      p_false = p_noraise * branch_decisions[:, 1]
      denominators, aux_ip = update_instruction_pointer_single_example(
          instruction_pointer, raise_decisions, branch_decisions,
          raise_index, true_indexes, false_indexes)
      denominators += 1e-7
      # denominator.shape: num_nodes

      def aggregate_state_component(h):
        # h.shape: num_nodes
        # p_true.shape: num_nodes
        # instruction_pointer.shape: num_nodes
        raise_contribution = jnp.sum(h * p_raise * instruction_pointer)
        # raise_contribution.shape: scalar.
        raise_contributions = (
            jnp.zeros((num_nodes,)).at[raise_index].set(raise_contribution)
        )
        # raise_contributions.shape: num_nodes
        true_contributions = jax.ops.segment_sum(
            h * p_true * instruction_pointer, true_indexes,
            num_segments=num_nodes)
        false_contributions = jax.ops.segment_sum(
            h * p_false * instruction_pointer, false_indexes,
            num_segments=num_nodes)
        # *_contributions.shape: num_nodes
        return (raise_contributions + true_contributions + false_contributions) / denominators
      aggregate_states = jax.vmap(aggregate_state_component, in_axes=1, out_axes=1)

      return jax.tree_map(aggregate_states, hidden_states)
    aggregate = jax.vmap(aggregate_single_example)

    def execute_single_node(hidden_state, node_embedding):
      # leaves(hidden_state).shape: hidden_size
      # node_embedding.shape: hidden_size
      # lstm outputs: (new_c, new_h), new_h
      # Recall new_h is derived from new_c.
      hidden_state, _ = self.lstm(hidden_state, node_embedding)
      return hidden_state
    execute = jax.vmap(execute_single_node)

    # We'll use the exit node's hidden state as it's hidden state contribution
    # to avoid "executing" the exit node. We'll do the same for the raise node.
    def mask_h(h_contribution, h, node_index):
      # h_contribution.shape: num_nodes, hidden_size
      # h.shape: num_nodes, hidden_size
      # node_index.shape: scalar.
      return h_contribution.at[node_index, :].set(h[node_index, :])
    batch_mask_h = jax.vmap(mask_h)

    # If we've taken allowed_steps steps already, keep the old values.
    def keep_old_if_done_single_example(old, new, current_step, step_limit):
      return jax.tree_multimap(
          lambda new, old: jnp.where(current_step < step_limit, new, old),
          new, old)
    keep_old_if_done = jax.vmap(keep_old_if_done_single_example)

    # Take a full step of IPAGNN
    hidden_state_contributions = execute(hidden_states, node_embeddings)
    # leaves(hidden_state_contributions).shape: batch_size, num_nodes, hidden_size

    # Don't execute the exit node.
    hidden_state_contributions = jax.tree_multimap(
        lambda h1, h2: batch_mask_h(h1, h2, exit_indexes),
        hidden_state_contributions, hidden_states)
    # leaves(hidden_state_contributions).shape: batch_size, num_nodes, hidden_size

    # Raise decisions:
    if config.raise_in_ipagnn:
      # Don't execute the raise node.
      hidden_state_contributions = jax.tree_multimap(
          lambda h1, h2: batch_mask_h(h1, h2, raise_indexes),
          hidden_state_contributions, hidden_states)

      raise_decision_logits = raise_decide(hidden_state_contributions)
      # raise_decision_logits.shape: batch_size, num_nodes, 2
      raise_decisions = nn.softmax(raise_decision_logits, axis=-1)
      # raise_decision.shape: batch_size, num_nodes, 2
    else:
      raise_decisions = jnp.concatenate([
          jnp.zeros(shape=(batch_size, num_nodes, 1)),  # raise
          jnp.ones(shape=(batch_size, num_nodes, 1)),  # don't raise
      ], axis=-1)
      # raise_decision.shape: batch_size, num_nodes, 2

    # Branch decisions:
    branch_decision_logits = branch_decide(hidden_state_contributions)
    # branch_decision_logits.shape: batch_size, num_nodes, 2
    branch_decisions = nn.softmax(branch_decision_logits, axis=-1)
    # branch_decision.shape: batch_size, num_nodes, 2

    # instruction_pointer.shape: batch_size, num_nodes
    # true_indexes.shape: batch_size, num_nodes
    # false_indexes.shape: batch_size, num_nodes
    instruction_pointer_new, aux_ip = update_instruction_pointer(
        instruction_pointer, raise_decisions, branch_decisions,
        raise_indexes, true_indexes, false_indexes)
    # instruction_pointer_new.shape: batch_size, num_nodes

    hidden_states_new = aggregate(
        hidden_state_contributions, instruction_pointer,
        raise_decisions, branch_decisions,
        raise_indexes, true_indexes, false_indexes)
    # leaves(hidden_states_new).shape: batch_size, num_nodes, hidden_size

    # current_step.shape: batch_size
    # step_limits.shape: batch_size
    hidden_states, instruction_pointer = keep_old_if_done(
        (hidden_states, instruction_pointer),
        (hidden_states_new, instruction_pointer_new),
        current_step,
        step_limits,
    )
    current_step = current_step + 1
    # leaves(hidden_states).shape: batch_size, num_nodes, hidden_size
    # instruction_pointer.shape: batch_size, num_nodes
    # current_step.shape: batch_size

    aux = {
        'instruction_pointer': instruction_pointer,
        'raise_decisions': raise_decisions,
        'branch_decisions': branch_decisions,
        'current_step': current_step,
        'hidden_states': hidden_states,
        'hidden_state_contributions': hidden_state_contributions,
    }
    aux.update(aux_ip)
    return (hidden_states, instruction_pointer, current_step), aux


class IPAGNNModule(nn.Module):
  """IPAGNN model with batch dimension (not graph batching)."""

  info: Any
  config: Any

  max_steps: int

  def setup(self):
    info = self.info
    config = self.config

    # TODO(dbieber): Once linen makes it possible, set prevent_cse=False.
    IPAGNNLayerRemat = nn.remat(IPAGNNLayer)
    self.ipagnn_layer_scan = nn.scan(
        IPAGNNLayerRemat,
        variable_broadcast='params',
        split_rngs={'params': False},
        in_axes=nn.broadcast,
        length=self.max_steps,
    )(info=info, config=config)

  def __call__(
      self,
      node_embeddings,
      edge_sources,
      edge_dests,
      edge_types,
      true_indexes,
      false_indexes,
      exit_indexes,
      step_limits,
  ):
    info = self.info
    config = self.config

    # Inputs.
    # true_indexes.shape: batch_size, num_nodes
    # faise_indexes.shape: batch_size, num_nodes
    hidden_size = config.hidden_size
    batch_size, num_nodes, unused_hidden_size = node_embeddings.shape
    # exit_indexes.shape: batch_size, 1
    exit_indexes = jnp.squeeze(exit_indexes, axis=-1)
    # exit_indexes.shape: batch_size

    def zero_node_embedding_single_example(node_embeddings, index):
      # node_embeddings.shape: num_nodes, hidden_size
      return node_embeddings.at[index, :].set(0)
    zero_node_embedding = jax.vmap(zero_node_embedding_single_example)

    # Set the initial node embedding for the exit node to zero.
    node_embeddings = zero_node_embedding(node_embeddings, exit_indexes)

    # Create a new exception node after the exit node
    raise_indexes = exit_indexes + 1
    if config.raise_in_ipagnn:
      # raise_indexes.shape: batch_size
      num_nodes += 1

      # Pad true_indexes and false_indexes.
      true_indexes = jnp.pad(
          true_indexes,
          ((0, 0), (0, 1)),
          constant_values=0
      )
      false_indexes = jnp.pad(
          false_indexes,
          ((0, 0), (0, 1)),
          constant_values=0
      )
      # Pad node_embeddings with a zero node embedding for the new node
      # node_embeddings.shape: batch_size, max_num_nodes, hidden_size
      node_embeddings = jnp.pad(
          node_embeddings,
          ((0, 0), (0, 1), (0, 0)),
          constant_values=0
      )
      # Set the initial node embedding for the exception node to zero.
      node_embeddings = zero_node_embedding(node_embeddings, raise_indexes)

      # Add a self-loop for the exception node to true_indexes and false_indexes.
      def add_self_loop(array, index):
        return array.at[index].set(index)
      add_self_loop_batch = jax.vmap(add_self_loop)
      true_indexes = add_self_loop_batch(true_indexes, raise_indexes)
      false_indexes = add_self_loop_batch(false_indexes, raise_indexes)

    # Initialize hidden states and soft instruction pointer.
    current_step = jnp.zeros((batch_size,), dtype=jnp.int32)
    hidden_states = self.ipagnn_layer_scan.lstm.initialize_carry(
        jax.random.PRNGKey(0),
        (batch_size, num_nodes,), hidden_size)
    # leaves(hidden_states).shape: batch_size, num_nodes, hidden_size
    instruction_pointer = jax.ops.index_add(
        jnp.zeros((batch_size, num_nodes,)),
        jax.ops.index[:, 0],  # TODO(dbieber): Use "start_indexes" instead of 0.
        1
    )
    # instruction_pointer.shape: batch_size, num_nodes

    # Run self.max_steps steps of IPAGNNLayer.
    (hidden_states, instruction_pointer, current_step), aux = self.ipagnn_layer_scan(
        # State:
        (hidden_states, instruction_pointer, current_step),
        # Inputs:
        node_embeddings,
        edge_sources,
        edge_dests,
        edge_types,
        true_indexes,
        false_indexes,
        exit_indexes,
        raise_indexes,
        step_limits,
    )

    def get_hidden_state_single_example(hidden_states, node_index):
      # leaves(hidden_states).shape: num_nodes, hidden_size
      # exit_index.shape: scalar.
      return jax.tree_map(lambda hs: hs[node_index], hidden_states)
    get_hidden_state = jax.vmap(get_hidden_state_single_example)
    # exit_indexes.shape: batch_size
    exit_node_hidden_states = get_hidden_state(hidden_states, exit_indexes)
    # leaves(exit_node_hidden_states).shape: batch_size, hidden_size
    exit_node_embeddings = jax.vmap(_rnn_state_to_embedding)(exit_node_hidden_states)
    # exit_node_embeddings.shape: batch_size, full_hidden_size
    raise_node_hidden_states = get_hidden_state(hidden_states, raise_indexes)
    # leaves(raise_node_hidden_states).shape: batch_size, hidden_size
    raise_node_embeddings = jax.vmap(_rnn_state_to_embedding)(raise_node_hidden_states)
    # raise_node_embeddings.shape: batch_size, full_hidden_size

    get_instruction_pointer_value = jax.vmap(lambda ip, node_index: ip[node_index])
    exit_node_instruction_pointer = get_instruction_pointer_value(instruction_pointer, exit_indexes)
    # exit_node_instruction_pointer.shape: batch_size
    raise_node_instruction_pointer = get_instruction_pointer_value(instruction_pointer, raise_indexes)
    # raise_node_instruction_pointer.shape: batch_size

    aux.update({
        'exit_node_instruction_pointer': exit_node_instruction_pointer,
        'exit_node_embeddings': exit_node_embeddings,
        'raise_node_instruction_pointer': raise_node_instruction_pointer,
        'raise_node_embeddings': raise_node_embeddings,
    })
    return aux
