import dataclasses
from typing import Any

from flax import linen as nn
import jax
import jax.numpy as jnp
from jax import lax

from core.modules.ipagnn import rnn


@dataclasses.dataclass
class Info:
  vocab_size: int


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
    output_token_vocabulary_size = info.vocab_size

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
    self.output_dense = nn.Dense(
        name='output_dense',
        features=output_token_vocabulary_size,
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
      step_limits,
  ):
    info = self.info
    config = self.config

    # State. Varies from step to step.
    hidden_states, instruction_pointer, current_step = carry

    # Inputs.
    vocab_size = info.vocab_size
    output_token_vocabulary_size = info.vocab_size
    hidden_size = config.hidden_size
    batch_size, num_nodes, unused_hidden_size = node_embeddings.shape
    # exit_indexes.shape: batch_size

    # State.
    # current_step.shape: batch_size,
    # leaves(hidden_states).shape: batch_size, num_nodes, hidden_size
    # instruction_pointer.shape: batch_size, num_nodes,

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
        true_indexes, false_indexes):
      # instruction_pointer.shape: num_nodes,
      # branch_decisions.shape: num_nodes, 2,
      # true_indexes.shape: num_nodes,
      # false_indexes.shape: num_nodes
      p_raise = raise_decisions[:, 0]
      p_noraise = raise_decisions[:, 1]
      p_true = p_noraise * branch_decisions[:, 0]
      p_false = p_noraise * branch_decisions[:, 1]
      raise_contributions = jax.ops.segment_sum(
          p_raise * instruction_pointer, XXX,
          num_segments=num_nodes)
      true_contributions = jax.ops.segment_sum(
          p_true * instruction_pointer, true_indexes,
          num_segments=num_nodes)
      false_contributions = jax.ops.segment_sum(
          p_false * instruction_pointer, false_indexes,
          num_segments=num_nodes)
      return raise_contributions + true_contributions + false_contributions
    update_instruction_pointer = jax.vmap(update_instruction_pointer_single_example)

    def aggregate_single_example(
        hidden_states, instruction_pointer, raise_decisions, branch_decisions,
        true_indexes, false_indexes):
      # leaves(hidden_states).shape: num_nodes, hidden_size
      # instruction_pointer.shape: num_nodes,
      # raise_decisions.shape: num_nodes, 2,
      # branch_decisions.shape: num_nodes, 2,
      # true_indexes.shape: num_nodes,
      # false_indexes.shape: num_nodes,
      p_raise = raise_decisions[:, 0]
      p_noraise = raise_decisions[:, 1]
      p_true = p_noraise * branch_decisions[:, 0]
      p_false = p_noraise * branch_decisions[:, 1]
      denominators = update_instruction_pointer_single_example(
          instruction_pointer, raise_decisions, branch_decisions,
          true_indexes, false_indexes)
      denominators += 1e-7
      # denominator.shape: num_nodes,

      def aggregate_state_component(h):
        # h.shape: num_nodes
        # p_true.shape: num_nodes
        # instruction_pointer.shape: num_nodes
        raise_contributions = jax.ops.segment_sum(
            h * p_raise * instruction_pointer, XXX,
            num_segments=num_nodes)
        true_contributions = jax.ops.segment_sum(
            h * p_true * instruction_pointer, true_indexes,
            num_segments=num_nodes)
        false_contributions = jax.ops.segment_sum(
            h * p_false * instruction_pointer, false_indexes,
            num_segments=num_nodes)
        # *_contributions.shape: num_nodes, hidden_size
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

    # Use the exit node's hidden state as it's hidden state contribution
    # to avoid "executing" the exit node.
    def mask_h(h_contribution, h, exit_index):
      # h_contribution.shape: num_nodes, hidden_size
      # h.shape: num_nodes, hidden_size
      # exit_index.shape: scalar.
      return h_contribution.at[exit_index, :].set(h[exit_index, :])
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

    hidden_state_contributions = jax.tree_multimap(
        lambda h1, h2: batch_mask_h(h1, h2, exit_indexes),
        hidden_state_contributions, hidden_states)
    # leaves(hidden_state_contributions).shape: batch_size, num_nodes, hidden_size

    # Raise decisions:
    raise_decision_logits = raise_decide(hidden_state_contributions)
    # raise_decision_logits.shape: batch_size, num_nodes, 2
    raise_decisions = nn.softmax(branch_decision_logits, axis=-1)
    # raise_decision.shape: batch_size, num_nodes, 2

    # Branch decisions:
    branch_decision_logits = branch_decide(hidden_state_contributions)
    # branch_decision_logits.shape: batch_size, num_nodes, 2
    branch_decisions = nn.softmax(branch_decision_logits, axis=-1)
    # branch_decision.shape: batch_size, num_nodes, 2

    # instruction_pointer.shape: batch_size, num_nodes
    # true_indexes.shape: batch_size, num_nodes
    # false_indexes.shape: batch_size, num_nodes
    instruction_pointer_new = update_instruction_pointer(
        instruction_pointer, raise_decisions, branch_decisions,
        true_indexes, false_indexes)
    # instruction_pointer_new.shape: batch_size, num_nodes

    hidden_states_new = aggregate(
        hidden_state_contributions, instruction_pointer,
        raise_decisions, branch_decisions,
        true_indexes, false_indexes)
    # leaves(hidden_states_new).shape: batch_size, num_nodes, hidden_size

    # current_step.shape: batch_size,
    # step_limits.shape: batch_size,
    hidden_states, instruction_pointer = keep_old_if_done(
        (hidden_states, instruction_pointer),
        (hidden_states_new, instruction_pointer_new),
        current_step,
        step_limits,
    )
    current_step = current_step + 1
    # leaves(hidden_states).shape: batch_size, num_nodes, hidden_size
    # instruction_pointer.shape: batch_size, num_nodes
    # current_step.shape: batch_size,
    return (hidden_states, instruction_pointer, current_step), None


class IPAGNNModule(nn.Module):
  """IPAGNN model with batch dimension (not graph batching)."""

  info: Any
  config: Any

  max_steps: int

  def setup(self):
    info = self.info
    config = self.config
    output_token_vocabulary_size = info.vocab_size

    # TODO(dbieber): Once linen makes it possible, set prevent_cse=False.
    IPAGNNLayerRemat = nn.remat(IPAGNNLayer)
    self.ipagnn_layer_scan = nn.scan(
        IPAGNNLayerRemat,
        variable_broadcast='params',
        split_rngs={'params': False},
        in_axes=(nn.broadcast,) * 8,
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
    hidden_size = config.hidden_size
    batch_size, num_nodes, unused_hidden_size = node_embeddings.shape
    # exit_indexes.shape: batch_size, 1
    exit_indexes = jnp.squeeze(exit_indexes, axis=-1)
    # exit_indexes.shape: batch_size

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
    # instruction_pointer.shape: batch_size, num_nodes,

    # Run self.max_steps steps of IPAGNNLayer.
    (hidden_states, instruction_pointer, current_step), _ = self.ipagnn_layer_scan(
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
        step_limits,
    )

    def get_final_state(hidden_states, exit_index):
      # leaves(hidden_states).shape: num_nodes, hidden_size
      # exit_index.shape: scalar.
      return jax.tree_map(lambda hs: hs[exit_index], hidden_states)
    # exit_indexes.shape: batch_size
    exit_node_hidden_states = jax.vmap(get_final_state)(hidden_states, exit_indexes)
    # leaves(exit_node_hidden_states).shape: batch_size, hidden_size
    exit_node_embeddings = jax.vmap(_rnn_state_to_embedding)(exit_node_hidden_states)
    # exit_node_embeddings.shape: batch_size, full_hidden_size

    return {
        'exit_node_embeddings': exit_node_embeddings,
    }
