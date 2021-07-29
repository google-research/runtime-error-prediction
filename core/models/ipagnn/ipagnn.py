import dataclasses
from typing import Any

from flax import linen as nn
import jax
import jax.numpy as jnp
from jax import lax


@dataclasses.dataclass
class Info:
  vocab_size: int


class IPAGNN(nn.Module):
  """IPAGNN model with batch dimension (not graph batching)."""

  info: Any
  config: Any

  rnn: Any
  max_steps: int

  def setup(self):
    info = self.info
    config = self.config
    output_token_vocabulary_size = info.vocab_size

    self.branch_decide_dense = nn.Dense(
        name='branch_decide_dense',
        features=2,
        kernel_init=nn.initializers.xavier_uniform(),
        bias_init=nn.initializers.normal(stddev=1e-6))
    self.output_dense = nn.Dense(
        name='output_dense',
        features=output_token_vocabulary_size,
        kernel_init=nn.initializers.xavier_uniform(),
        bias_init=nn.initializers.normal(stddev=1e-6))

  def __call__(
      self,
      node_embeddings,
      edge_sources,
      edge_dests,
      edge_types,
      exit_indexes,
      all_steps,
  ):
    info = self.info
    config = self.config

    # Inputs.
    vocab_size = info.vocab_size
    output_token_vocabulary_size = info.vocab_size
    hidden_size = config.model.hidden_size
    batch_size, num_nodes, unused_hidden_size = node_embeddings.shape

    # Initialize hidden states and soft instruction pointer.
    hidden_states = self.rnn.initialize_carry(
        jax.random.PRNGKey(0),
        (batch_size, num_nodes,), hidden_size)
    # leaves(hidden_states).shape: batch_size, num_nodes, hidden_size
    instruction_pointer = jax.ops.index_add(
        jnp.zeros((batch_size, num_nodes,)),
        jax.ops.index[:, 0],  # TODO(dbieber): Use "start_indexes" instead of 0.
        1
    )
    # instruction_pointer.shape: batch_size, num_nodes,

    def branch_decide_single_node(hidden_state):
      # leaves(hidden_state).shape: hidden_size
      hidden_state_concat = jnp.concatenate(
          jax.tree_leaves(hidden_state), axis=0)
      return self.branch_decide_dense(hidden_state_concat)
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
      # false_indexes: num_nodes
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

    def execute_single_node(hidden_state, node_embedding):
      # leaves(hidden_state).shape: hidden_size
      # node_embedding.shape: hidden_size
      # lstm outputs: (new_c, new_h), new_h
      # Recall new_h is derived from new_c.
      hidden_state, _ = self.rnn(hidden_state, node_embedding)
      return hidden_state
    execute = jax.vmap(execute_single_node)

    def step_single_example(hidden_states, instruction_pointer,
                            node_embeddings, true_indexes, false_indexes,
                            exit_index):
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

      aux = {
          'branch_decisions': branch_decisions,
          'hidden_state_contributions': hidden_state_contributions,
          'hidden_states_before': hidden_states,
          'hidden_states': hidden_states_new,
          'instruction_pointer_before': instruction_pointer,
          'instruction_pointer': instruction_pointer_new,
          'true_indexes': true_indexes,
          'false_indexes': false_indexes,
      }
      return hidden_states_new, instruction_pointer_new, aux

    def compute_logits_single_example(
        hidden_states, instruction_pointer, exit_index, steps,
        node_embeddings, true_indexes, false_indexes):
      """single_example refers to selecting a single exit node hidden state."""
      # leaves(hidden_states).shape: num_nodes, hidden_size

      def step_(carry, _):
        hidden_states, instruction_pointer, index = carry
        hidden_states_new, instruction_pointer_new, aux = (
            step_single_example(
                hidden_states, instruction_pointer,
                node_embeddings, true_indexes, false_indexes,
                exit_index)
        )
        carry = jax.tree_multimap(
            lambda new, old, index=index: jnp.where(index < steps, new, old),
            (hidden_states_new, instruction_pointer_new, index + 1),
            (hidden_states, instruction_pointer, index + 1),
        )
        return carry, aux
      if config.model.ipagnn.checkpoint and not self.is_initializing():
        step_ = jax.checkpoint(step_)

      carry = (hidden_states, instruction_pointer, jnp.array([0]))
      (hidden_states, instruction_pointer, _), aux = lax.scan(
          step_, carry, None, length=max_steps)

      final_state = jax.tree_map(lambda hs: hs[exit_index], hidden_states)
      # leaves(final_state).shape: hidden_size
      final_state_concat = jnp.concatenate(jax.tree_leaves(final_state), axis=0)
      logits = output_dense(final_state_concat)
      aux.update({
          'instruction_pointer_final': instruction_pointer,
          'hidden_states_final': hidden_states,
      })
      return logits, aux
    compute_logits = jax.vmap(compute_logits_single_example,
                              in_axes=(0, 0, 0, 0, 0, 0, 0))

    logits, aux = compute_logits(
        hidden_states, instruction_pointer, exit_indexes, all_steps,
        node_embeddings, true_indexes, false_indexes)
    logits = jnp.expand_dims(logits, axis=1)
    return logits
