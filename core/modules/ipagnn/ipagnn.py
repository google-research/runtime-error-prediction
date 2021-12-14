from typing import Any

from flax import linen as nn
import jax
import jax.numpy as jnp

from core.lib.metrics import EvaluationMetric
from core.modules.ipagnn import rnn
from core.modules.ipagnn import raise_contributions as raise_contributions_lib


def _rnn_state_to_embedding(hidden_state):
  return jnp.concatenate(
      jax.tree_leaves(hidden_state), axis=-1)


def replace(indexes, value, replacement_value):
  return jnp.where(indexes == value, replacement_value, indexes)
batch_replace = jax.vmap(replace)


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

    if config.use_film:
      self.film_f_layer = nn.Dense(
          features=config.hidden_size,
          kernel_init=nn.initializers.xavier_uniform(),
          bias_init=nn.initializers.normal(stddev=1e-6))
      self.film_g_layer = nn.Dense(
          features=config.hidden_size,
          kernel_init=nn.initializers.xavier_uniform(),
          bias_init=nn.initializers.normal(stddev=1e-6))

    if config.use_cross_attention:
      num_heads = config.cross_attention_num_heads
      self.attention = nn.MultiHeadDotProductAttention(
          num_heads=num_heads,
          out_features=config.hidden_size,
          # ...
      )

    cells = rnn.create_lstm_cells(config.rnn_layers)
    self.lstm = rnn.StackedRNNCell(cells)

  def film_f(self, x):
    return nn.relu(self.film_f_layer(x))

  def film_g(self, x):
    return nn.relu(self.film_g_layer(x))

  def __call__(
      self,
      # State. Varies from step to step.
      carry,
      # Inputs. Shared across all steps.
      node_embeddings,
      docstring_embeddings,
      docstring_mask,
      edge_sources,
      edge_dests,
      edge_types,
      true_indexes,
      false_indexes,
      raise_indexes,
      exit_node_indexes,
      raise_node_indexes,
      step_limits,
  ):
    info = self.info
    config = self.config

    # State. Varies from step to step.
    hidden_states, instruction_pointer, attribution, current_step = carry

    # Inputs.
    vocab_size = info.vocab_size
    hidden_size = config.hidden_size
    batch_size, num_nodes, unused_hidden_size = node_embeddings.shape
    # true_indexes.shape: batch_size, num_nodes
    # faise_indexes.shape: batch_size, num_nodes
    # exit_node_indexes.shape: batch_size
    # raise_node_indexes.shape: batch_size

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
        raise_node_index, true_indexes, false_indexes, raise_indexes):
      # instruction_pointer.shape: num_nodes
      # raise_decisions.shape: num_nodes, 2
      # branch_decisions.shape: num_nodes, 2
      # raise_node_index.shape: scalar.
      # true_indexes.shape: num_nodes
      # false_indexes.shape: num_nodes

      p_raise = raise_decisions[:, 0]
      p_noraise = raise_decisions[:, 1]
      # assert p_raise + p_noraise == 1
      p_true = p_noraise * branch_decisions[:, 0]
      p_false = p_noraise * branch_decisions[:, 1]
      raise_contributions = jax.ops.segment_sum(
          p_raise * instruction_pointer, raise_indexes,
          num_segments=num_nodes)
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
          'raise_node_index': raise_node_index,
          'max_contribution': max_contribution,
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
        raise_node_index, true_indexes, false_indexes, raise_indexes):
      # leaves(hidden_states).shape: num_nodes, hidden_size
      # instruction_pointer.shape: num_nodes
      # raise_decisions.shape: num_nodes, 2
      # branch_decisions.shape: num_nodes, 2
      # raise_node_index.shape: scalar.
      # true_indexes.shape: num_nodes
      # false_indexes.shape: num_nodes
      p_raise = raise_decisions[:, 0]
      p_noraise = raise_decisions[:, 1]
      p_true = p_noraise * branch_decisions[:, 0]
      p_false = p_noraise * branch_decisions[:, 1]
      denominators, aux_ip = update_instruction_pointer_single_example(
          instruction_pointer, raise_decisions, branch_decisions,
          raise_node_index, true_indexes, false_indexes, raise_indexes)
      denominators += 1e-7
      # denominator.shape: num_nodes

      def aggregate_state_component(h):
        # h.shape: num_nodes
        # p_true.shape: num_nodes
        # instruction_pointer.shape: num_nodes
        raise_contributions = jax.ops.segment_sum(
            h * p_raise * instruction_pointer, raise_indexes,
            num_segments=num_nodes)
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

    def film_modulate_single(node_embedding, hidden_state, docstring_embeddings, docstring_mask):
      # node_embedding.shape: hidden_size
      # docstring_mask: length
      # docstring_embeddings: length, hidden_size
      # leaves(hidden_state).shape: hidden_size
      hidden_state_embedding = _rnn_state_to_embedding(hidden_state)
      # hidden_state_embedding.shape: hidden_size
      beta = self.film_f(jnp.concatenate([hidden_state_embedding, node_embedding]))
      # beta.shape: hidden_size
      gamma = self.film_g(jnp.concatenate([hidden_state_embedding, node_embedding]))
      # gamma.shape: hidden_size
      modulated_docstring_embedding = (
          beta * docstring_embeddings + gamma)
      # modulated_docstring_embedding.shape: length, hidden_size
      modulated_docstring_embedding = jnp.where(
          docstring_mask[:, None], modulated_docstring_embedding, -jnp.inf)

      # If the docstring_mask is False everywhere (because no docstring is available),
      # then the modulated_docstring_embedding is -inf everywhere.
      # In this case, the node_embedding should remain unchanged.

      # modulated_docstring_embedding.shape: length, hidden_size
      docstring_pooled = jnp.max(modulated_docstring_embedding, axis=0)
      # docstring_pooled.shape: hidden_size
      docstring_pooled = jnp.maximum(docstring_pooled, 0)
      # docstring_pooled.shape: hidden_size
      modulated_node_embedding = node_embedding + docstring_pooled
      # modulated_node_embedding.shape: hidden_size
      return modulated_node_embedding
    film_modulate_all_nodes = jax.vmap(film_modulate_single, in_axes=(0, 0, None, None))
    film_modulate = jax.vmap(film_modulate_all_nodes)

    def cross_attention_single(node_embedding, hidden_state, docstring_embeddings, docstring_mask):
      # node_embedding.shape: hidden_size
      # docstring_embeddings.shape: length, hidden_size
      # leaves(hidden_state).shape: hidden_size
      hidden_state_embedding = _rnn_state_to_embedding(hidden_state)
      # hidden_state_embedding.shape: hidden_size

      def attend_to_docstring(
          hidden_state_embedding,
          docstring_embeddings,
          node_embedding,
          docstring_mask):
        # hidden_state_embedding.shape: n * hidden_size
        # docstring_embeddings.shape: length, hidden_size
        # docstring_mask: length.
        # node_embedding.shape: hidden_size
        q = jnp.concatenate([hidden_state_embedding, node_embedding], axis=0)
        # q.shape: (n+1) * hidden_size
        inputs_q = jnp.expand_dims(q, axis=0)
        # inputs_q.shape: 1, (n+1) * hidden_size

        num_heads = config.cross_attention_num_heads
        
        # docstring_mask.shape: length
        mask = docstring_mask[None, None, :]  # adds leading dims.
        # mask.shape: 1, 1, length
        mask = jnp.tile(mask, (num_heads, 1, 1))
        # mask.shape: num_heads, 1, length
        y = self.attention(
            inputs_q=inputs_q,
            inputs_kv=docstring_embeddings,
            mask=mask,
        )
        # y.shape: 1, hidden_size
        y = jnp.squeeze(y, axis=0)
        # y.shape: hidden_size
        return y

      docstring_summary_embedding = attend_to_docstring(
          hidden_state_embedding,
          docstring_embeddings,
          node_embedding,
          docstring_mask)
      # docstring_summary_embedding.shape: hidden_size
      # node_embedding: hidden_size
      new_node_embeddings = jnp.concatenate(
          [node_embedding, docstring_summary_embedding],
          axis=0)
      # new_node_embeddings.shape: 2 * hidden_size
      return new_node_embeddings
    cross_attention_all_nodes = jax.vmap(cross_attention_single, in_axes=(0, 0, None, None))
    cross_attention = jax.vmap(cross_attention_all_nodes)

    # Take a full step of IPAGNN
    if config.use_film:
      node_embeddings = film_modulate(node_embeddings, hidden_states, docstring_embeddings, docstring_mask)
      # node_embeddings.shape: batch_size, num_nodes, hidden_size
    elif config.use_cross_attention:
      # node_embeddings.shape: batch_size, num_nodes, hidden_size
      node_embeddings = cross_attention(node_embeddings, hidden_states, docstring_embeddings, docstring_mask)
      # node_embeddings.shape: batch_size, num_nodes, 2 * hidden_size
    hidden_state_contributions = execute(hidden_states, node_embeddings)
    # leaves(hidden_state_contributions).shape: batch_size, num_nodes, hidden_size

    # Don't execute the exit node.
    hidden_state_contributions = jax.tree_multimap(
        lambda h1, h2: batch_mask_h(h1, h2, exit_node_indexes),
        hidden_state_contributions, hidden_states)
    # leaves(hidden_state_contributions).shape: batch_size, num_nodes, hidden_size

    # Raise decisions:
    if config.raise_in_ipagnn:
      # Don't execute the raise node.
      hidden_state_contributions = jax.tree_multimap(
          lambda h1, h2: batch_mask_h(h1, h2, raise_node_indexes),
          hidden_state_contributions, hidden_states)

      def set_values(a, value, index):
        # a.shape: num_nodes, 2
        # value.shape: 2.
        # index.shape: scalar.
        return a.at[index, :].set(value)
      batch_set = jax.vmap(set_values, in_axes=(0, None, 0))

      raise_decision_logits = raise_decide(hidden_state_contributions)
      # raise_decision_logits.shape: batch_size, num_nodes, 2
      raise_decisions = nn.softmax(raise_decision_logits, axis=-1)
      # raise_decision.shape: batch_size, num_nodes, 2
      # Make sure you cannot raise from the exit node.
      raise_decisions = batch_set(raise_decisions, jnp.array([0, 1]), exit_node_indexes)
      # Make sure you cannot raise from the raise node.
      raise_decisions = batch_set(raise_decisions, jnp.array([0, 1]), raise_node_indexes)
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
        raise_node_indexes, true_indexes, false_indexes, raise_indexes)
    # instruction_pointer_new.shape: batch_size, num_nodes

    hidden_states_new = aggregate(
        hidden_state_contributions, instruction_pointer,
        raise_decisions, branch_decisions,
        raise_node_indexes, true_indexes, false_indexes, raise_indexes)
    # leaves(hidden_states_new).shape: batch_size, num_nodes, hidden_size

    attribution = raise_contributions_lib.get_raise_contribution_step_batch(
        attribution,
        instruction_pointer,
        branch_decisions,
        raise_decisions,
        true_indexes,
        false_indexes,
        raise_indexes,
        num_nodes,
    )
    # attribution.shape: batch_size, num_nodes, num_nodes

    # current_step.shape: batch_size
    # step_limits.shape: batch_size
    instruction_pointer_orig = instruction_pointer
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
        'instruction_pointer_orig': instruction_pointer_orig,
        EvaluationMetric.INSTRUCTION_POINTER.value: instruction_pointer,
        'raise_decisions': raise_decisions,
        'branch_decisions': branch_decisions,
        'current_step': current_step,
        'hidden_states': hidden_states,
        'hidden_state_contributions': hidden_state_contributions,
    }
    aux.update(aux_ip)
    return (hidden_states, instruction_pointer, attribution, current_step), aux


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
    info = self.info
    config = self.config

    # Inputs.
    # true_indexes.shape: batch_size, num_nodes
    # faise_indexes.shape: batch_size, num_nodes
    hidden_size = config.hidden_size
    batch_size, num_nodes, unused_hidden_size = node_embeddings.shape
    # start_node_indexes.shape: batch_size, 1
    # exit_node_indexes.shape: batch_size, 1
    start_node_indexes = jnp.squeeze(start_node_indexes, axis=-1)
    exit_node_indexes = jnp.squeeze(exit_node_indexes, axis=-1)
    # start_node_indexes.shape: batch_size
    # exit_node_indexes.shape: batch_size

    def zero_node_embedding_single_example(node_embeddings, index):
      # node_embeddings.shape: num_nodes, hidden_size
      return node_embeddings.at[index, :].set(0)
    zero_node_embedding = jax.vmap(zero_node_embedding_single_example)

    # Set the initial node embedding for the exit node to zero.
    node_embeddings = zero_node_embedding(node_embeddings, exit_node_indexes)

    # Create a new exception node after the exit node
    raise_node_indexes = exit_node_indexes + 1
    if config.raise_in_ipagnn:
      # raise_node_indexes.shape: batch_size
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
      raise_indexes = jnp.pad(
          raise_indexes,
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
      node_embeddings = zero_node_embedding(node_embeddings, raise_node_indexes)

      # Add a self-loop for the exception node to true_indexes and false_indexes.
      def add_self_loop(array, index):
        return array.at[index].set(index)
      add_self_loop_batch = jax.vmap(add_self_loop)
      true_indexes = add_self_loop_batch(true_indexes, raise_node_indexes)
      false_indexes = add_self_loop_batch(false_indexes, raise_node_indexes)
      raise_indexes = add_self_loop_batch(raise_indexes, raise_node_indexes)
    else:
      # In the vanilla IPAGNN, replace any edge to the raise node with an
      # edge to the exit node.
      true_indexes = batch_replace(true_indexes, raise_node_indexes, exit_node_indexes)
      false_indexes = batch_replace(false_indexes, raise_node_indexes, exit_node_indexes)


    # Initialize hidden states and soft instruction pointer.
    current_step = jnp.zeros((batch_size,), dtype=jnp.int32)
    hidden_states = self.ipagnn_layer_scan.lstm.initialize_carry(
        jax.random.PRNGKey(0),
        (batch_size, num_nodes,), hidden_size)
    # leaves(hidden_states).shape: batch_size, num_nodes, hidden_size

    def make_instruction_pointer(start_node_index):
      return jnp.zeros((num_nodes,)).at[start_node_index].set(1)
    instruction_pointer = jax.vmap(make_instruction_pointer)(start_node_indexes)
    # instruction_pointer.shape: batch_size, num_nodes

    attribution = jnp.zeros((batch_size, num_nodes, num_nodes))

    # Run self.max_steps steps of IPAGNNLayer.
    (hidden_states, instruction_pointer, attribution, current_step), aux = self.ipagnn_layer_scan(
        # State:
        (hidden_states, instruction_pointer, attribution, current_step),
        # Inputs:
        node_embeddings,
        docstring_embeddings,
        docstring_mask,
        edge_sources,
        edge_dests,
        edge_types,
        true_indexes,
        false_indexes,
        raise_indexes,
        exit_node_indexes,
        raise_node_indexes,
        step_limits,
    )

    def get_hidden_state_single_example(hidden_states, node_index):
      # leaves(hidden_states).shape: num_nodes, hidden_size
      # exit_index.shape: scalar.
      return jax.tree_map(lambda hs: hs[node_index], hidden_states)

    # TODO(dbieber): Use only final LSTM layer's hidden state here, not full hidden states.
    get_hidden_state = jax.vmap(get_hidden_state_single_example)
    # exit_node_indexes.shape: batch_size
    exit_node_hidden_states = get_hidden_state(hidden_states, exit_node_indexes)
    # leaves(exit_node_hidden_states).shape: batch_size, hidden_size
    exit_node_embeddings = jax.vmap(_rnn_state_to_embedding)(exit_node_hidden_states)
    # exit_node_embeddings.shape: batch_size, full_hidden_size
    raise_node_hidden_states = get_hidden_state(hidden_states, raise_node_indexes)
    # leaves(raise_node_hidden_states).shape: batch_size, hidden_size
    raise_node_embeddings = jax.vmap(_rnn_state_to_embedding)(raise_node_hidden_states)
    # raise_node_embeddings.shape: batch_size, full_hidden_size

    get_instruction_pointer_value = jax.vmap(lambda ip, node_index: ip[node_index])
    exit_node_instruction_pointer = get_instruction_pointer_value(instruction_pointer, exit_node_indexes)
    # exit_node_instruction_pointer.shape: batch_size
    raise_node_instruction_pointer = get_instruction_pointer_value(instruction_pointer, raise_node_indexes)
    # raise_node_instruction_pointer.shape: batch_size

    if config.raise_in_ipagnn and config.unsupervised_localization:
      localization_logits = attribution[jnp.arange(batch_size), raise_node_indexes]
      aux['localization_logits'] = localization_logits

    aux.update({
        'exit_node_instruction_pointer': exit_node_instruction_pointer,
        'exit_node_embeddings': exit_node_embeddings,
        'raise_node_instruction_pointer': raise_node_instruction_pointer,
        'raise_node_embeddings': raise_node_embeddings,
    })
    return aux
