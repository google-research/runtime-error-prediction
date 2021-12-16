import numpy as np


def get_post_domination_matrix(cfg):
  """Computes the post-domination matrix for nodes in the control flow graph.

  A node i is post-dominated by another node j if every path from i to the exit
  includes j.

  Args:
    cfg: The control flow graph to compute the post domination matrix for.
  Returns:
    The 0/1 post-domination matrix. output[i, j] is 1 if i is post-dominated by
    j, 0 otherwise.
  """
  post_dominator_sets = get_post_dominator_sets(cfg)
  num_nodes = len(cfg.nodes) + 1  # Add one for the exit node.
  mat = np.zeros((num_nodes, num_nodes))

  # Create mapping from cfg_node to index in matrix.
  node_indexes = {
      cfg_node: i for i, cfg_node in enumerate(cfg.nodes)
  }
  node_indexes['<exit>'] = len(cfg.nodes)

  for node, post_dominators in post_dominator_sets.items():
    for post_dominator in post_dominators:
      mat[node_indexes[node], node_indexes[post_dominator]] = 1
  return mat


def get_post_dominator_sets(cfg):
  """Computes the set of post-dominating nodes for each node in the graph.

  A node i is post-dominated by another node j if every path from i to the exit
  includes j.

  Args:
    cfg: The control flow graph to compute the post domination matrix for.
  Returns:
    A dict with the post-domination sets for all control flow nodes. output[i]
    is the set of all control flow nodes j such that j post-dominates i. The
    exit node is represented by '<exit>'.
  """
  dominator_sets = {cfg_node: set(cfg.nodes) | {'<exit>'}
                    for cfg_node in cfg.nodes}
  dominator_sets['<exit>'] = {'<exit>'}

  def succ(cfg_node):
    """Returns the set of successors for a given control flow graph node."""
    cfg_node_is_end_of_block = cfg_node == cfg_node.block.control_flow_nodes[-1]
    if (cfg_node_is_end_of_block and
        any(block.label == '<exit>' for block in cfg_node.block.next)):
      # The node exits to the program exit node.
      return cfg_node.next | {'<exit>'}
    return cfg_node.next

  # Iterate the fixed-point equation until convergence to find post-dominators:
  # D(parent) = Union({parent}, Intersect(D(children)))
  # A node n is post-dominated by itself (trivially), and by any node which post
  # dominates all of n's immediate successors.
  change = True
  while change:
    change = False
    for cfg_node in reversed(cfg.nodes):
      old_value = dominator_sets[cfg_node].copy()
      for p in succ(cfg_node):
        dominator_sets[cfg_node] &= dominator_sets[p]
      dominator_sets[cfg_node] |= {cfg_node}
      if old_value != dominator_sets[cfg_node]:
        change = True
  return dominator_sets
