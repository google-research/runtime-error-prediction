import jax
import jax.numpy as jnp


def get_additional_logit(target_p, logits):
  """Computes an additional logit that when added to logits has probability target_p.

  Args:
    target_p: The target probability the additional logit should correspond to.
      softmax(concat([logits, result])) should give target_p as the probability
      for the final class.
    logits: The existing logits.
  Returns:
    A new logit, that when concatenated to logits, corresponds to a probability
    of target_p.
  """
  # target_p.shape: scalar.
  # logits.shape: NUM_CLASSES

  # Find target_logit such that:
  #   softmax(concat([target_logit, logits]) == target_p
  #   exp(target_logit) / sum(exp(concat([target_logit, logits])) == target_p
  #   exp(target_logit) == target_p * sum(exp(concat([target_logit, logits])))
  #   exp(target_logit) == target_p * sum(exp(logits)) + target_p * exp(target_logit)
  #   (1 - target_p) * exp(target_logit) == target_p * sum(exp(logits))
  #   exp(target_logit) == target_p / (1 - target_p) * sum(exp(logits))
  #   target_logit == log(target_p / (1 - target_p) * sum(exp(logits)))
  #   target_logit == log(target_p) - log(1 - target_p) + log(sum(exp(logits)))
  target_logit = (
      # TODO(dbieber): Make numerically stable. log(1-x).
      jnp.log(target_p) - jnp.log1p(-target_p)
      + jax.scipy.special.logsumexp(logits)
  )
  # TODO(dbieber): Test this.
  # target_logit.shape: scalar.
  return target_logit
