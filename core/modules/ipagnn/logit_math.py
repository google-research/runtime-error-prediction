# Copyright (C) 2021 Google Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import jax
import jax.numpy as jnp


def get_additional_logit(target_p, logits_p, logits):
  """Computes an additional logit that when added to logits has probability target_p.

  Args:
    target_p: The target probability the additional logit should correspond to.
      softmax(concat([logits, result])) should give target_p as the probability
      for the result class.
    logits_p: The target probability mass to be held collectively by the logits.
      This can be 1 - target_p, but if it is not, then this function behaves as
      if first normalizing target_p and logits_p by (target_p + logits_p).
    logits: The existing logits.
  Returns:
    A new logit, that when concatenated to logits, corresponds to a probability
    of target_p.
  """
  # target_p.shape: scalar.
  # logits.shape: num_classes

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
      jnp.log(target_p) - jnp.log(logits_p)
      + jax.scipy.special.logsumexp(logits)
  )
  # target_logit.shape: scalar.
  return target_logit
