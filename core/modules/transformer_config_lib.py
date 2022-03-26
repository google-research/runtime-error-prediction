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

"""Transformer Config builder library."""

from third_party.flax_examples import transformer_modules


def make_transformer_config(config, vocab_size, deterministic):
  """Returns a transformer_modules.TransformerConfig to spec."""
  return make_transformer_config_num_layers(
      config.transformer_num_layers, config, vocab_size, deterministic)


def make_transformer_config_num_layers(num_layers, config, vocab_size, deterministic):
  """Returns a transformer_modules.TransformerConfig to spec."""
  return transformer_modules.TransformerConfig(
      vocab_size=vocab_size,
      output_vocab_size=vocab_size,
      emb_dim=config.transformer_emb_dim,
      num_heads=config.transformer_num_heads,
      num_layers=num_layers,
      qkv_dim=config.transformer_qkv_dim,
      mlp_dim=config.transformer_mlp_dim,
      dropout_rate=config.transformer_dropout_rate,
      attention_dropout_rate=config.transformer_attention_dropout_rate,
      max_len=config.max_tokens,
      deterministic=deterministic,
  )
