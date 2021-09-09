"""Transformer Config builder library."""

from third_party.flax_examples import transformer_modules


def make_transformer_config(config, vocab_size, deterministic):
  """Returns a transformer_modules.TransformerConfig to spec."""
  return transformer_modules.TransformerConfig(
      vocab_size=vocab_size,
      output_vocab_size=vocab_size,
      emb_dim=config.transformer_emb_dim,
      num_heads=config.transformer_num_heads,
      num_layers=config.transformer_num_layers,
      qkv_dim=config.transformer_qkv_dim,
      mlp_dim=config.transformer_mlp_dim,
      max_len=config.max_tokens,
  )
