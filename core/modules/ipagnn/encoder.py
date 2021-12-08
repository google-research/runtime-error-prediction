from flax import linen as nn

from third_party.flax_examples import transformer_modules

Encoder1DBlock = transformer_modules.Encoder1DBlock


class TransformerEncoder(nn.Module):
  """Transformer Model Encoder for sequence to sequence translation.

  Unlike transformer_modules.Encoder, this Encoder does not encode the input
  tokens itself. It assumes the tokens have already been encoded, and any
  desired positional embeddings have already been added.

  Attributes:
    config: TransformerConfig dataclass containing hyperparameters.
  """
  config: transformer_modules.TransformerConfig

  @nn.compact
  def __call__(self,
               encoded_inputs,
               encoder_mask=None):
    """Applies Transformer model on the encoded inputs.

    Args:
      encoded_inputs: pre-encoded input data.
      encoder_mask: decoder self-attention mask.

    Returns:
      output of a transformer encoder.
    """
    cfg = self.config

    x = encoded_inputs
    x = nn.Dropout(rate=cfg.dropout_rate)(x, deterministic=cfg.deterministic)
    x = x.astype(cfg.dtype)

    # Input Encoder
    for lyr in range(cfg.num_layers):
      x = Encoder1DBlock(config=cfg, name=f'encoderblock_{lyr}')(x, encoder_mask)

    encoded = nn.LayerNorm(dtype=cfg.dtype, name='encoder_norm')(x)

    return encoded


class TokenEncoder(nn.Module):
  """Only sums token-content embeddings and position embeddings."""

  transformer_config: transformer_modules.TransformerConfig
  num_embeddings: int
  features: int

  def setup(self):
    self.embed = nn.Embed(
        num_embeddings=self.num_embeddings,
        features=self.features,
        embedding_init=nn.initializers.normal(stddev=1.0))
    self.add_position_embeds = transformer_modules.AddPositionEmbs(
        config=self.transformer_config, decode=False, name='posembed_input')

  def __call__(self, tokens):
    # num_nodes.shape: batch_size.
    x = tokens.astype('int32')
    # x.shape: batch_size, max_tokens
    x = self.embed(x)
    # x.shape: batch_size, max_tokens, features
    x = self.add_position_embeds(x)
    return x
