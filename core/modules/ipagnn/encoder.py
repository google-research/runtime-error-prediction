from flax import linen as nn

from third_party.flax_examples import transformer_modules, lstm_modules

Encoder1DBlock = transformer_modules.Encoder1DBlock


class TransformerEncoder(nn.Module):
  """Transformer Model Encoder for sequence to sequence translation.

  Unlike transformer_modules.Encoder, this Encoder does not encode the input
  tokens itself. It assumes the tokens have already been encoded, and any
  desired positional embeddings have already been aded.

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


class LSTMEncoder(nn.Module):
  """LSTM Model Encoder for sequence to sequence translation.

  This Encoder does not encode the input
  tokens itself. It assumes the tokens have already been encoded, and any
  desired positional embeddings have already been aded.

  Attributes:
    config: LSTMConfig dataclass containing hyperparameters.
  """
  config: lstm_modules.LSTMConfig

  @nn.compact
  def __call__(self,
               encoded_inputs,):
    """Applies Transformer model on the encoded inputs.

    Args:
      encoded_inputs: pre-encoded input data.

    Returns:
      output of a lstm encoder.
    """
    cfg = self.config
    batch_size, max_tokens, _ = encoded_inputs.shape
    x = encoded_inputs
    x = nn.Dropout(rate=cfg.dropout_rate)(x, deterministic=cfg.deterministic)
    x = x.astype(cfg.dtype)

    # Input Encoder
    for lyr in range(cfg.num_layers):
      initial_state = lstm_modules.SimpleLSTM().initialize_carry((batch_size,), cfg.hidden_dim)
      _, x = lstm_modules.SimpleLSTM()(initial_state, x) # TODO(rgoel): Add layer norm to each layer

    encoded = nn.LayerNorm(dtype=cfg.dtype, name='encoder_norm')(x)

    return encoded
