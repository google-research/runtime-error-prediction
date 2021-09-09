"""Models library."""

from core.models import ipagnn
from core.models import mlp
from core.models import transformer
from core.modules import transformer_config_lib


def make_model(config, deterministic):
  model_class = config.model_class

  vocab_size = config.vocab_size  # TODO(dbieber): Load from tokenizer / info.
  transformer_config = transformer_config_lib.make_transformer_config(
      config,
      vocab_size,
      deterministic,
  )

  if model_class == 'MlpModel':
    return mlp.MlpModel()
  elif model_class == 'Transformer':
    return transformer.Transformer(
        config=config,
        transformer_config=transformer_config,
    )
  elif model_class == 'IPAGNN':
    return ipagnn.IPAGNN(
        config=config,
        transformer_config=transformer_config,
    )
  else:
    raise ValueError('Unexpected model_class.')
