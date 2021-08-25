"""Models library."""

from core.models import ipagnn
from core.models import mlp
from core.models import transformer


def make_model(config):
  model_class = config.model_class
  if model_class == 'MlpModel':
    return mlp.MlpModel()
  elif model_class == 'Transformer':
    return transformer.Transformer(config=config)
  elif model_class == 'IPAGNN':
    return ipagnn.IPAGNN(config=config)
  else:
    raise ValueError('Unexpected model_class.')
