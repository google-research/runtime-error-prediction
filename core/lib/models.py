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

"""Models library."""

from core.models import ipagnn
from core.models import ggnn
from core.models import mlp
from core.models import rnn
from core.models import transformer
from core.modules import transformer_config_lib


def make_model(config, info, deterministic):
  model_class = config.model_class

  vocab_size = info.vocab_size
  transformer_config = transformer_config_lib.make_transformer_config(
      config,
      vocab_size,
      deterministic,
  )
  docstring_transformer_config = transformer_config_lib.make_transformer_config_num_layers(
      config.docstring_transformer_num_layers,
      config,
      vocab_size,
      deterministic,
  )

  if model_class == 'MlpModel':
    return mlp.MlpModel(info=info)
  elif model_class == 'Transformer':
    return transformer.Transformer(
        config=config,
        info=info,
        transformer_config=transformer_config,
    )
  elif model_class == 'MILTransformer':
    return transformer.MILTransformer(
        config=config,
        info=info,
        transformer_config=transformer_config,
    )
  elif model_class == 'IPAGNN':
    return ipagnn.IPAGNN(
        config=config,
        info=info,
        transformer_config=transformer_config,
        docstring_transformer_config=docstring_transformer_config,
    )
  elif model_class == 'GGNN':
    return ggnn.GGNN(
        config=config,
        info=info,
        transformer_config=transformer_config,
        docstring_transformer_config=docstring_transformer_config,
    )
  elif model_class == 'LSTM':
    return rnn.LSTM(
        config=config,
        info=info,
        transformer_config=transformer_config,
    )
  else:
    raise ValueError('Unexpected model_class.')
