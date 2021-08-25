from typing import List, Optional, Text

import ml_collections

Config = ml_collections.ConfigDict


def default_config():
  """The default config."""
  config = Config()
  config.rnn_layers = 2
  config.vocab_size = 30000  # TODO(dbieber): Load from tokenizer / move to Info.
  config.model_class: Text = 'IPAGNN'
  config.epochs: Optional[int] = 0
  config.batch_size: int = 128
  config.max_tokens: int = 512
  config.max_num_nodes: int = 128
  config.max_num_edges: int = 128
  config.max_steps: int = 174
  config.hidden_size: int = 16
  config.allowlist: Optional[List[int]] = None
  config.multidevice: bool = True
  config.restore_checkpoint_dir: Optional[Text] = None

  config.eval_freq = 1000
  config.save_freq = 1000
  config.eval_metric_name = 'F1-score'

  config.early_stopping_delta = 0.001
  config.early_stopping_threshold = 4
  return config


def get_config():
  """Gets the config."""
  config = default_config()
  config.lock()
  return config
