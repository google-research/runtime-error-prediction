import ml_collections

DEFAULT_DATASET_PATH = codenet_paths.DEFAULT_DATASET_PATH
Config = ml_collections.ConfigDict


def default_config():
  config = Config()
  config.rnn_layers = 2
  config.vocab_size = 30000  # TODO(dbieber): Load from tokenizer / move to Info.
  config.model_class: Text = 'IPAGNN'
  config.epochs: Optional[int] = None
  config.batch_size: int = 128
  config.max_tokens: int = 512
  config.max_num_nodes: int = 128
  config.max_num_edges: int = 128
  config.max_steps: int = 174
  config.hidden_size: int = 16
  config.allowlist: Optional[List[int]] = None
  config.multidevice: bool = True
  config.restore_checkpoint_dir: Optional[Text] = None
  return config
