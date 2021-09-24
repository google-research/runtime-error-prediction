import enum
from typing import List, Optional, Text, Tuple

import ml_collections

Config = ml_collections.ConfigDict


class EvaluationMetric(enum.Enum):
  """Evaluation metric kinds."""
  ACCURACY = 'accuracy'
  F1_SCORE = 'f1_score'
  CONFUSION_MATRIX = 'confusion_matrix'

  @staticmethod
  def all_metric_names() -> Tuple[str]:
    return tuple(x.value for x in EvaluationMetric)


def default_config():
  """The default config."""
  config = Config()

  # Trainer configs
  config.multidevice: bool = True
  config.restore_checkpoint_dir: Optional[Text] = ''
  config.study_id: Optional[Text] = ''  # A study is a way of organizing experiments.
  config.experiment_id: Optional[Text] = ''  # An experiment is launched by a single command, may have multiple runs.
  config.run_id: Optional[Text] = ''  # A run is a single trainer run with a single set of hparams. run_id should identify hparams.
  config.notes: Optional[Text] = ''  # Any notes to record about the run.

  # Training configs
  config.learning_rate = 0.03
  config.grad_clip_value: float = 0.0  # 0 means no clipping.

  # Model HParams
  config.model_class: Text = 'IPAGNN'  # IPAGNN, Transformer
  config.raise_in_ipagnn: bool = False
  config.rnn_layers = 2
  config.hidden_size: int = 16
  config.span_encoding_method = 'first'  # first, mean, max, sum

  # Dataset filtering and configs
  config.epochs: Optional[int] = 0
  config.batch_size: int = 128
  config.allowlist: Optional[List[int]] = None
  config.max_tokens: int = 512
  config.max_num_nodes: int = 128
  config.max_num_edges: int = 128
  config.max_steps: int = 174

  # Transformer configs
  config.transformer_emb_dim: int = 512
  config.transformer_num_heads: int = 8
  config.transformer_num_layers: int = 6
  config.transformer_qkv_dim: int = 512
  config.transformer_mlp_dim: int = 2048
  config.transformer_dropout_rate: float = 0.1
  config.transformer_attention_dropout_rate: float = 0.1

  # Runner configs
  config.eval_freq = 10000
  config.save_freq = 5000
  config.eval_metric_names: Tuple[str] = EvaluationMetric.all_metric_names()
  config.eval_subsample = 1.0
  config.eval_max_batches = 30

  # Logging
  config.printoptions_threshold = 256

  config.early_stopping_on = False
  config.early_stopping_delta = 0.001
  config.early_stopping_threshold = 4
  return config


def get_config():
  """Gets the config."""
  config = default_config()
  config.lock()
  return config


def get_test_config():
  config = default_config()
  config.hidden_size = 10
  config.span_encoding_method = 'first'
  config.max_tokens = 64

  config.transformer_emb_dim = 32
  config.transformer_num_heads = 4
  config.transformer_num_layers = 2
  config.transformer_qkv_dim = 32
  config.transformer_mlp_dim = 64
  config.transformer_dropout_rate = 0.1
  config.transformer_attention_dropout_rate = 0.1
  return config
