# coding=utf-8
# Copyright 2021 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Lint as: python3

import ml_collections

from core.data import codenet_paths, data_io

DEFAULT_DATASET_PATH = codenet_paths.DEFAULT_DATASET_PATH
Config = ml_collections.ConfigDict


def default_config():
    """Gets the default config for the interpreter model."""
    config = Config()
    config.overrides = ""
    config.seed_id = 0
    config.debug = False

    config.setup = Config()
    config.setup.setup_dataset = True
    config.setup.setup_model = True

    config.logging = Config()
    config.logging.summary_freq = 500
    config.logging.save_freq = 2

    config.runner = Config()
    config.runner.mode = "train"
    config.runner.method = "supervised"
    config.runner.experiment_kind = "train + eval"
    config.runner.restart_behavior = "restore"  # abort or restore
    config.runner.early_stopping_threshold = 1
    config.runner.early_stopping_delta = 0.0
    config.runner.epochs = 8

    config.checkpoint = Config()
    config.checkpoint.run_dir = ""
    config.checkpoint.path = "checkpoints/"
    config.checkpoint.id = 0

    config.dataset = Config()
    config.dataset.name = "handcrafted-10"
    config.dataset.train_path = DEFAULT_DATASET_PATH
    config.dataset.eval_path = DEFAULT_DATASET_PATH
    config.dataset.version = "default"  # Set to use an explicit dataset version.
    config.dataset.split = "default"
    config.dataset.representation = "code"  # code, trace
    config.dataset.max_length = 10000
    config.dataset.max_tokens = 71
    config.dataset.batch_size = 8
    config.dataset.vocab_size = 20000
    config.dataset.batch = True
    config.dataset.in_memory = False

    config.train = Config()
    config.train.total_steps = 0  # 0 means no limit.

    config.opt = Config()
    config.opt.learning_rate = 0.0003

    # Model configs.
    config.model = Config()
    config.model.name = "StackedLSTMModel"
    config.model.hidden_size = 64

    # Other configs.
    config.eval_name = ""
    config.eval_steps = 2
    # Number of seconds to wait without receiving checkpoint before timing out.
    config.eval_timeout = 30 * 60  # 30 minutes.
    config.eval_metric = "F1-score"

    config.index = 0
    return config


def get_config():
    """Gets the config for the interpreter model."""
    config = default_config()
    config.lock()
    return config
