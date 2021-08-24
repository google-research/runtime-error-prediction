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
"""Performs setup tasks for Learned Interpreters binaries."""

import json
import os
import random

import jax
import jax.numpy as jnp
import ml_collections
import numpy as np
import optax

from typing import Any
from absl import logging
from flax import linen as nn
from flax.training import checkpoints, train_state

from core.data import codenet_paths, data_io, error_kinds
from models import models_lib

DEFAULT_DATASET_PATH = codenet_paths.DEFAULT_DATASET_PATH
NUM_CLASSES = error_kinds.NUM_CLASSES
Config = ml_collections.ConfigDict


class TrainState(train_state.TrainState):
  rng: Any


def restore_checkpoint(state, workdir):
  if os.path.exists(workdir):
    return checkpoints.restore_checkpoint(workdir, state)
  logging.info("No checkpoint found.")
  return state


def seed(seed_id):
  random.seed(seed_id)
  np.random.seed(seed_id)


def create_train_state(rng, model, config):
  """Creates initial TrainState."""
  batch_size = config.dataset.batch_size
  max_tokens = config.dataset.max_tokens
  max_num_nodes = config.dataset.max_num_nodes
  max_num_edges = config.dataset.max_num_edges
  fake_input = data_io.get_fake_input(
    batch_size, max_tokens, max_num_nodes, max_num_edges)
  rng, params_rng, dropout_rng = jax.random.split(rng, 3)
  variables = model.init(
    {'params': params_rng, 'dropout': dropout_rng},
    fake_input)
  params = variables['params']
  learning_rate = 0.03
  tx = optax.sgd(learning_rate)
  return TrainState.create(
    apply_fn=model.apply, params=params, tx=tx, rng=rng)


def create_fake_inputs(batch_size, max_tokens, max_length, model_name):
  if model_name == "StackedLSTMModel":
    fake_inputs = {
      "tokens": jnp.ones((batch_size, max_tokens), dtype=jnp.int64),
      "node_token_span_starts": jnp.ones(
        (batch_size, max_length), dtype=jnp.int64
      ),
      "node_token_span_ends": jnp.ones((batch_size, max_length), dtype=jnp.int64),
    }
    return fake_inputs
  raise ValueError(f"{model_name} not implemented.")


def create_model(config):
  models_factory = models_lib.ModelFactory()
  model = models_factory(config.model.name)(config)
  return model

# Taken From David's code
def load_dataset(dataset_path=DEFAULT_DATASET_PATH, split='train', config=None):
  epochs = config.runner.epochs
  batch_size = config.dataset.batch_size
  max_tokens = config.dataset.max_tokens
  max_num_nodes = config.dataset.max_num_nodes
  max_num_edges = config.dataset.max_num_edges
  max_steps = config.max_steps
  padded_shapes = data_io.get_padded_shapes(
    max_tokens, max_num_nodes, max_num_edges)
  filter_fn = data_io.make_filter(
    max_tokens, max_num_nodes, max_num_edges, max_steps, allowlist=None)
  return (
    data_io.load_dataset(dataset_path, split=split)
    .filter(filter_fn)
    .repeat(epochs)
    .padded_batch(batch_size, padded_shapes=padded_shapes)
  )

def setup(config):
  seed(config.seed_id)
  train_dataset = load_dataset(config.dataset.train_path, config=config)
  eval_dataset = None

  if config.runner.experiment_kind == "train + eval":
    eval_dataset = load_dataset(config.dataset.eval_path, split="valid", config=config)

  model = create_model(config)

  rng = jax.random.PRNGKey(config.seed_id)
  rng, init_rng = jax.random.split(rng)

  train_state = create_train_state(init_rng, model, config)
  train_state = restore_checkpoint(train_state, config.checkpoint.path)
  return train_state, train_dataset, eval_dataset
