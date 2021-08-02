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
from absl import logging
from flax import linen as nn
from flax.training import checkpoints, train_state

from core.data import codenet_paths, data_io, error_kinds
from models import models_lib

NUM_CLASSES = error_kinds.NUM_CLASSES
Config = ml_collections.ConfigDict


def restore_checkpoint(state, workdir):
    if os.path.exists(workdir):
        return checkpoints.restore_checkpoint(workdir, state)
    logging.info("No checkpoint found.")
    return state


def seed(seed_id):
    random.seed(seed_id)
    np.random.seed(seed_id)


def create_train_state(rng, model, fake_input, fake_config):
    """Creates initial TrainState."""
    variables = model.init(rng, fake_input, fake_config)
    params = variables["params"]
    tx = optax.sgd(0.03)
    return train_state.TrainState.create(apply_fn=model.apply, params=params, tx=tx)


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
    model = models_factory(config.model.name)()
    return model


def setup(config):
    seed(config.seed_id)
    train_dataset = (
        data_io.load_dataset(config.dataset.train_path)
        .repeat(config.runner.epochs)
        .padded_batch(config.dataset.batch_size)
    )
    eval_dataset = None

    if config.runner.experiment_kind == "train + eval":
        eval_dataset = (
            data_io.load_dataset(config.dataset.eval_path)
            .repeat(config.runner.epochs)
            .padded_batch(config.dataset.batch_size)
        )

    model = create_model(config)

    rng = jax.random.PRNGKey(config.seed_id)
    rng, init_rng = jax.random.split(rng)

    fake_input = create_fake_inputs(
        config.dataset.batch_size,
        config.dataset.max_tokens,
        config.dataset.max_length,
        config.model.name,
    )
    train_state = create_train_state(init_rng, model, fake_input, config)
    train_state = restore_checkpoint(train_state, config.checkpoint.path)
    return train_state, train_dataset, eval_dataset
