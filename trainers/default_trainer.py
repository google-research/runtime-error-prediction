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
"""Jax runner binary for the Learned Interpreters project."""

import os
import dataclasses
from typing import Any

import fire
from flax import linen as nn
from flax.training import train_state
import jax
import jax.numpy as jnp
import ml_collections
import optax
import tensorflow_datasets as tfds

from core.data import codenet_paths
from core.data import data_io
from core.data import error_kinds
from core.models.ipagnn import encoder
from core.models.ipagnn import ipagnn
from core.models.ipagnn import spans
from third_party.flax_examples import transformer_modules

from absl import app, flags, logging
from flax.metrics import tensorboard
from flax.training import checkpoints, early_stopping
from ml_collections.config_flags import config_flags

from core.data import error_kinds
from evaluators import default_evaluator
from lib import misc_utils, setup
from models import models_lib

NUM_CLASSES = error_kinds.NUM_CLASSES
DEFAULT_DATASET_PATH = codenet_paths.DEFAULT_DATASET_PATH

# @jax.jit
def train_step(state, batch, config):
  """The on-device part of a train step."""
  model = models_lib.ModelFactory()(config.model.name)(config)

  new_rng, dropout_rng = jax.random.split(state.rng, 2)
  state = dataclasses.replace(state, rng=new_rng)

  def loss_fn(params):
  logits = model.apply(
    {'params': params},
    batch,
    rngs={'dropout': dropout_rng}
  )
  labels = jax.nn.one_hot(jnp.squeeze(batch['target'], axis=-1), NUM_CLASSES)
  losses = optax.softmax_cross_entropy(
    logits=logits,
    labels=labels)
  loss = jnp.mean(losses)
  return loss, {
    'logits': logits,
  }

  grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
  (loss, aux), grads = grad_fn(state.params)
  state = state.apply_gradients(grads=grads)
  # TODO(dbieber): Optionally compute on-device metrics here.
  return state, {
    'logits': aux['logits'],
    'loss': loss,
  }

# @jax.jit
# def train_step(state, batch, config):
#     """The on-device part of a train step."""
#     model = models_lib.ModelFactory()(config.model.name)()

#     def loss_fn(params):
#         logits = model.apply(
#             {"params": params},
#             batch,
#             config,
#         )
#         loss = jnp.mean(
#             optax.softmax_cross_entropy(
#                 logits=logits, labels=jax.nn.one_hot(batch["target"], NUM_CLASSES)
#             )
#         )
#         return loss, {"logits": logits, "loss": loss}

#     grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
#     (_, aux), grads = grad_fn(state.params)
#     state = state.apply_gradients(grads=grads)
#     return state, {
#         "logits": aux["logits"],
#         "loss": aux["loss"],
#     }


def trainer(config):
  train_state, train_dataset, eval_dataset = setup.setup(config)
  iter_id = 0
  # TODO (rishab): store the state of the early stopping.
  es = early_stopping.EarlyStopping(
    min_delta=config.runner.early_stopping_delta,
    patience=config.runner.early_stopping_threshold,
  )
  summary_writer = tensorboard.SummaryWriter(config.checkpoint.path)
  summary_writer.hparams(config.to_dict())

  logging.info("Starting the training loop.")

  for step, batch in enumerate(tfds.as_numpy(train_dataset)):
    train_state, aux = train_step(train_state, batch, config)

    if iter_id % config.logging.save_freq == 0:
      misc_utils.save_checkpoint(train_state, config.checkpoint.path)

    if iter_id % config.eval_steps == 0:
      if eval_dataset is None:
        logging.info(
          "Validation dataset unspecified. Using train dataset for evaluation."
        )
        eval_loss, eval_classification_score = default_evaluator.evaluate(
          train_dataset, train_state, config
        )
      else:
        eval_loss, eval_classification_score = default_evaluator.evaluate(
          eval_dataset, train_state, config
        )
      logging.info(
        f"Validation loss: {eval_loss}\n Validation {config.eval_metric}: {eval_classification_score}"
      )
      (
        _,
        batch_loss,
        batch_classification_score,
      ) = default_evaluator.evaluate_batch(batch, train_state, config)
      summary_writer.scalar("train_loss", batch_loss, iter_id)
      summary_writer.scalar("train_metric", batch_classification_score, iter_id)
      summary_writer.scalar("eval_loss", eval_loss, iter_id)
      summary_writer.scalar("eval_metric", eval_classification_score, iter_id)

      did_improve, es = es.update(-1 * eval_loss)
      if es.should_stop:
        logging.info("Early stopping triggered.")
        break

    iter_id += 1
