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

from absl import app
from absl import flags
from absl import logging

import numpy as np
import jax
import jax.numpy as jnp

import optax
from flax.training import checkpoints
from flax.training import early_stopping

from flax.metrics import tensorboard

from ml_collections.config_flags import config_flags

from lib import setup, misc_utils
from evaluators import default_evaluator
from models import models_lib

import tensorflow_datasets as tfds
from core.data import error_kinds

NUM_CLASSES = error_kinds.NUM_CLASSES
DEFAULT_DATA_DIR = 'data'
DEFAULT_CONFIG = 'config/default.py'

# @jax.jit
def train_step(state, batch, config):
  """The on-device part of a train step."""
  model = models_lib.ModelFactory()(config.model.name)(vocab_size=config.dataset.vocab_size, emb_dim=config.model.hidden_size)
  def loss_fn(params):
    logits = model.apply(
        {'params': params},
        batch,
        config,
    )
    loss = jnp.mean(
        optax.softmax_cross_entropy(
            logits=logits,
            labels=jax.nn.one_hot(batch['target'], NUM_CLASSES)))
    return loss, {
        'logits': logits,
        'loss' : loss
    }

  grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
  (_, aux), grads = grad_fn(state.params)
  state = state.apply_gradients(grads=grads)
  return state, {
      'logits': aux['logits'],
      'loss': aux['loss'],
  }

def trainer(config):
  train_state, train_dataset, eval_dataset = setup.setup(config)
  iter_id = 0
  #TODO rishab: store the state of the early stopping.
  es = early_stopping.EarlyStopping(min_delta=config.runner.early_stopping_delta, patience=config.runner.early_stopping_threshold)
  summary_writer = tensorboard.SummaryWriter(config.checkpoint.path)
  summary_writer.hparams(config.to_dict())

  logging.info("Starting the training loop.")

  for batch in tfds.as_numpy(train_dataset):
    train_state, aux = train_step(train_state, batch, config)
    
    if iter_id % config.logging.save_freq == 0:
      misc_utils.save_checkpoint(train_state, config.checkpoint.path)
    
    if iter_id % config.eval_steps == 0:
      if eval_dataset is None:
        logging.info("Validation dataset unspecified. Using train dataset for evaluation.")
        eval_loss, eval_classification_score = default_evaluator.evaluate(train_dataset, train_state, config)
      else:
        eval_loss, eval_classification_score = default_evaluator.evaluate(eval_dataset, train_state, config)
      logging.info(f"Validation loss: {eval_loss}\n Validation {config.eval_metric}: {eval_classification_score}")
      _, batch_loss, batch_classification_score = default_evaluator.evaluate_batch(batch, train_state, config)
      summary_writer.scalar('train_loss', batch_loss, iter_id)
      summary_writer.scalar('train_metric', batch_classification_score, iter_id)
      summary_writer.scalar('eval_loss', eval_loss, iter_id)
      summary_writer.scalar('eval_metric', eval_classification_score, iter_id)

      did_improve, es = es.update(-1*eval_loss)
      if es.should_stop:
        logging.info("Early stopping triggered.")
        break
    
    iter_id+=1
