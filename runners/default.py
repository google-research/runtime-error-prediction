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
from ml_collections.config_flags import config_flags

from lib import setup, evaluation
from models import models_lib

import tensorflow_datasets as tfds
from core.data import error_kinds

NUM_CLASSES = error_kinds.NUM_CLASSES
DEFAULT_DATA_DIR = 'data'
DEFAULT_CONFIG = 'config/default.py'

flags.DEFINE_string('data_dir', DEFAULT_DATA_DIR, 'Where to place the data.')
config_flags.DEFINE_config_file(
  name='config',
  default=DEFAULT_CONFIG,
  help_string='config file')
FLAGS = flags.FLAGS


def main(argv):
  del argv  # Unused.

  data_dir = FLAGS.data_dir
  config = FLAGS.config
  train_state, train_dataset, eval_dataset = setup.setup(config)
  trainer(train_state, train_dataset, eval_dataset, config)


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

def save_checkpoint(state, workdir):
  os.makedirs(workdir, exist_ok=True)
  step = int(state.step)
  checkpoints.save_checkpoint(workdir, state, step, keep=3)

def evaluate(dataset, state, config):
  predictions = []
  ground_truth = []
  loss = []
  model = models_lib.ModelFactory()(config.model.name)(vocab_size=config.dataset.vocab_size, emb_dim=config.model.hidden_size)
  for batch in tfds.as_numpy(dataset):
    logits = model.apply({'params': state.params}, batch, config)
    predictions.append(jnp.argmax(logits, -1))
    ground_truth.append(batch['target'])
    loss.append(jnp.sum(
        optax.softmax_cross_entropy(
            logits=logits,
            labels=jax.nn.one_hot(batch['target'], NUM_CLASSES))))
  predictions = np.array(jnp.concatenate(predictions))
  ground_truth = np.array(jnp.concatenate(ground_truth)).flatten()
  eval_loss = sum(loss)/predictions.shape[0]
  assert predictions.shape[0] == ground_truth.shape[0]
  classification_score = evaluation.evaluate(ground_truth, predictions, config.eval_metric)
  return eval_loss, classification_score

def trainer(train_state, train_dataset, eval_dataset, config):
  #TODO
  iter_id = 0
  logging.info("Starting the training loop.")

  for batch in tfds.as_numpy(train_dataset):
    train_state, aux = train_step(train_state, batch, config)
    
    if iter_id % config.logging.save_freq == 0:
      save_checkpoint(train_state, config.checkpoint.path)
    
    if iter_id % config.eval_steps == 0:
      if eval_dataset is None:
        logging.info("Validation dataset unspecified. Using train dataset for evaluation.")
        eval_loss, eval_f1_score = evaluate(train_dataset, train_state, config)
      else:
        eval_loss, eval_classification_score = evaluate(eval_dataset, train_state, config)
      logging.info(f"Validation loss: {eval_loss}\n Validation {config.eval_metric}: {eval_classification_score}")
    
    iter_id+=1
    

if __name__ == '__main__':
  app.run(main)
