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

"""Generate and load tokenizers for processing source code."""

import os

import fire

from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
from transformers import PreTrainedTokenizerFast

from core.data import codenet_paths

DEFAULT_TOKENIZER_PATH = codenet_paths.DEFAULT_TOKENIZER_PATH
SAMPLE_FILES = [
    'datasets/handcrafted-10/122_A_1122406.txt',
    'datasets/handcrafted-10/427_E_6564337.txt',
    'datasets/handcrafted-10/525_A_11774226.txt',
    'datasets/handcrafted-10/7_A_33250.txt',
    'datasets/handcrafted-10/186_B_1655897.txt',
    'datasets/handcrafted-10/432_B_9339909.txt',
    'datasets/handcrafted-10/611_B_15111551.txt',
    'datasets/handcrafted-10/306_A_3703719.txt',
    'datasets/handcrafted-10/514_C_10082646.txt',
    'datasets/handcrafted-10/658_A_17342467.txt',
]


def generate_tokenizer(path=DEFAULT_TOKENIZER_PATH, files=SAMPLE_FILES):
  tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
  trainer = BpeTrainer(special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"])

  # Make the tokenizers directory.
  directory = os.path.dirname(path)
  os.makedirs(directory, exist_ok=True)

  tokenizer.pre_tokenizer = Whitespace()
  tokenizer.train(files, trainer)
  tokenizer.save(path)
  return tokenizer


def load_tokenizer(path=DEFAULT_TOKENIZER_PATH):
  return PreTrainedTokenizerFast(tokenizer_file=path)


if __name__ == '__main__':
  fire.Fire()
