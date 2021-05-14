"""Tokenize source."""

from python_graphs import control_flow
from transformers import AutoTokenizer

import fire


def filepath_to_features(filepath):
  with open(filepath, 'r') as f:
    source = f.read()
  return source_to_features(source)


def source_to_features(source):
  tokenizer = AutoTokenizer.from_pretrained('bert-base-cased')
  encoded_input = tokenizer(source)
  return {
      'tokens': encoded_input.data['input_ids'],
  }


if __name__ == '__main__':
  fire.Fire()
