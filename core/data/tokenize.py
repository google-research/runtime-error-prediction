"""Generate and load tokenizers for processing source code."""

import fire

from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
from transformers import PreTrainedTokenizerFast


DEFAULT_TOKENIZER_PATH = 'out/tokenizers/default.json'


def generate_tokenizer(path=DEFAULT_TOKENIZER_PATH):
  tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
  trainer = BpeTrainer(special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"])

  tokenizer.pre_tokenizer = Whitespace()
  files = [
      'data/handcrafted-10/122_A_1122406-original.txt',
      'data/handcrafted-10/427_E_6564337-original.txt',
      'data/handcrafted-10/525_A_11774226-original.txt',
      'data/handcrafted-10/7_A_33250-original.txt',
      'data/handcrafted-10/186_B_1655897-original.txt',
      'data/handcrafted-10/432_B_9339909-original.txt',
      'data/handcrafted-10/611_B_15111551-original.txt',
      'data/handcrafted-10/306_A_3703719-original.txt',
      'data/handcrafted-10/514_C_10082646-original.txt',
      'data/handcrafted-10/658_A_17342467-original.txt',
  ]
  tokenizer.train(files, trainer)
  tokenizer.save(path)
  return tokenizer


def load_tokenizer(path=DEFAULT_TOKENIZER_PATH):
  return PreTrainedTokenizerFast(tokenizer_file=path)


if __name__ == '__main__':
  fire.Fire()