from core.data import codenet
from core.data import process
from core.data import tokenize

import fire

DEFAULT_TOKENIZER_PATH = 'out/tokenizers/full.json'


def generate_tokenizer(path=DEFAULT_TOKENIZER_PATH, max_files=None):
  files = []  
  for problem_id, submission_id in codenet.get_all_problem_and_submission_ids():
    python_path = codenet.get_python_path(problem_id, submission_id)
    files.append(python_path)
    if max_files and len(files) >= max_files:
      break
  return tokenize.generate_tokenizer(path=path, files=files)


def process_codenet(tokenizer_path=DEFAULT_TOKENIZER_PATH):
  """Makes RuntimeErrorProblem objects per submission using the tokenizer."""
  tokenizer = tokenize.load_tokenizer(path=tokenizer_path)

  count = 0
  for problem_id, submission_id in codenet.get_all_problem_and_submission_ids():
    python_path = codenet.get_python_path(problem_id, submission_id)
    with open(python_path, 'r') as f:
      source = f.read()
      target = python_path

    try:
      raw = process.make_runtimeerrorproblem(source, target, tokenizer=tokenizer)
    except SyntaxError:
      print(f'SyntaxError: {python_path}')
    except IndexError:
      print(f'IndexError: {python_path}')

    count += 1
    if count % 10 == 0:
      print(count)


def process_codenet_raw(max_files=None):
  """For debugging purposes, makes RawRuntimeErrorProblem objects per submission."""
  count = 0
  for problem_id, submission_id in codenet.get_all_problem_and_submission_ids():
    python_path = codenet.get_python_path(problem_id, submission_id)
    with open(python_path, 'r') as f:
      source = f.read()
      target = python_path

    try:
      raw = process.make_rawruntimeerrorproblem(source, target)
    except SyntaxError:
      print(f'SyntaxError: {python_path}')
    except IndexError:
      print(f'IndexError: {python_path}')

    count += 1
    if max_files and count >= max_files:
      break


if __name__ == '__main__':
  fire.Fire()
