from core.data import codenet
from core.data import process
from core.data import tokenize

import fire

DEFAULT_TOKENIZER_PATH = 'out/tokenizers/full.json'


def generate_tokenizer(path=DEFAULT_TOKENIZER_PATH, max_files=None):
  files = []  
  for problem_id in codenet.get_all_problem_ids():
    for submission_id in codenet.get_all_submission_ids(problem_id):
      python_path = codenet.get_python_path(problem_id, submission_id)
      files.append(python_path)
      if max_files and len(files) >= max_files:
        break
    if max_files and len(files) >= max_files:
      break
  return tokenize.generate_tokenizer(path=path, files=files)


def process_codenet():
  count = 0
  for problem_id in codenet.get_all_problem_ids():
    for submission_id in codenet.get_all_submission_ids(problem_id):
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
      if count % 3000 == 0:
        print(count)


if __name__ == '__main__':
  fire.Fire()
