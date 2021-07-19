from core.data import codenet
from core.data import process

import fire

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
        return


if __name__ == '__main__':
  fire.Fire()
