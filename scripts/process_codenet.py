from core.data import codenet
from core.data import process

import fire

def process_codenet():
  count = 0
  for problem_id in codenet.get_all_problem_ids():
    for submission_id in codenet.get_all_submission_ids(problem_id):
      python_path = codenet.get_python_path(problem_id, submission_id)
      count += 1
      if count % 1000 == 0:
        print(count)


if __name__ == '__main__':
  fire.Fire()
