from core.data import codenet

import fire


def find_all_runtime_error_resolutions():
  """Finds runtime error resolutions for all problems."""
  for problem_id in codenet.get_all_problem_ids():
    for (_, user_id, runtime_error[user_id], accepted[user_id]) in find_runtime_error_resolutions(problem_id):
      yield problem_id, user_id, runtime_error[user_id], accepted[user_id]


def find_runtime_error_resolutions(problem_id):
  """Finds runtime error resolutions for a particular problem."""
  metadata = codenet.get_problem_metadata(problem_id)
  accepted = {}
  runtime_error = {}
  for submission_id, submission_metadata in metadata.items():
    user_id = metadata['user_id']
    if submission_metadata['status'] == 'Accepted':
      accepted[user_id] = submission_id
    if submission_metadata['status'] == 'Runtime Error':
      runtime_error[user_id] = submission_id
  user_ids = set(accepted) & set(runtime_error)
  for user_id in user_ids:
    yield (problem_id, user_id, runtime_error[user_id], accepted[user_id])


if __name__ == '__main__':
  fire.Fire()
