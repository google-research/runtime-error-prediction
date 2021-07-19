from core.data import codenet

import fire


def find_all_runtime_error_resolutions():
  """Finds runtime error resolutions for all problems."""
  for problem_id in codenet.get_all_problem_ids():
    for (_, user_id, runtime_error_submission, accepted_submission) in find_runtime_error_resolutions(problem_id):
      yield problem_id, user_id, runtime_error_submission, accepted_submission


def find_runtime_error_resolutions(problem_id):
  """Finds runtime error resolutions for a particular problem."""
  metadata = codenet.get_problem_metadata(problem_id)
  accepted = {}
  runtime_error = {}
  for submission_id, submission_metadata in metadata.items():
    user_id = submission_metadata['user_id']
    if submission_metadata['status'] == 'Accepted':
      accepted[user_id] = submission_id
    if submission_metadata['status'] == 'Runtime Error':
      runtime_error[user_id] = submission_id
  user_ids = set(accepted) & set(runtime_error)
  for user_id in user_ids:
    yield (problem_id, user_id, runtime_error[user_id], accepted[user_id])


def find_runtime_error_submissions(problem_id):
  """Finds runtime error submissions for a particular problem."""
  metadata = codenet.get_problem_metadata(problem_id)
  for submission_id, submission_metadata in metadata.items():
    if submission_metadata['status'] == 'Runtime Error':
      yield submission_id

def find_submissions_by_user(problem_id, user_id):
  """Finds all submissions for a problem by a particular user."""
  metadata = codenet.get_problem_metadata(problem_id)
  for submission_id, submission_metadata in metadata.items():
    if submission_metadata['user_id'] == user_id:
      yield submission_id


def count_all_sessions():
  """Counts the total number of sessions across the CodeNet dataset.

  A session is a unique (problem_id, user_id) pair. It corresponds to one or
  more submissions from a single user on a single problem.

  A session is termed a "runtime error resolution session" if the user submits
  a submission with a runtime error, eventually followed by an Accepted
  submission.

  Returns:
    (int) The total number of sessions in the dataset.
  """
  total = 0
  for problem_id in codenet.get_all_problem_ids():
    total += count_sessions(problem_id)
  return total


def count_sessions(problem_id):
  """Finds runtime error resolutions for a particular problem."""
  metadata = codenet.get_problem_metadata(problem_id)
  user_ids = set()
  for submission_id, submission_metadata in metadata.items():
    user_id = submission_metadata['user_id']
    user_ids.add(user_id)
  return len(user_ids)


if __name__ == '__main__':
  fire.Fire()
