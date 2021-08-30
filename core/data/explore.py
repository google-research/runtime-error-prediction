import fire

from core.data import codenet_paths
from core.data import process
from core.data import tokenization


DEFAULT_TOKENIZER_PATH = codenet_paths.DEFAULT_TOKENIZER_PATH


def get_spans(problem_id, submission_id, tokenizer_path=DEFAULT_TOKENIZER_PATH):
  tokenizer = tokenization.load_tokenizer(path=tokenizer_path)
  source, target = process.get_source_and_target_for_submission(problem_id, submission_id)

  problem = make_runtimeerrorproblem(
      source, target, tokenizer=tokenizer, problem_id=problem_id, submission_id=submission_id)

  print(source)
  tokens = tokenizer.convert_ids_to_tokens(problem.tokens)
  for span_index, (span_start, span_end) in enumerate(zip(problem.node_token_span_starts, problem.node_token_span_ends)):
    print(f'Span {span_index}: {tokens[span_start:span_end + 1]}')


if __name__ == '__main__':
  fire.Fire()
