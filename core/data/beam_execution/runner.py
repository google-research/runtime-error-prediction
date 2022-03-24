import argparse
import logging
import re

import apache_beam as beam
from apache_beam.io import ReadFromText
from apache_beam.io import WriteToText
from apache_beam.options.pipeline_options import PipelineOptions
from apache_beam.options.pipeline_options import SetupOptions

import fire


def run(**flags):
  """Main entry point; defines and runs the wordcount pipeline."""

  # We use the save_main_session option because one or more DoFn's in this
  # workflow rely on global context (e.g., a module imported at module level).
  save_main_session = True
  pipeline_options = PipelineOptions.from_dictionary(flags)
  pipeline_options.view_as(SetupOptions).save_main_session = save_main_session
  with beam.Pipeline(options=pipeline_options) as p:

    # Read the text file[pattern] into a PCollection.
    lines = p | ReadFromText(flags['input'])

    # Count the occurrences of each word.
    counts = (
        lines
        | 'Split' >> (
            beam.FlatMap(
                lambda x: re.findall(r'[A-Za-z\']+', x)).with_output_types(str))
        | 'PairWithOne' >> beam.Map(lambda x: (x, 1))
        | 'GroupAndSum' >> beam.CombinePerKey(sum))

    # Format the counts into a PCollection of strings.
    def format_result(word_count):
      (word, count) = word_count
      return '%s: %s' % (word, count)

    output = counts | 'Format' >> beam.Map(format_result)

    # Write the output using a "Write" transform that has side effects.
    # pylint: disable=expression-not-assigned
    output | WriteToText(flags['output'])


def run_codenet_submissions(**flags):
  last_problem_id = None
  problem_ids = list(range(4053))
  all_problem_and_submision_ids = list(codenet.get_all_problem_and_submission_ids())

  save_main_session = True
  pipeline_options = PipelineOptions.from_dictionary(flags)
  pipeline_options.view_as(SetupOptions).save_main_session = save_main_session
  with beam.Pipeline(options=pipeline_options) as p:
    _ = (
        p
        | 'ProblemNumbers' >> beam.Create(problem_ids)
        | 'ProblemPaths' >> beam.Map(
            lambda problem_id: os.path.join(codenet_paths.DATA_ROOT, 'data', problem_id))
        | 'SubmissionPaths' >> beam.FlatMap(
            lambda problem_path: os.path.join(codenet_paths.DATA_ROOT, 'data', problem_id))
        | 'Run' >> beam.Map(lambda x: codenet_paths.run_for_errors(*x))
        | 'One' >> beam.Map(lambda x: ('done', 1))
        | 'GroupAndSum' >> beam.CombinePerKey(sum)
        | 'Write' >> WriteToText(flags['output'])
    )


if __name__ == '__main__':
  logging.getLogger().setLevel(logging.INFO)
  fire.Fire(run)
