import logging
import os
import re

import apache_beam as beam
from apache_beam.io import ReadFromText
from apache_beam.io import WriteToText
from apache_beam.options.pipeline_options import PipelineOptions
from apache_beam.options.pipeline_options import SetupOptions
from apache_beam.io.gcp import gcsio

from core.data import codenet
from core.data import codenet_paths

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


def _get_submission_ids(problem_id):
  gcs_data_root = codenet_paths.DATA_ROOT.replace('/mnt/', 'gs://')
  problem_dir = os.path.join(gcs_data_root, 'data', problem_id, 'Python')
  return [
      (problem_id, _get_submission_id(submission_path))
      for submission_path in gcsio.GcsIO().list_prefix(problem_dir).keys()
  ]


def _get_submission_id(submission_path):
  return submission_path.split('/')[-1].split('.')[0]


def run_codenet_submissions(**flags):
  problem_ids = [f'p{problem_number:05d}' for problem_number in range(4053)]

  save_main_session = True
  pipeline_options = PipelineOptions.from_dictionary(flags)
  pipeline_options.view_as(SetupOptions).save_main_session = save_main_session
  with beam.Pipeline(options=pipeline_options) as p:
    _ = (
        p
        | 'ProblemIds' >> beam.Create(problem_ids)
        | 'SubmissionIds' >> beam.FlatMap(_get_submission_ids)
        | 'Run' >> beam.MapTuple(codenet.run_for_errors)
        | 'One' >> beam.Map(lambda x: ('done', 1))
        | 'GroupAndSum' >> beam.CombinePerKey(sum)
        | 'Write' >> WriteToText(flags['output'])
    )


if __name__ == '__main__':
  logging.getLogger().setLevel(logging.INFO)
  fire.Fire(run_codenet_submissions)
