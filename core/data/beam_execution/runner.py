import json
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
from core.data import error_kinds

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
        | 'Reshuffle' >> beam.Reshuffle()
        | 'Run' >> beam.MapTuple(codenet.run_for_errors)
    )


def _get_statuses(problem_id, submission_id):
  metadata = codenet.get_submission_metadata(problem_id, submission_id)
  codenet_error_status = metadata['status'] in ('Runtime Error', 'Time Limit Exceeded')
  # codenet_error_status: True indicates an error (incl. timeout).

  our_error_kind = codenet.get_submission_error_kind(problem_id, submission_id)
  our_error_status = our_error_kind not in (error_kinds.NO_ERROR, error_kinds.NO_ERROR_WITH_STDERR, error_kinds.NO_DATA)
  # our_error_status: True indicates an error (incl. timeout).
  return codenet_error_status, our_error_status


def _check_matches(codenet_error_status, our_error_status):
  matches = codenet_error_status == our_error_status
  results = [
      ('matches', matches),
      ('raw', (codenet_error_status, our_error_status)),
  ]
  if codenet_error_status:
    results.append(
        ('codenet error', matches),
    )
  else:
    results.append(
        ('no codenet error', matches),
    )
  if our_error_status:
    results.append(
        ('our error', matches),
    )
  else:
    results.append(
        ('no error', matches),
    )
  return results


def run_check_matches(num_problems=4053, **flags):
  problem_ids = [f'p{problem_number:05d}' for problem_number in range(1, num_problems)]

  save_main_session = True
  pipeline_options = PipelineOptions.from_dictionary(flags)
  pipeline_options.view_as(SetupOptions).save_main_session = save_main_session
  with beam.Pipeline(options=pipeline_options) as p:
    statuses = (
        p
        | 'ProblemIds' >> beam.Create(problem_ids)
        | 'SubmissionIds' >> beam.FlatMap(_get_submission_ids)
        | 'Reshuffle' >> beam.Reshuffle()
        | 'Statuses' >> beam.MapTuple(_get_statuses)
        | 'Matches' >> beam.FlatMapTuple(_check_matches)
        | 'Count' >> beam.combiners.Count.PerElement()
        | 'Write' >> beam.io.WriteToText(flags['output'])
    )


def _get_problem_and_submission_ids(ids_file):
  gcsio_client = gcsio.GcsIO()
  with gcsio_client.open(ids_file, 'rb') as f:
    raw = f.read()
    text = raw.decode('utf-8')
    return json.loads(text)


def run_check_matches_new(**flags):
  save_main_session = True
  pipeline_options = PipelineOptions.from_dictionary(flags)
  pipeline_options.view_as(SetupOptions).save_main_session = save_main_session

  ids_files = [
      # 'gs://python-runtime-errors/datasets/project-codenet/2021-12-29/train-ids.json',
      # 'gs://python-runtime-errors/datasets/project-codenet/2021-12-29/valid-ids.json',
      # 'gs://python-runtime-errors/datasets/project-codenet/2021-12-29/test-ids.json',
      'gs://runtime-error-problems-experiments/datasets/project-codenet/2021-12-29-sampled-test/test-ids.json',
  ]

  with beam.Pipeline(options=pipeline_options) as p:
    statuses = (
        p
        | 'IdFiles' >> beam.Create(ids_files)
        | 'SubmissionIds' >> beam.FlatMap(_get_problem_and_submission_ids)
        | 'Reshuffle' >> beam.Reshuffle()
        | 'Statuses' >> beam.MapTuple(_get_statuses)
        | 'Matches' >> beam.FlatMapTuple(_check_matches)
        | 'Count' >> beam.combiners.Count.PerElement()
        | 'Write' >> beam.io.WriteToText(flags['output'])
    )


if __name__ == '__main__':
  logging.getLogger().setLevel(logging.INFO)
  fire.Fire()
