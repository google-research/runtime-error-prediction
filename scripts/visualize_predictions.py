"""Visualize model predictions."""

import dataclasses
import os
import re
import subprocess

from absl import app
from absl import flags
from absl import logging

from flax.training import checkpoints
from flax.training import common_utils
import imageio
import jax
import jax.numpy as jnp
import jinja2
from ml_collections.config_flags import config_flags
import numpy as np
import optax
import tensorflow as tf
import tensorflow_datasets as tfds
import termcolor

from core.data import codenet
from core.data import codenet_paths
from core.data import error_kinds
from core.distributed import gcp
from core.data import info as info_lib
from core.data import process
from core.lib import metrics
from core.lib import trainer

DEFAULT_DATASET_PATH = codenet_paths.DEFAULT_DATASET_PATH
DEFAULT_CONFIG_PATH = codenet_paths.DEFAULT_CONFIG_PATH


flags.DEFINE_string('dataset_path', DEFAULT_DATASET_PATH, 'Dataset path.')
flags.DEFINE_string('latex_template_path',
                    'scripts/visualization_template/visualization_template.tex',
                    'LaTeX template path.')
config_flags.DEFINE_config_file(
    name='config', default=DEFAULT_CONFIG_PATH, help_string='Config file.'
)
flags.DEFINE_list('problem_id', 'p02389', 'Specific problem_id to visualize.')
flags.DEFINE_list('submission_id', 's553788892', 'Specific problem_id to visualize.')
FLAGS = flags.FLAGS


@dataclasses.dataclass
class VisualizationInfo:
  """Information for visualizing model predictions."""
  raw: process.RawRuntimeErrorProblem
  source: str
  model_class: str
  raise_in_ipagnn: bool
  target: int
  target_error: str
  logits: jnp.array
  prediction: int
  prediction_error: str
  instruction_pointer: jnp.array
  error_contributions: jnp.array


def get_output_directory(checkpoints_dir, problem_id, submission_id):
  if not checkpoints_dir:
    raise ValueError('checkpoints_dirs must not be empty.')
  checkpoints_parent_dir = os.path.dirname(checkpoints_dir)
  return os.path.join(checkpoints_parent_dir, 'visualizations', problem_id, submission_id)


def get_raise_contribution_at_step(instruction_pointer, raise_decisions, raise_index):
  # instruction_pointer.shape: num_nodes
  # raise_decisions.shape: num_nodes, 2
  # raise_index.shape: scalar.
  p_raise = raise_decisions[:, 0]
  raise_contribution = p_raise * instruction_pointer
  # raise_contribution.shape: num_nodes
  raise_contribution = raise_contribution.at[raise_index].set(0)
  return raise_contribution
get_raise_contribution_at_steps = jax.vmap(get_raise_contribution_at_step, in_axes=(0, 0, None))


def get_raise_contribution(instruction_pointer, raise_decisions, raise_index, step_limit):
  # instruction_pointer.shape: steps, num_nodes
  # raise_decisions.shape: steps, num_nodes, 2
  # raise_index.shape: scalar.
  # step_limit.shape: scalar.
  raise_contributions = get_raise_contribution_at_steps(
      instruction_pointer, raise_decisions, raise_index)
  # raise_contributions.shape: steps, num_nodes
  mask = jnp.arange(instruction_pointer.shape[0]) < step_limit
  # mask.shape: steps
  raise_contributions = jnp.where(mask[:, None], raise_contributions, 0)
  raise_contribution = jnp.sum(raise_contributions, axis=0)
  # raise_contribution.shape: num_nodes
  return raise_contribution
get_raise_contribution_batch = jax.vmap(get_raise_contribution)


def print_spans(raw):
  span_starts = raw.node_span_starts
  span_ends = raw.node_span_ends
  for i, (span_start, span_end) in enumerate(zip(span_starts, span_ends)):
    print(f'Span {i}: {raw.source[span_start:span_end]}')


def get_spans(raw):
  span_starts = raw.node_span_starts
  span_ends = raw.node_span_ends
  for i, (span_start, span_end) in enumerate(zip(span_starts, span_ends)):
    yield raw.source[span_start:span_end]


def set_config(config):
  """This function is hard-coded to load a particular checkpoint.

  It also sets the model part of the config to match the config of that checkpoint.
  Everything related to parameter construction must match.
  """

  config.multidevice = False

  # Exception IPA-GNN, with docstrings
  config.model_class = 'IPAGNN'
  config.batch_size = 32
  config.raise_in_ipagnn = True
  config.optimizer = 'sgd'
  config.hidden_size = 128
  config.span_encoding_method = 'max'
  config.permissive_node_embeddings = False
  config.transformer_emb_dim = 512
  config.transformer_num_heads = 8
  config.transformer_num_layers = 6
  config.transformer_qkv_dim = 512
  config.transformer_mlp_dim = 2048
  config.restore_checkpoint_dir=(
      '/mnt/runtime-error-problems-experiments/experiments/2021-11-02-docstring/33/'
      'E1952,o=sgd,bs=32,lr=0.3,gc=0.5,hs=128,span=max,'
      'tdr=0.1,tadr=0,pe=False,T=default/checkpoints'
  )

  # MIL Transformer, with docstrings
  # config.model_class = 'Transformer'
  # config.mil_pool = 'logsumexp'
  # config.batch_size = 32
  # config.optimizer = 'sgd'
  # config.hidden_size = 64
  # config.span_encoding_method = 'sum'
  # config.permissive_node_embeddings = True
  # config.restore_checkpoint_dir=(
  #   '/mnt/runtime-error-problems-experiments/experiments/2021-12-13-transformer/101/'
  #   'C12383,o=sgd,bs=32,lr=0.3,gc=1,hs=64,span=sum,'
  #   'tdr=0.1,tadr=0.1,pe=False,canh=2,mp=logsumexp,T=default/checkpoints'
  # )

  # Exception IPA-GNN, with FiLM and concat
  config.model_class = 'IPAGNN'
  config.batch_size = 32
  config.raise_in_ipagnn = True
  config.use_film = True
  config.modulate_mode = 'concat'
  config.cross_attention_num_heads = 2

  config.optimizer = 'sgd'
  config.hidden_size = 64
  config.span_encoding_method = 'mean'
  config.transformer_dropout_rate: float = 0.1
  config.transformer_attention_dropout_rate: float = 0.
  config.permissive_node_embeddings = False
  config.mil_pool = 'mean'
  config.compressive_mask_maker = 'default'
  config.raise_decision_offset = -1.0
  config.transformer_emb_dim = 512
  config.transformer_num_heads = 8
  config.transformer_num_layers = 6
  config.transformer_qkv_dim = 512
  config.transformer_mlp_dim = 2048
  config.restore_checkpoint_dir=(
      '/mnt/runtime-error-problems-experiments/experiments/2021-12-29-film-concat/140/'
      'EF14315,o=sgd,bs=32,lr=0.03,gc=0,hs=64,span=mean,tdr=0.1,tadr=0,pe=False,canh=2,'
      'mp=mean,cmm=default,rdo=-1.0,T=default,rii=True,uf=True,mm=concat/top-checkpoints'
  )

  # config.model_class = 'IPAGNN'
  # config.batch_size = 32
  # config.raise_in_ipagnn = True
  # config.use_film = True
  # config.modulate_mode = 'concat'
  # config.cross_attention_num_heads = 2

  # config.optimizer = 'sgd'
  # config.hidden_size = 64
  # config.span_encoding_method = 'mean'
  # config.transformer_dropout_rate: float = 0.1
  # config.transformer_attention_dropout_rate: float = 0.
  # config.permissive_node_embeddings = False
  # config.mil_pool = 'mean'
  # config.compressive_mask_maker = 'default'
  # config.raise_decision_offset = -1.0
  # config.transformer_emb_dim = 512
  # config.transformer_num_heads = 8
  # config.transformer_num_layers = 6
  # config.transformer_qkv_dim = 512
  # config.transformer_mlp_dim = 2048
  # # Exception IPA-GNN, with no input
  # config.restore_checkpoint_dir=(
  #     '/mnt/runtime-error-problems-experiments/experiments/2021-12-22-no-input/104/'
  #     'EN3578,o=sgd,bs=32,lr=0.03,gc=0,hs=64,span=mean,tdr=0.1,tadr=0,pe=False,canh=2,'
  #     'mp=mean,T=default/top-checkpoints'
  # )

  return config


def call(args, stdin=None):
  """Uses subprocess to call the command given by the args."""
  shell_str = gcp.as_shell_string(args)
  logging.info(shell_str)
  print(termcolor.colored('RUNNING: ', 'green') + shell_str)
  return subprocess.run(args, stdin=stdin, capture_output=True)


def latex_escape(s: str) -> str:
  """Escapes the given string to be valid in LaTeX."""
  replacements = {
      '&': r'\&',
      '%': r'\%',
      '$': r'\$',
      '#': r'\#',
      '_': r'\_',
      '{': r'\{',
      '}': r'\}',
      '~': r'\textasciitilde{}',
      '^': r'\^{}',
      '\\': r'\textbackslash{}',
      '<': r'\textless{}',
      '>': r'\textgreater{}',
  }
  pattern = re.compile('|'.join(
      re.escape(str(key))
      for key in sorted(replacements.keys(), key=lambda item: -len(item))))
  return pattern.sub(lambda match: replacements[match.group()], s)


def get_model_name(model_class, raise_in_ipagnn):
  if model_class == 'IPAGNN':
    if raise_in_ipagnn:
      return 'Exception IPA-GNN'
    else:
      return 'IPA-GNN'
  else:
    return model_class


def show_latex_predictions(info: VisualizationInfo, latex_template: jinja2.Template):
  raw = info.raw
  spans = tuple(get_spans(raw))
  error_contributions = info.error_contributions
  instruction_pointer = info.instruction_pointer

  latex_table_lines = []
  span_count = len(spans)
  error_contribution_count = info.error_contributions.shape[0]
  if span_count != error_contribution_count:
    print(
      f'Expected span count {span_count} to match error contribution count '
      f'{error_contribution_count}')

  for i, (span,
          error_contribution) in enumerate(zip(spans, error_contributions)):
    escaped_span = latex_escape(span)
    latex_table_lines.append(
      f'\code{{{i}}} & '
      f'\code{{{escaped_span}}} & '
      f'\code{{{error_contribution:0.2f}}}'
    )

  line_separator = ' \\\\ \hdashline\n'
  latex_table_content = line_separator.join(latex_table_lines) + line_separator
  latex_table_content = latex_table_content.strip()
  model_name = get_model_name(info.model_class, info.raise_in_ipagnn)
  rendered = latex_template.render(
      model_class=info.model_class,
      raise_in_ipagnn=info.raise_in_ipagnn,
      target=info.target,
      target_error=info.target_error,
      prediction=info.prediction,
      prediction_error=info.prediction_error,
      source_code=info.source,
      table_contents=latex_table_content)

  config = FLAGS.config
  study_id = config.study_id
  exp_id = config.experiment_id or codenet_paths.make_experiment_id()
  run_id = config.run_id or codenet_paths.make_run_id()
  run_dir = codenet_paths.make_run_dir(study_id, exp_id, run_id)
  checkpoints_dir = codenet_paths.make_checkpoints_path(run_dir)
  output_directory = get_output_directory(config.restore_checkpoint_dir, raw.problem_id, raw.submission_id)
  os.makedirs(output_directory, exist_ok=True)
  latex_file = os.path.join(output_directory, 'viz.tex')
  print(f'latex_file: {latex_file}')
  with open(latex_file, 'w') as f:
    f.write(rendered)
  call(['pdflatex', '-output-directory', output_directory, latex_file])

  np.save('instruction-pointer.npy', instruction_pointer)
  image = metrics.instruction_pointer_to_image(instruction_pointer)
  np.save('viz-instruction-pointer.npy', image)
  imageio.imwrite('viz-instruction-pointer.png', image, format='png')


def create_train_state_from_params(config, rng, model, params, step):
  """Creates initial TrainState. Skips init and uses params."""
  rng, params_, dropout_rng = jax.random.split(rng, 3)
  learning_rate = config.learning_rate
  if config.optimizer == 'sgd':
    tx = optax.sgd(learning_rate)
  elif config.optimizer == 'adam':
    tx = optax.adam(learning_rate)
  else:
    raise ValueError('Unexpected optimizer', config.optimizer)
  # TODO(dbieber): I don't think model.apply is used from here.
  # Instead, it's used from make_loss_fn.
  opt_state = tx.init(params)
  return trainer.TrainState(
      step=step,
      apply_fn=model.apply,
      params=params,
      tx=tx,
      opt_state=opt_state,
      rng=rng,
  )


def restore_checkpoint(config, restore_checkpoint_dir, init_rng, model):
  state_dict = checkpoints.restore_checkpoint(restore_checkpoint_dir, None)
  return create_train_state_from_params(config, init_rng, model, state_dict['params'], state_dict['step'])


def main(argv):
  del argv  # Unused.

  dataset_path = FLAGS.dataset_path
  config = FLAGS.config
  latex_template_path = FLAGS.latex_template_path
  problem_id = FLAGS.problem_id
  submission_id = FLAGS.submission_id
  config = set_config(config)

  jnp.set_printoptions(threshold=config.printoptions_threshold)
  info = info_lib.get_dataset_info(dataset_path, config)
  t = trainer.Trainer(config=config, info=info)

  split = 'valid'
  dataset = t.load_dataset(
      dataset_path=dataset_path, split=split, include_strings=True)

  # Load the prediction visualization jinja2 LaTeX template.
  env = jinja2.Environment(
      loader=jinja2.FileSystemLoader(searchpath='.'),
      autoescape=jinja2.select_autoescape()
  )
  latex_template = env.get_template(FLAGS.latex_template_path)

  # Initialize / load the model state.
  rng = jax.random.PRNGKey(0)
  rng, init_rng = jax.random.split(rng)
  model = t.make_model(deterministic=False)
  state = t.create_train_state(init_rng, model)
  if config.restore_checkpoint_dir:
    # state = checkpoints.restore_checkpoint(config.restore_checkpoint_dir, state)
    state = restore_checkpoint(config, config.restore_checkpoint_dir, init_rng, model)
    print('Checkpoint loading done')

  def filter_function(x):
    return tf.logical_and(tf.reduce_any(tf.equal(x['problem_id'], problem_id)),
                          tf.reduce_any(tf.equal(x['submission_id'], submission_id)))

  if problem_id and submission_id:
    print(f'Filtering for problem_id {problem_id} and submission_id {submission_id}')
    dataset = dataset.filter(filter_function)
    print('filtered dataset')
    print(dataset)

  # print('numpy_dataset')
  # import timeit
  # start_time = timeit.default_timer()
  # numpy_dataset = list(tfds.as_numpy(filtered_dataset))
  # duration = timeit.default_timer() - start_time
  # print('Duration:', duration)
  # print(numpy_dataset)
  # for batch in tfds.as_numpy(filtered_dataset):
  #   print(batch)

  train_step = t.make_train_step()
  for batch in tfds.as_numpy(dataset):
  # for batch in tfds.as_numpy(filtered_dataset):
    assert not config.multidevice
    # We do not allow multidevice in this script.
    # if config.multidevice:
    #   batch = common_utils.shard(batch)
    problem_ids = batch.pop('problem_id')
    submission_ids = batch.pop('submission_id')
    state, aux = train_step(state, batch)

    instruction_pointer = aux['instruction_pointer_orig']
    # instruction_pointer.shape: steps, batch_size, num_nodes
    instruction_pointer = jnp.transpose(instruction_pointer, [1, 0, 2])
    # instruction_pointer.shape: batch_size, steps, num_nodes
    exit_index = batch['exit_index']
    raise_index = exit_index + 1
    raise_decisions = aux['raise_decisions']
    # raise_decisions.shape: steps, batch_size, num_nodes, 2
    raise_decisions = jnp.transpose(raise_decisions, [1, 0, 2, 3])
    # raise_decisions.shape: batch_size, steps, num_nodes, 2
    contributions = get_raise_contribution_batch(instruction_pointer, raise_decisions, raise_index, batch['step_limit'])
    # contributions.shape: batch_size, num_nodes

    for index, (problem_id, submission_id, contribution) \
        in enumerate(zip(problem_ids, submission_ids, contributions)):
      problem_id = problem_id[0].decode('utf-8')
      submission_id = submission_id[0].decode('utf-8')
      python_path = codenet.get_python_path(problem_id, submission_id)
      r_index = int(raise_index[index])
      num_nodes = int(raise_index[index]) + 1
      target = int(batch['target'][index])
      target_error = error_kinds.to_error(target)
      logits = aux['logits'][index]
      prediction = int(jnp.argmax(logits))
      prediction_error = error_kinds.to_error(prediction)
      step_limit = batch['step_limit'][index, 0]
      instruction_pointer_single = instruction_pointer[index]
      instruction_pointer_single_trim = instruction_pointer_single[:step_limit + 1, :num_nodes].T
      # instruction_pointer_single_trim.shape: num_nodes, timesteps

      total_contribution = jnp.sum(contribution)
      actual_value = instruction_pointer[index, -1, r_index]
      max_contributor = int(jnp.argmax(contribution))
      max_contribution = contribution[max_contributor]

      # Not all submissions are in the copy of the dataset in gs://project-codenet-data.
      # So we only visualize those that are in the copy.
      if not os.path.exists(python_path):
        print(f'Submission path not found: {python_path}')
        continue

      with open(python_path, 'r') as f:
        source = f.read()
      error_lineno = codenet.get_error_lineno(problem_id, submission_id)
      raw = process.make_rawruntimeerrorproblem(
          source, target,
          target_lineno=error_lineno, problem_id=problem_id, submission_id=submission_id)

      # Visualize the data.
      print('---')
      print(f'Problem: {problem_id} {submission_id} ({split})')
      print(f'Batch index: {index}')
      print(f'Target: {target} ({target_error})')
      print(f'Prediction: {prediction} ({prediction_error})')
      print(f'Logits: {logits}')
      print()
      print(source.strip() + '\n')
      print_spans(raw)
      print(contribution[:num_nodes])
      print(f'Main contributor: Node {max_contributor} ({max_contribution})')
      print(f'Total contribution: {total_contribution} (Actual: {actual_value})')

      if error_lineno:
        nodes_at_error = process.get_nodes_at_lineno(raw, error_lineno)
        print(f'Error lineno: {error_lineno} (nodes {nodes_at_error})')
        print(source.split('\n')[error_lineno - 1])  # -1 for line index.

      visualization_info = VisualizationInfo(
          raw=raw,
          source=source.strip(),
          model_class=config.model_class,
          raise_in_ipagnn=config.raise_in_ipagnn,
          target=target,
          target_error=target_error,
          logits=logits,
          prediction=prediction,
          prediction_error=prediction_error,
          instruction_pointer=instruction_pointer_single_trim,
          error_contributions=contribution[:num_nodes])

      show_latex_predictions(info=visualization_info, latex_template=latex_template)

      # Wait for the user to press enter, then continue visualizing.
      input()


if __name__ == '__main__':
  app.run(main)
