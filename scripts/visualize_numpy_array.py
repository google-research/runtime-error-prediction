"""Visualize a pickled NumPy array."""

import fire
import imageio
import os
import subprocess

from absl import app
from absl import flags

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import termcolor

from core.distributed import gcp
from core.lib import metrics


def call(args, stdin=None):
  """Uses subprocess to call the command given by the args."""
  shell_str = gcp.as_shell_string(args)
  print(termcolor.colored('RUNNING: ', 'green') + shell_str)
  return subprocess.run(args, stdin=stdin, capture_output=True)


def load_array(input_path):
  array = np.load(input_path)
  print(array)


def instruction_pointer_to_image(instruction_pointer):
  """Converts the given instruction pointer array to an image."""
  # Slice the trailing dimensions where there are no changes.
  np.set_printoptions(precision=3, suppress=True)
  # print(instruction_pointer)
  instruction_pointer_raise_node = instruction_pointer[-1]
  instruction_pointer_raise_node_last_value = instruction_pointer[-1]
  print(instruction_pointer_raise_node)
  print(np.isclose(instruction_pointer_raise_node, 1))
  instruction_pointer_raise_node_is_one = np.isclose(instruction_pointer_raise_node, 1)
  counts = np.cumsum(instruction_pointer_raise_node_is_one)
  idx = np.searchsorted(counts, 6)
  # instruction_pointer_trimmed = instruction_pointer[:, :idx]
  instruction_pointer_trimmed = instruction_pointer[:, :24]
  instruction_pointer_figure = make_figure(
      # data=instruction_pointer_trimmed,
      data=instruction_pointer_trimmed,
      title='Instruction Pointer',
      xlabel='Timestep',
      ylabel='Node')
  # return metrics.figure_to_image(instruction_pointer_figure)
  return instruction_pointer_figure


def make_figure(*,
                data,
                title,
                xlabel,
                ylabel,
                interpolation='nearest',
                **kwargs):
  """"Creates a matplotlib plot from the given data."""
  fig = plt.figure()
  plt.imshow(data, cmap='gist_gray', interpolation=interpolation, **kwargs)
  # Hide y-axis labels.
  plt.tick_params(axis='y', which='both', labelsize=0)
  # Add colorbar.
  # colorbar = plt.colorbar(orientation='vertical', ticks=[0., 1.])
  # colorbar.ax.set_yticklabels(['0.0', '1.0'])
  return fig


def load_instruction_pointer(input_path, output_path=None):
  # plt.style.use('dark_background')
  instruction_pointer_array = np.load(input_path)
  print(instruction_pointer_array.shape)
  figure = instruction_pointer_to_image(instruction_pointer_array)
  # np.save('viz-instruction-pointer.npy', image)
  base_filename = os.path.splitext(input_path)[0]
  if output_path:
    pdf_output_path = output_path
  else:
    pdf_output_path = f'{base_filename}.pdf'
  image_output_path = f'{base_filename}.png'
  figure.savefig(pdf_output_path, bbox_inches='tight', transparent=True, format='pdf', dpi=300)

  image = metrics.figure_to_image(figure)
  imageio.imwrite(image_output_path, image, format='png')
  print(f'Saved files to {pdf_output_path} and {image_output_path}.')
  call(['open', pdf_output_path])
  # read_image = plt.imread('viz-instruction-pointer.png')
  # plt.imshow(read_image)
  # plt.show()
  # vv.imshow(image)
  # imageio.imread


if __name__ == '__main__':
  fire.Fire()
