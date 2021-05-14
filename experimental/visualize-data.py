"""Construct visualizations of program control flow graphs.

Usage:
mkdir out
python2 visualize-data.py run --data_dir=data/handcrafted-10 --out_dir=out
"""

import os

import fire

from python_graphs import control_flow
from python_graphs import control_flow_graphviz


def run(data_dir, out_dir=None):
  for filename in os.listdir(data_dir):
    path = os.path.join(data_dir, filename)
    with open(path, 'r') as f:
      source = f.read()
    graph = control_flow.get_control_flow_graph(source)
    if out_dir:
      out_path = os.path.join(out_dir, filename.split('.')[0] + '.png')
      control_flow_graphviz.render(graph, include_src=source, path=out_path)


if __name__ == '__main__':
  fire.Fire()
