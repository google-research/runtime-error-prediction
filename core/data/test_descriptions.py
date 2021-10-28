"""Tests for descriptions.py."""

import unittest

from core.data import descriptions
from core.data import example_problem_descriptions


class ProcessTest(unittest.TestCase):

  def test_extract_input_description(self):
    self.maxDiff = None

    # Known issues in p00130's input description:
    # * s_i is extracted as si. The `<sub>` tag is lost.
    d = example_problem_descriptions.p00130
    c = descriptions.extract_input_description(d)
    self.assertEqual(
        c,
        r"""
１行目に巡回記録の個数n(n≤ 50)、続くn行に巡回記録iを表す文字列si(1024文字までの半角文字列) が与えられます。
""")

    # Known issues in p01950's input description:
    # * No space is included between the li items in the ul.
    d = example_problem_descriptions.p01950
    c = descriptions.extract_input_description(d)
    self.assertEqual(
        c,
        r"""
The input consists of a single test case formatted as follows.

$N$ $M$
$U_1$ $V_1$
...
$U_M$ $V_M$

The first line consists of two integers $N$ ($2 \leq N \leq 100,000$) and $M$ ($1 \leq M \leq 100,000$), where $N$ is the number of vertices and $M$ is the number of edges in a given undirected graph, respectively. The $i$-th line of the following $M$ lines consists of two integers $U_i$ and $V_i$ ($1 \leq U_i, V_i \leq N$), which means the vertices $U_i$ and $V_i$ are adjacent in the given graph. The vertex 1 is the start vertex, i.e. $start\_vertex$ in the pseudo codes. You can assume that the given graph also meets the following conditions.

The graph has no self-loop, i.e., $U_i \ne V_i$ for all $1 \leq i \leq M$.The graph has no multi-edge, i.e., $\{Ui,Vi\} \ne \{U_j,V_j\}$ for all $1 \leq i < j \leq M$.The graph is connected, i.e., there is at least one path from $U$ to $V$ (and vice versa) for all vertices $1 \leq U, V \leq N$
""")

    # Known issues in p03050's input description:
    # * N/A
    d = example_problem_descriptions.p03050
    c = descriptions.extract_input_description(d)
    self.assertEqual(
        c,
        r"""Input is given from Standard Input in the following format:

N
""")

    # Known issues in p04019's input description:
    # * N/A
    d = example_problem_descriptions.p04019
    c = descriptions.extract_input_description(d)
    self.assertEqual(
        c,
        r"""The input is given from Standard Input in the following format:

S
""")




if __name__ == '__main__':
  unittest.main()
