"""Tests for descriptions.py."""

import unittest

from core.data import descriptions
from core.data import example_problem_descriptions


class ProcessTest(unittest.TestCase):

  def test_extract_input_description(self):
    self.maxDiff = None

    d = example_problem_descriptions.p00022
    c = descriptions.extract_input_description(d)
    self.assertEqual(
        c,
        r"""The input consists of multiple datasets. Each data set consists of:

n
a1
a2
.
.
an

You can assume that 1 ≤ n ≤ 5000 and -100000 ≤ ai ≤ 100000.

The input end with a line consisting of a single 0.""")

    d = example_problem_descriptions.p00023
    c = descriptions.extract_input_description(d)
    self.assertEqual(
        c,
        r"""The input consists of multiple datasets. The first line consists of an integer $N$ ($N \leq 50$), the number of datasets. There will be $N$ lines where each line represents each dataset. Each data set consists of real numbers:

$x_a$ $y_a$ $r_a$ $x_b$ $y_b$ $r_b$""")

    # Known issues in p00130's input description:
    # * s_i is extracted as si. The `<sub>` tag is lost.
    d = example_problem_descriptions.p00130
    c = descriptions.extract_input_description(d)
    self.assertEqual(
        c,
        r"""１行目に巡回記録の個数 n (n ≤ 50)、続く n 行に巡回記録 i を表す文字列 si (1024文字までの半角文字列) が与えられます。""")

    d = example_problem_descriptions.p00352
    c = descriptions.extract_input_description(d)
    self.assertEqual(
        c,
        r"""The input is given in the following format.

a b

A line of data is given that contains two values of money: a (1000 ≤ a ≤ 50000) for Alice and b (1000 ≤ b ≤ 50000) for Brown.""")

    # * There's no 'Input' section. This task is in Japanese.
    d = example_problem_descriptions.p00569
    c = descriptions.extract_input_description(d)
    self.assertEqual(c, '')

    d = example_problem_descriptions.p00729_abbr
    c = descriptions.extract_input_description(d)
    self.assertEqual(c, r"""Example
Unclosed item""")

    d = example_problem_descriptions.p01950
    c = descriptions.extract_input_description(d)
    self.assertEqual(
        c,
        r"""The input consists of a single test case formatted as follows.

$N$ $M$
$U_1$ $V_1$
...
$U_M$ $V_M$

The first line consists of two integers $N$ ($2 \leq N \leq 100,000$) and $M$ ($1 \leq M \leq 100,000$), where $N$ is the number of vertices and $M$ is the number of edges in a given undirected graph, respectively. The $i$-th line of the following $M$ lines consists of two integers $U_i$ and $V_i$ ($1 \leq U_i, V_i \leq N$), which means the vertices $U_i$ and $V_i$ are adjacent in the given graph. The vertex 1 is the start vertex, i.e. $start\_vertex$ in the pseudo codes. You can assume that the given graph also meets the following conditions.

The graph has no self-loop, i.e., $U_i \ne V_i$ for all $1 \leq i \leq M$.
The graph has no multi-edge, i.e., $\{Ui,Vi\} \ne \{U_j,V_j\}$ for all $1 \leq i < j \leq M$.
The graph is connected, i.e., there is at least one path from $U$ to $V$ (and vice versa) for all vertices $1 \leq U, V \leq N$""")

    d = example_problem_descriptions.p03050
    c = descriptions.extract_input_description(d)
    self.assertEqual(
        c,
        r"""Input is given from Standard Input in the following format:

N""")

    d = example_problem_descriptions.p04019
    c = descriptions.extract_input_description(d)
    self.assertEqual(
        c,
        r"""The input is given from Standard Input in the following format:

S""")

  def test_extract_input_constraints(self):
    d = example_problem_descriptions.p00130
    c = descriptions.extract_input_constraints(d)
    self.assertEqual(c, '')

    # There's no 'Constraints' section. This problem is in Japanese.
    d = example_problem_descriptions.p00569
    c = descriptions.extract_input_constraints(d)
    self.assertEqual(c, '')

    d = example_problem_descriptions.p01950
    c = descriptions.extract_input_constraints(d)
    self.assertEqual(c, '')

    d = example_problem_descriptions.p03050
    c = descriptions.extract_input_constraints(d)
    self.assertEqual(c, r"""All values in input are integers.
1 \leq N \leq 10^{12}""")

    d = example_problem_descriptions.p04019
    c = descriptions.extract_input_constraints(d)
    self.assertEqual(c, r"""1 ≦ | S | ≦ 1000
S consists of the letters N, W, S, E.""")

  def test_extract_input_information(self):
    self.maxDiff = None

    d = example_problem_descriptions.p00569
    c = descriptions.extract_input_information(d)
    self.assertEqual(c, r"""制約:
1 \leq N \leq 200000
1 \leq K \leq N
1 \leq a_i \leq N
1 \leq L
JOI 君が書き出す整数は L 個以上である．""")



if __name__ == '__main__':
  unittest.main()
