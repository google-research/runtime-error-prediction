# Copyright (C) 2021 Google Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

p00022 = r"""


<H1>Maximum Sum Sequence</H1>

<p>
Given a sequence of numbers <var>a<sub>1</sub></var>, <var>a<sub>2</sub></var>, <var>a<sub>3</sub></var>, ..., <var>a<sub>n</sub></var>, find the maximum sum of a contiguous subsequence of those numbers. Note that, a subsequence of one element is also a <i>contiquous</i> subsequence.            
</p>

<H2>Input</H2>
<p>
The input consists of multiple datasets. Each data set consists of:

<pre>
<var>n</var>
<var>a<sub>1</sub></var>
<var>a<sub>2</sub></var>
.
.
<var>a<sub>n</sub></var>
</pre>

<p>
You can assume that 1 &le; <var>n</var> &le; 5000 and -100000 &le; <var>a<sub>i</sub></var> &le; 100000.
</p>

<p>
The input end with a line consisting of a single 0.
</p>

<H2>Output</H2>

<p>
For each dataset, print the maximum sum in a line.
</p>

<H2>Sample Input</H2>
<pre>
7
-5
-1
6
4
9
-6
-7
13
1
2
3
2
-2
-1
1
2
3
2
1
-2
1
3
1000
-200
201
0
</pre>

<H2>Output for the Sample Input</H2>

<pre>
19
14
1001
</pre>
"""

p00023 = r"""
<script type="text/x-mathjax-config">
  MathJax.Hub.Config({ tex2jax: { inlineMath: [["$","$"], ["\\(","\\)"]], processEscapes: true }});
</script>
<script type="text/javascript" src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-MML-AM_CHTML">
</script>

<H1>Circles Intersection</H1>

<p>
You are given circle $A$ with radius $r_a$ and with central coordinate $(x_a, y_a)$ and circle $B$ with radius $r_b$ and with central coordinate $(x_b, y_b)$.
</p>
<p>
Write a program which prints:
</p>
<ul>
<li>"2" if $B$ is in $A$,</li>
<li>"-2" if $A$ is in $B$, </li>
<li>"1" if circumference of $A$ and $B$ intersect, and</li>
<li>"0" if $A$ and $B$ do not overlap.</li>
</ul>

<p>
You may assume that $A$ and $B$ are not identical.
</p>

<H2>Input</H2>

<p>
The input consists of multiple datasets. The first line consists of an integer $N$ ($N \leq 50$), the number of datasets. There will be $N$ lines where each line represents each dataset. Each data set consists of real numbers:<br/>
<br/>
$x_a$ $y_a$ $r_a$ $x_b$ $y_b$ $r_b$<br/>
</p>

<H2>Output</H2>

<p>
For each dataset, print 2, -2, 1, or 0 in a line.
</p>

<H2>Sample Input</H2>

<pre>
2
0.0 0.0 5.0 0.0 0.0 4.0
0.0 0.0 2.0 4.1 0.0 2.0
</pre>

<H2>Output for the Sample Input</H2>

<pre>
2
0
</pre>
"""

p00130 = r"""
<h1>??????</h1>

<p>
26 ?????????????????????????????????????????????????????????????????????????????????????????? a ?????? z ???????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????
???????????????????????????????????????????????????????????????????????????????????????????????????????????????
</p>

<p>
??????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????? 1 ?????? 1 ???????????????????????????????????????????????????????????? 1 ???????????? <span><-</span> ????????? <span>-></span> ????????????????????????????????????????????????<span><-</span> ?????????????????????????????????<span>-></span> ?????????????????????????????????????????????
???
</p>

<p>
????????????<span>a->b<-a<-c</span> ????????? a ?????????????????????????????? b ????????????b ??????????????? a ????????????a ??????????????? c ??????????????????????????????????????????????????????????????????????????????????????? <span>cab</span> ??????????????????
</p>


<H2>Input</H2>

<p>
????????????????????????????????? <var>n</var> (<var>n</var> &le; 50)????????? <var>n</var> ?????????????????? <var>i</var> ?????????????????? <var>s<sub>i</sub></var> (1024??????????????????????????????) ????????????????????????
</p>

<H2>Output</H2>

<p>
???????????? <var>i</var> ???????????????????????????????????????????????????????????????????????? <var>i</var> ????????????????????????????????????
</p>


<H2>Sample Input</H2>

<pre>
4
a->e->c->b->d
b<-c<-a<-d<-e
b->a->c<-a->c->d<-c<-a<-b->a->c->d->e<-d
a->e<-a<-d->a->e<-a<-d<-c->d->a<-d<-c<-b->c->d<-c
</pre>

<H2>Output for the Sample Input</H2>

<pre>
aecbd
edacb
bacde
bcdae
</pre>
"""

p00352 = r"""
<H1>Handsel</H1>
<!-- New Year???s gift money -->

<p>
Alice and Brown are brothers in a family and each receives pocket money in celebration of the coming year. They are very close and share the total amount of the money fifty-fifty. The pocket money each receives is a multiple of 1,000 yen. 
</p>

<p>
  Write a program to calculate each one???s share given the amount of money Alice and Brown received.
</p>

<h2>Input</h2>

<p>
  The input is given in the following format.
</p>
<pre>
<var>a</var> <var>b</var>
</pre>

<p>
A line of data is given that contains two values of money: <var>a</var> (1000 &le; <var>a</var> &le; 50000) for Alice and <bar>b</var> (1000 &le; <var>b</var> &le; 50000) for Brown.
</p>


<h2>Output</h2>

<p>
  Output the amount of money each of Alice and Brown receive in a line.
</p>

<h2>Sample Input 1</h2>

<pre>
1000 3000
</pre>

<h2>Sample Output 1</h2>
<pre>
2000
</pre>
"""

p01950 = r"""
<script type="text/x-mathjax-config">
  MathJax.Hub.Config({ tex2jax: { inlineMath: [["$","$"], ["\\(","\\)"]], skipTags: ["script","noscript","style","textarea","code"], processEscapes: true }});
</script>
<script type="text/javascript" async src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.1/MathJax.js?config=TeX-AMS_HTML"></script>

<H1>
Endless BFS
</H1>

<p>
Mr. Endo wanted to write the code that performs breadth-first search (BFS), which is a search algorithm to explore all vertices on an undirected graph. An example of pseudo code of BFS is as follows:
</p>

<pre>
1: $current \leftarrow \{start\_vertex\}$
2: $visited \leftarrow current$
3: while $visited \ne $ the set of all the vertices
4:   $found \leftarrow \{\}$
5:   for $v$ in $current$
6:     for each $u$ adjacent to $v$
7:       $found \leftarrow found \cup\{u\}$
8:   $current \leftarrow found \setminus visited$
9:   $visited \leftarrow visited \cup found$
</pre>

<p>
However, Mr. Endo apparently forgot to manage visited vertices in his code. More precisely, he wrote the following
code:
</p>

<pre>
1: $current \leftarrow \{start\_vertex\}$
2: while $current \ne $ the set of all the vertices
3:   $found \leftarrow \{\}$
4:   for $v$ in $current$
5:     for each $u$ adjacent to $v$
6:       $found \leftarrow found \cup \{u\}$
7:   $current \leftarrow found$
</pre>

<p>
You may notice that for some graphs, Mr. Endo's program will not stop because it keeps running infinitely. Notice that it does not necessarily mean the program cannot explore all the vertices within finite steps. See example 2 below for more details.Your task here is to make a program that determines whether Mr. Endo's program will stop within finite ste
ps for a given graph in order to point out the bug to him. Also, calculate the minimum number of loop iterations required for the program to stop if it is finite.
</p>

<H2>Input</H2>

<p>
The input consists of a single test case formatted as follows.
</p>

<pre>
$N$ $M$
$U_1$ $V_1$
...
$U_M$ $V_M$
</pre>

<p>
  The first line consists of two integers $N$ ($2 \leq N \leq 100,000$) and $M$ ($1 \leq M \leq 100,000$), where $N$ is the number of vertices and $M$ is the number of edges in a given undirected graph, respectively. The $i$-th line of the following $M$ lines consists of two integers $U_i$ and $V_i$ ($1 \leq U_i, V_i \leq N$), which means the vertices $U_i$ and $V_i$ are adjacent in the given graph. The vertex 1 is the start vertex, i.e. $start\_vertex$ in the pseudo codes. You can assume that the given graph also meets the following conditions.
</p>

<ul>
<li>The graph has no self-loop, i.e., $U_i \ne V_i$ for all $1 \leq i \leq M$.</li>
<li>The graph has no multi-edge, i.e., $\{Ui,Vi\} \ne \{U_j,V_j\}$ for all $1 \leq i < j \leq M$.</li>
<li>The graph is connected, i.e., there is at least one path from $U$ to $V$ (and vice versa) for all vertices $1 \leq U, V \leq N$</li>
</ul>


<H2>Output</H2>

<p>
If Mr. Endo's wrong BFS code cannot stop within finite steps for the given input graph, print -1 in a line. Otherwise, print the minimum number of loop iterations required to stop.
</p>

<H2>Sample Input 1</H2>
<pre>
3 3
1 2
1 3
2 3
</pre>

<H2>Output for Sample Input 1</H2>
<pre>
2
</pre>



<H2>Sample Input 2</H2>
<pre>
4 3
1 2
2 3
3 4
</pre>

<H2>Output for Sample Input 2</H2>
<pre>
-1
</pre>

<p>
Transition of $current$ is $\{1\} \rightarrow \{2\} \rightarrow \{1,3\} \rightarrow \{2,4\} \rightarrow \{1,3\} \rightarrow \{2,4\} \rightarrow ... $. Although Mr. Endo's program will achieve to visit all the vertices (in 3 steps), will never become the same set as all the vertices.
</p>


<H2>Sample Input 3</H2>
<pre>
4 4
1 2
2 3
3 4
4 1
</pre>

<H2>Output for Sample Input 3</H2>
<pre>
-1
</pre>


<H2>Sample Input 4</H2>
<pre>
8 9
2 1
3 5
1 6
2 5
3 1
8 4
2 7
7 1
7 4
</pre>

<H2>Output for Sample Input 4</H2>
<pre>
3
</pre>
"""

p00729_abbr = r"""<h1><font color="#000">Problem B:</font> Analyzing Login/Logout Records</h1>

This shouldn't be included.

<h2>Input</h2>

Example

<p>
<nl>
  <li>Unclosed item
</nl>
</p>


<h2>Output</h2>

This shouldn't be included.
"""

p03050 = r"""
<span class="lang-en">
<p>Score : <var>500</var> points</p>
<div class="part">
<section>
<h3>Problem Statement</h3><p>Snuke received a positive integer <var>N</var> from Takahashi.
A positive integer <var>m</var> is called a <em>favorite number</em> when the following condition is satisfied:</p>
<ul>
<li>The quotient and remainder of <var>N</var> divided by <var>m</var> are equal, that is, <var>\lfloor \frac{N}{m} \rfloor = N \bmod m</var> holds.</li>
</ul>
<p>Find all favorite numbers and print the sum of those.</p>
</section>
</div>
<div class="part">
<section>
<h3>Constraints</h3><ul>
<li>All values in input are integers.</li>
<li><var>1 \leq N \leq 10^{12}</var></li>
</ul>
</section>
</div>
<hr/>
<div class="io-style">
<div class="part">
<section>
<h3>Input</h3><p>Input is given from Standard Input in the following format:</p>
<pre><var>N</var>
</pre>
</section>
</div>
<div class="part">
<section>
<h3>Output</h3><p>Print the answer.</p>
</section>
</div>
</div>
<hr/>
<div class="part">
<section>
<h3>Sample Input 1</h3><pre>8
</pre>
</section>
</div>
<div class="part">
<section>
<h3>Sample Output 1</h3><pre>10
</pre>
<p>There are two favorite numbers: <var>3</var> and <var>7</var>. Print the sum of these, <var>10</var>.</p>
</section>
</div>
<hr/>
<div class="part">
<section>
<h3>Sample Input 2</h3><pre>1000000000000
</pre>
</section>
</div>
<div class="part">
<section>
<h3>Sample Output 2</h3><pre>2499686339916
</pre>
<p>Watch out for overflow.</p></section>
</div>
</span>
"""


p04019 = """
<span class="lang-en">
<p>Score : <var>200</var> points</p>
<div class="part">
<section>
<h3>Problem Statement</h3><p>Snuke lives on an infinite two-dimensional plane. He is going on an <var>N</var>-day trip.
At the beginning of Day <var>1</var>, he is at home. His plan is described in a string <var>S</var> of length <var>N</var>.
On Day <var>i(1 ??? i ??? N)</var>, he will travel a positive distance in the following direction:</p>
<ul>
<li>North if the <var>i</var>-th letter of <var>S</var> is <code>N</code></li>
<li>West if the <var>i</var>-th letter of <var>S</var> is <code>W</code></li>
<li>South if the <var>i</var>-th letter of <var>S</var> is <code>S</code></li>
<li>East if the <var>i</var>-th letter of <var>S</var> is <code>E</code></li>
</ul>
<p>He has not decided each day's travel distance. Determine whether it is possible to set each day's travel distance so that he will be back at home at the end of Day <var>N</var>.</p>
</section>
</div>
<div class="part">
<section>
<h3>Constraints</h3><ul>
<li><var>1 ??? | S | ??? 1000</var></li>
<li><var>S</var> consists of the letters <code>N</code>, <code>W</code>, <code>S</code>, <code>E</code>.</li>
</ul>
</section>
</div>
<hr/>
<div class="io-style">
<div class="part">
<section>
<h3>Input</h3><p>The input is given from Standard Input in the following format:</p>
<pre><var>S</var>
</pre>
</section>
</div>
<div class="part">
<section>
<h3>Output</h3><p>Print <code>Yes</code> if it is possible to set each day's travel distance so that he will be back at home at the end of Day <var>N</var>. Otherwise, print <code>No</code>.</p>
</section>
</div>
</div>
<hr/>
<div class="part">
<section>
<h3>Sample Input 1</h3><pre>SENW
</pre>
</section>
</div>
<div class="part">
<section>
<h3>Sample Output 1</h3><pre>Yes
</pre>
<p>If Snuke travels a distance of <var>1</var> on each day, he will be back at home at the end of day <var>4</var>.</p>
</section>
</div>
<hr/>
<div class="part">
<section>
<h3>Sample Input 2</h3><pre>NSNNSNSN
</pre>
</section>
</div>
<div class="part">
<section>
<h3>Sample Output 2</h3><pre>Yes
</pre>
</section>
</div>
<hr/>
<div class="part">
<section>
<h3>Sample Input 3</h3><pre>NNEW
</pre>
</section>
</div>
<div class="part">
<section>
<h3>Sample Output 3</h3><pre>No
</pre>
</section>
</div>
<hr/>
<div class="part">
<section>
<h3>Sample Input 4</h3><pre>W
</pre>
</section>
</div>
<div class="part">
<section>
<h3>Sample Output 4</h3><pre>No
</pre></section>
</div>
</span>
"""

p00569 = r"""
<h1>L?????????K???????????? (LthKthNumber)</h1>


<h2>?????????</h2>
<p>
??????????????????????????? <var>N</var> ???????????????????????????????????? <var>i</var> ??????(<var>1 ??? i ??? N</var>)??????????????????????????? <var>a_i</var> ????????????????????????</p>

<p>
JOI ?????????????????????????????????????????????????????????????????????????????????????????? <var>K</var> ???????????????????????????????????????????????????????????????</p>


<ul>
<li>???????????????????????????????????????????????????????????????????????????????????????</li>
<li>??????????????????????????????????????? <var>K</var> ???????????????????????????????????????????????????????????????</li>
<li>?????????????????????????????????????????????????????????</li>
</ul>

<p>
?????????????????????????????? <var>K</var> ????????????????????????????????????????????????????????????????????????<var>1 ??? l ??? r ??? N</var> ?????? <var>K ??? r - l + 1</var> ???????????????????????? <var>(l,r)</var> ???????????????<var>a_l, a_{l+1}, ..., a_r</var> ????????? <var>K</var> ??????????????????????????????????????????</p>

<p>
?????????????????????????????????????????????????????????????????????????????????????????????????????????????????? <var>L</var> ????????????????????????????????????????????? JOI ????????????????????????JOI ???????????????????????????</p>

<h2>??????</h2>

<ul>
<li><var>1 \leq N \leq 200000</var></li>
<li><var>1 \leq K \leq N</var></li>
<li><var>1 \leq a_i \leq N</var></li>
<li><var>1 \leq L</var></li>
<li>JOI ??????????????????????????? <var>L</var> ?????????????????????</li>
</ul>

<h2>???????????????</h2>

<p>
<b>??????</b><br>
???????????????????????????????????????????????????????????????<br>
<var>N</var> <var>K</var> <var>L</var><br>
<var>a_1</var> <var>a_2</var> <var>...</var> <var>a_N</var>
</p>

<p>
<b>??????</b><br>
JOI ??????????????? <var>1</var> ?????????????????????<br>

<!--
<h2>?????????</h2>

<p>
<b>????????? 1 [6???]</b>
</p>

<ul>
<li><var>N ??? 100</var></li>
</ul>

<p>
<b>????????? 2 [33???]</b>
</p>

<ul>
<li><var>N ??? 4000</var></li>
</ul>

<p>
<b>????????? 3 [61???]</b>
</p>

<ul>
  <li>???????????????????????????</li>
</ul>
-->

<h2>????????????</h2>


<b>????????? 1</b><br>
<pre>
4 3 2
4 3 1 2
</pre>


<b>????????? 1</b><br>
<pre>
3
</pre>

<p>
<var>1 \leq l \leq r \leq N (= 4)</var> ?????? <var>K (= 3) \leq r - l + 1</var> ???????????? <var>(l,r)</var> ??????<var>(1,3), (1,4), (2,4)</var> ??? <var>3</var> ???????????????</p>

<p>
???????????? <var>(l,r)</var> ????????????<var>a_l, a_{l+1}, ..., a_r</var> ??? <var>3</var> ?????????????????????????????????????????? <var>4, 3, 3</var> ????????????</p>

<p>
???????????? <var>L (= 2)</var> ??????????????????????????? <var>3</var> ????????????JOI ??????????????? <var>3</var> ????????????????????????????????????????????????????????????????????????????????????????????????</p>

<hr>

<b>????????? 2</b><br>
<pre>
5 3 3
1 5 2 2 4
</pre>


<b>????????? 2</b><br>
<pre>
4
</pre>

<p>
JOI ??????????????????????????????</p>

<ul>
<li><var>(l,r) = (1,3)</var> ????????? <var>5</var></li>
<li><var>(l,r) = (1,4)</var> ????????? <var>2</var></li>
<li><var>(l,r) = (1,5)</var> ????????? <var>2</var></li>
<li><var>(l,r) = (2,4)</var> ????????? <var>5</var></li>
<li><var>(l,r) = (2,5)</var> ????????? <var>4</var></li>
<li><var>(l,r) = (3,5)</var> ????????? <var>4</var></li>
</ul>

<p>
???????????????????????? <var>L (= 3)</var> ??????????????????????????? <var>4</var> ????????????
</p>

<hr>


<b>????????? 3</b><br>
<pre>
6 2 9
1 5 3 4 2 4
</pre>


<b>????????? 3</b><br>
<pre>
4
</pre>

<hr>


<b>????????? 4</b><br>
<pre>
6 2 8
1 5 3 4 2 4
</pre>

<b>????????? 4</b><br>
<pre>
3
</pre>
"""
