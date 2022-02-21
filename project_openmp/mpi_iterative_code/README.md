## Solution of a computationally intensive problem using MPI:

Given an undirected graph on a set of nodes `V` and edges `E`. Propose an algorithm to calculate the number of hamiltonian cycles in this graph.

* We already have the serial algorithm which makes use of the backtracking paradigm.
* I found out the parallelizable part of code and parallelized it using the MPI Programming.
  * Details on the serial code analysis can be found here: https://docs.google.com/document/d/1ed1IqOakGUQwgQ3cTgRAVgryF-csT1hngPdxkphoKyo/edit?usp=sharing
* This MPI program was tested on a cluster of 3 virtual machines.
