### Cuda Solution of a computationally intensive problem:

Given an undirected graph on a set of nodes `V` and edges `E`. Propose an algorithm to calculate the number of hamiltonian cycles in this graph.

* We already have the serial algorithm which makes use of the backtracking paradigm.
* I found out the parallelizable part of code and serialized it using CUDA Programming.
  * Details on the serial code analysis can be found here: https://docs.google.com/document/d/1ed1IqOakGUQwgQ3cTgRAVgryF-csT1hngPdxkphoKyo/edit?usp=sharing


Details on the algorithm and synchronization strategies used in the CUDA implementation: https://docs.google.com/document/d/1u1an9uIP3MsZY791sEN_7KzPqY0LtUcPftclO7s3emA/edit?usp=sharing 
