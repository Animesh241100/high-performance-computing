#include "utils.cuh"

int main() {
    struct Graph G;
    // init_graph_auto(&G, 100, 105);
    init_graph(&G);
    int GRID_SIZE, BLOCK_SIZE;
    printf("Enter the grid size and the block size respectively:\n");
    scanf("%d %d", &GRID_SIZE, &BLOCK_SIZE);
    auto start = std::chrono::high_resolution_clock::now();
    check_hamiltonian(G, GRID_SIZE, BLOCK_SIZE);
    auto stop = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
    printf("Time Taken: %ld\n", duration.count());
    Free2DMemory(&G.adj);
    return 0;
}

/* Input types
8 9
0 1
1 2
2 3
3 4
4 1
4 6
3 5
5 6
6 7

ham

8 9
0 1
1 2
2 3
3 4
0 4
4 7
3 5
5 6
6 7



print the graph



17 30
0 1
1 2
2 8
0 8
2 4
2 3
3 4
8 3
3 12
4 12
4 15
14 15
1 9
0 10
9 10
9 14
13 14
13 10
13 16
11 16
10 11
6 11
0 6
5 12
5 6
6 11
7 6
7 5
5 8
3 5
==> 28 cycles

6 8
0 1
1 2
2 5
4 5
3 4
0 3
1 4
2 3
==> 4 cycles

6 8
0 1
1 2
2 5
5 4
4 3
0 3
2 3
1 4

==> 4 cycles

7 12
0 1
1 2
2 3
3 4
4 0
1 5
0 6
5 3
2 5
5 6
6 4
3 6
==> 14 cycles

*/