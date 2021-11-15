#include "utils.h"
// #include<stdio.h>
// #include<mpi.h>

int main() {
    struct Graph G;
    double start, end;
    MPI_Init(NULL, NULL);
    // init_graph_auto(&G, 100, 105);
    int my_rank = -1;
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    if(my_rank == 0) {
        start = MPI_Wtime();
        init_graph(&G);
    }
    check_hamiltonian(G, my_rank);
    Free2DMemory(&G.adj);
    if(my_rank == 0) {
        end = MPI_Wtime();
        printf("Time taken: %f\n", end-start);
    }
    // MPI_Finalize();
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


*/