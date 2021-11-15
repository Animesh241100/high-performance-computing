#include "utils.h"

// auto intialises a graph 'G' with 'V' vertices and 'E' edges
void init_graph_auto(struct Graph *G, int V, int E) {
    G->V = V;
    G->E = E;
    srand(time(0));
    Allocate2DMemory(&G->adj, G->V, G->V);
    int u, v;
    for(int i = 0; i < G->E; i++) {
        u = rand() % G->V;
        v = rand() % G->V;
        G->adj[u][v] = 1;
        G->adj[v][u] = 1;
    }
}

// intialises a graph 'G' w.r.t. the input from STDIN
void init_graph(struct Graph *G) {
    scanf("%d %d", &G->V, &G->E);
    Allocate2DMemory(&G->adj, G->V, G->V);
    int u, v;
    for(int i = 0; i < G->E; i++) {
        scanf("%d %d", &u, &v);
        G->adj[v][u] = 1;
        G->adj[u][v] = 1;
    }
}

// prints the graph 'G' on the STDOUT
void print_graph(struct Graph *G) {
    for(int i = 0; i < G->V; i++) {
        printf("%d: ", i);
        for(int j = 0; j < G->V; j++) {
            printf("%d ", G->adj[i][j]);
        }
        printf("\n");
    }   
}


// allocates the memory for the pseudo 2D array
int Allocate2DMemory(int ***array, int n, int m) {
    int *p = (int *)malloc(n*m*sizeof(int));
    if (!p) return -1;

    (*array) = (int **)malloc(n*sizeof(int*));
    if (!(*array)) {
       free(p);
       return -1;
    }

    for (int i=0; i<n; i++) 
       (*array)[i] = &(p[i*m]);

    return 0;
}

// frees the memory of the pseudo 2D array
int Free2DMemory(int ***array) {
    free(&((*array)[0][0]));
    free(*array);
    return 0;
}

