#include "utils.h"


void init_graph_auto(struct Graph *G, int V, int E) {
    G->V = V;
    G->E = E;
    srand(time(0));
    G->adj = (int**)malloc(sizeof(int*)*G->V);
    for(int i = 0; i < G->V; i++) {
        G->adj[i] = (int *)malloc(sizeof(int)*G->V);
        memset(G->adj[i], 0, sizeof(int)*G->V);
    }
    int u, v;
    for(int i = 0; i < G->E; i++) {
        u = rand() % G->V;
        v = rand() % G->V;
        G->adj[u][v] = 1;
        G->adj[v][u] = 1;
    }
}

void init_graph(struct Graph *G) {
    scanf("%d %d", &G->V, &G->E);
    G->adj = (int**)malloc(sizeof(int*)*G->V);
    for(int i = 0; i < G->V; i++) {
        G->adj[i] = (int *)malloc(sizeof(int)*G->V);
        memset(G->adj[i], 0, sizeof(int)*G->V);
    }
    int u, v;
    for(int i = 0; i < G->E; i++) {
        scanf("%d %d", &u, &v);
        G->adj[v][u] = 1;
        G->adj[u][v] = 1;
    }
}

void print_graph(struct Graph *G) {
    for(int i = 0; i < G->V; i++) {
        printf("%d: ", i);
        for(int j = 0; j < G->V; j++) {
            printf("%d ", G->adj[i][j]);
        }
        printf("\n");
    }   
}
