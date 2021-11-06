#include "utils.h"

int main() {
    struct Graph G;
    // init_graph_auto(&G, 100, 105);
    init_graph(&G);
    check_hamiltonian(G);
    for (int i = 0; i < G.V; i++) 
        free(G.adj[i]);
    free(G.adj);
    return 0;
}
