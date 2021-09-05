#include<stdio.h>
#include<string.h>
#include<stdlib.h>
#include<time.h>

#define ERROR -20

struct Stack {
    int * arr;
    int max_size;
    int top;
};

struct Graph {
    int E;
    int V;
    int ** adj; // adjacency matrix
};

void init_graph_auto(struct Graph *G, int V, int E);
void init_graph(struct Graph *G);
void check_hamiltonian(struct Graph G);
int is_hamiltonian(int position, int * visit, struct Graph G, struct Stack *Path);
void push(struct Stack *S, int data);
int pop(struct Stack *S);
int show_stack(struct Stack *S);

int main() {
    struct Graph G;
    init_graph_auto(&G, 100, 105);
    check_hamiltonian(G);
    for (int i = 0; i < G.V; i++) 
        free(G.adj[i]);
    free(G.adj);
    return 0;
}



/**************************** stack utils ********************************/
void push(struct Stack *S, int data) {
    if(S->top >= S->max_size - 1)
        printf("Stack overflow\n");
    else {
        S->top++;
        S->arr[S->top] = data;
    }
}

int pop(struct Stack *S) {
    if(S->top > -1) {
        int val = S->arr[S->top];
        S->top--;
        return val;
    }
    printf("Popping out of empty stack\n");
    return ERROR;
}

int show_stack(struct Stack *S) {
    printf("[");
    for(int i = 0; i <= S->top; i++) {
        printf("%d ", S->arr[i]);
    }
    printf("\b<---top\n");
}

/**************************** Graph related ********************************/


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

// void init_graph(struct Graph *G) {
//     scanf("%d %d", &G->V, &G->E);
//     G->adj = (int**)malloc(sizeof(int*)*G->V);
//     for(int i = 0; i < G->V; i++) {
//         G->adj[i] = (int *)malloc(sizeof(int)*G->V);
//         memset(G->adj[i], 0, sizeof(int)*G->V);
//     }
//     int u, v;
//     for(int i = 0; i < G->E; i++) {
//         scanf("%d %d", &u, &v);
//         G->adj[v][u] = 1;
//         G->adj[u][v] = 1;
//     }
// }

/***************************** Hamiltonian **********************************/

void check_hamiltonian(struct Graph G) {
    int visit[G.V];
    for(int i = 0; i < G.V; i++)
        visit[i] = 0;
    struct Stack Path;
    Path.max_size = G.V + 1;
    Path.top = -1;
    Path.arr = (int*)malloc(sizeof(int)*(G.V + 1));
    push(&Path, 0);
    visit[0] = 1;
    if(is_hamiltonian(1, visit, G, &Path)) {
        printf("Hamiltonian Cycle is present\n"); 
        show_stack(&Path);
    }
    else
        printf("No hamiltonian cycle is present.\n");
}

int is_hamiltonian(int position, int * visit, struct Graph G, struct Stack *Path) {
    if(position == G.V) {
        if(G.adj[Path->arr[position-1]][0]) {
            push(Path, 0);
            return 1;
        }
        return 0;
    }
    for(int i = 0; i < G.V; i++) {
        if(G.adj[Path->arr[position - 1]][i] && !visit[i]) {
            push(Path, i);
            visit[i] = 1;
            if(is_hamiltonian(position + 1, visit, G, Path))
                return 1;
            visit[i] = 0;
            pop(Path);
        }
    }
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
for(int i = 0; i < G->V; i++) {
    printf("%d: ", i);
    for(int j = 0; j < G->V; j++) {
        printf("%d ", G->adj[i][j]);
    }
    printf("\n");
}   


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






*/