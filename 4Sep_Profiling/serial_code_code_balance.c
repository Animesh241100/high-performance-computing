#include<stdio.h>
#include<string.h>
#include<stdlib.h>
#include<time.h>

#define ERROR -20

// Represents a Stack
struct Stack {
    int * arr;
    int max_size;
    int top;
};

// Represents an Undirected Graph
struct Graph {
    int E;
    int V;
    int ** adj; // adjacency matrix
};

/***************************************************************************/

void init_graph_auto(struct Graph *G, int V, int E);
void init_graph(struct Graph *G);
void check_hamiltonian(struct Graph G);
int num_hamiltonian_cycles(int position, int * visit, struct Graph G, struct Stack *Path);
void push(struct Stack *S, int data);
int pop(struct Stack *S);
void show_stack(struct Stack *S);
void print_graph(struct Graph *G);

/***************************************************************************/


int main() {
    struct Graph G;     // #flops = 0 ==> BC = NA
    init_graph_auto(&G, 100, 105);      // uncomment to initialize the graph randomly
    // init_graph(&G);                  // uncomment to initialize the graph with user input
    check_hamiltonian(G);

    for (int i = 0; i < G.V; i++) 
        free(G.adj[i]);
    free(G.adj);
    return 0;   // #flops = 0 ==> BC = NA
}



/********************** stack operations implementation *******************/
void push(struct Stack *S, int data) {
    if(S->top > S->max_size - 1) // #words = 2; #flops = 2 ==> BC = 2/2 = 1
        printf("Stack overflow\n"); 
    else {
        S->top++; // #words = 1; #flops = 1 ==> BC = 1/1 = 1
        S->arr[S->top] = data; // #words = 3; #flops = 0 ==> BC = 3/0 = NA
    }
}

int top(struct Stack *S) {
    if(S->top > -1)  // #words = 1; #flops = 1 ==> BC = 1/1 = 1
        return S->arr[S->top]; // #words = 2; #flops = 0 ==> BC = 2/0 = NA
    printf("Stack is empty!\n");
    return ERROR;  // #flops = 0 ==> BC = NA

}

int pop(struct Stack *S) {
    if(S->top > -1) {  // #words = 1; #flops = 1 ==> BC = 1/1 = 1
        int val = S->arr[S->top];  // #words = 3; #flops = 0 ==> BC = 3/0 = NA
        S->top--; // #words = 1; #flops = 1 ==> BC = 1/1 = 1
        return val;  // #flops = 0 ==> BC = NA
    }
    printf("Popping out of empty stack\n");
    return ERROR;  // #flops = 0 ==> BC = NA
}

void show_stack(struct Stack *S) {
    printf("[");
    for(int i = 0; i <= S->top; i++) {
        printf("%d ", S->arr[i]);
    }
    printf("\b<---top\n");
}

/******************** Graph Initialization related ************************/

// initializes an undirected graph with V nodes and E edges
void init_graph_auto(struct Graph *G, int V, int E) {
    G->V = V;   // #words = 2; #flops = 0 ==> BC = 2/0 = NA
    G->E = E;    // #words = 2; #flops = 0 ==> BC = 2/0 = NA
    srand(time(0));
    G->adj = (int**)malloc(sizeof(int*)*G->V);  // #words = 2; #flops = 1 ==> BC = 2/1 = 2
    for(int i = 0; i < G->V; i++) {
        G->adj[i] = (int *)malloc(sizeof(int)*G->V);  // #words = 2; #flops = 1 ==> BC = 2/1 = 2
        memset(G->adj[i], 0, sizeof(int)*G->V);
    }
    int u, v;
    for(int i = 0; i < G->E; i++) {
        u = rand() % G->V;  // #words = 2; #flops = 1 ==> BC = 2/1 = 2
        v = rand() % G->V;  // #words = 2; #flops = 1 ==> BC = 2/1 = 2
        G->adj[u][v] = 1;   // #words = 3; #flops = 0 ==> BC = 3/0 = NA
        G->adj[v][u] = 1;   // #words = 3; #flops = 0 ==> BC = 3/0 = NA
    }
}

// initializes an undirected graph based on the user input
void init_graph(struct Graph *G) {
    scanf("%d %d", &G->V, &G->E);
    G->adj = (int**)malloc(sizeof(int*)*G->V); // #words = 2; #flops = 1 ==> BC = 2/1 = 2
    for(int i = 0; i < G->V; i++) {
        G->adj[i] = (int *)malloc(sizeof(int)*G->V); // #words = 2; #flops = 1 ==> BC = 2/1 = 2
        memset(G->adj[i], 0, sizeof(int)*G->V);
    }
    int u, v;
    for(int i = 0; i < G->E; i++) {
        scanf("%d %d", &u, &v);
        G->adj[v][u] = 1;  // #words = 3; #flops = 0 ==> BC = 3/0 = NA
        G->adj[u][v] = 1;  // #words = 3; #flops = 0 ==> BC = 3/0 = NA
    }
}

// prints the graph
void print_graph(struct Graph *G) {
    for(int i = 0; i < G->V; i++) {
        printf("%d: ", i);
        for(int j = 0; j < G->V; j++) {
            printf("%d ", G->adj[i][j]);
        }
        printf("\n");
    }   
}

/************************ Counting Hamiltonian cycles *********************/

void check_hamiltonian(struct Graph G) {
    int visit[G.V];  // #flops = 0 ==> BC = NA
    for(int i = 0; i < G.V; i++)
        visit[i] = 0;  // #words = 1; #flops = 0 ==> BC = 1/0 = NA
    struct Stack Path;  // #flops = 0 ==> BC = NA
    Path.max_size = G.V + 1;  // #words = 2; #flops = 1 ==> BC = 2/1 = 2
    Path.top = -1;   // #words = 1; #flops = 0 ==> BC = 1/0 = NA
    Path.arr = (int*)malloc(sizeof(int)*(G.V + 1));  // #words = 2; #flops = 1 ==> BC = 2/1 = 2
    push(&Path, 0);
    visit[0] = 1; // #words = 1; #flops = 0 ==> BC = 1/0 = NA
    int num_cycles = num_hamiltonian_cycles(1, visit, G, &Path);
    printf("#Hamiltonian Cycles = %d\n", num_cycles); 
}

// returns the number of hamiltonian cycles present in the undirected graph `G`
int num_hamiltonian_cycles(int position, int * visit, struct Graph G, struct Stack *Path) {
    if(position == G.V) {    // #words = 2; #flops = 1 ==> BC = 2/1 = 2
        if(G.adj[top(Path)][0]) { // #words = 2; #flops = 1 ==> BC = 2/1 = 2
            push(Path, 0);
            show_stack(Path);
            pop(Path);
            return 1;  // #flops = 0 ==> BC = NA
        }
        return 0;  // #flops = 0 ==> BC = NA
    }
    int num_cycles = 0;  // #flops = 0 ==> BC = NA
    for(int i = 0; i < G.V; i++) {
        if((G.adj[top(Path)][i] && !visit[i])) {  // #words = 4; #flops = 2 ==> BC = 4/2 = 2
            push(Path, i);
            visit[i] = 1;   // #flops = 0 ==> BC = NA
            num_cycles += num_hamiltonian_cycles(position + 1, visit, G, Path);
            visit[i] = 0;   // #flops = 0 ==> BC = NA
            pop(Path);
        }
    }
    return num_cycles;  // #flops = 0 ==> BC = NA
}

/***************************************************************************/

/* Input types

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


*/