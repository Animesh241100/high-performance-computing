#include<stdio.h>
#include<string.h>
#include<stdlib.h>
#include<time.h>

#define EMPTY -20


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

struct Args {
    int position;
    int * visit;
    struct Stack *path;
};

struct Stack_Args {
    struct Args * arr;
    int max_size;
    int top;
};


void init_graph_auto(struct Graph *G, int V, int E);
void init_graph(struct Graph *G);
void check_hamiltonian(struct Graph G);
int num_hamiltonian_cycles(int position, int * visit, struct Graph G, struct Stack *Path);
int num_hamiltonian_cycles2(int position, int * visit, struct Graph G, struct Stack *Path);
void push(struct Stack *S, int data);
int pop(struct Stack *S);
void show_stack(struct Stack *S);
void print_graph(struct Graph *G);

void show_stack_args(struct Stack_Args *S, int V);
void push_args(struct Stack_Args *S, struct Args data, int V);
struct Args top_args(struct Stack_Args *S);
int pop_args(struct Stack_Args *S);
void copy_path(struct Stack * st, struct Stack * copy_st);
void copy_visit(int * visit, int * copy_visit, int len);
void print_args(struct Args args, int V);

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



/**************************** stack utils ********************************/
void push(struct Stack *S, int data) {
    if(S->top > S->max_size - 1)
        printf("Stack overflow\n");
    else {
        S->top++;
        S->arr[S->top] = data;
    }
}

int top(struct Stack *S) {
    if(S->top > -1) 
        return S->arr[S->top];
    printf("Stack is empty!\n");
    return EMPTY;

}

int pop(struct Stack *S) {
    if(S->top > -1) {
        int val = S->arr[S->top];
        S->top--;
        return val;
    }
    printf("Popping out of empty stack\n");
    return EMPTY;
}

void show_stack(struct Stack *S) {
    printf("[");
    for(int i = 0; i <= S->top; i++) {
        printf("%d ", S->arr[i]);
    }
    printf("\b<---top\n");
}

/****************** another ************************/
void push_args(struct Stack_Args *S, struct Args data, int V) {
    if(S->top > S->max_size - 1)
        printf("Stack overflow\n");
    else {
        S->top++;
        S->arr[S->top] = *(struct Args*)malloc(sizeof(struct Args)+sizeof(int)*V); 
        S->arr[S->top].path = data.path;
        S->arr[S->top].position = data.position;
        S->arr[S->top].visit = data.visit;
        // data;
    }
}

struct Args top_args(struct Stack_Args *S) {
    if(S->top > -1) 
        return S->arr[S->top];
    printf("lol Stack is empty!\n");
    struct Args * args = (struct Args*)malloc(sizeof(struct Args));
    args->position = EMPTY;
    return *args;
}

int pop_args(struct Stack_Args *S) {
    if(S->top > -1) {
        S->top--;
        return 1;
    }
    printf("Popping out of empty stack\n");
    return EMPTY;
}

void show_stack_args(struct Stack_Args *S, int V) {
    printf("[");
    for(int i = 0; i <= S->top; i++) {
        print_args(S->arr[i], V);
        printf("___]\n");
    }
    printf("\b<-----------------top args\n");
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
    int num_cycles = num_hamiltonian_cycles2(1, visit, G, &Path);
    printf("#Hamiltonian Cycles = %d\n", num_cycles); 
}

int num_hamiltonian_cycles(int position, int * visit, struct Graph G, struct Stack *Path) {
    if(position == G.V) {
        if(G.adj[top(Path)][0]) {
            push(Path, 0);
            show_stack(Path);
            pop(Path);
            return 1;
        }
        return 0;
    }
    int num_cycles = 0;
    for(int i = 0; i < G.V; i++) {
        if((G.adj[top(Path)][i] && !visit[i])) {
            push(Path, i);
            visit[i] = 1;
            num_cycles += num_hamiltonian_cycles(position + 1, visit, G, Path);
            visit[i] = 0;
            pop(Path);
        }
    }
    return num_cycles;
}

int num_hamiltonian_cycles2(int pos, int * vis, struct Graph G, struct Stack *P) {
    int num = 0;
    struct Stack_Args args_stack;
    args_stack.max_size = 100;
    args_stack.top = -1;
    args_stack.arr = (struct Args*)malloc(sizeof(struct Args)*(100));
    struct Args args;
    args.path = P;
    args.visit = vis;
    args.position = pos;
    push_args(&args_stack, args, G.V);
    while(top_args(&args_stack).position != EMPTY) {
        show_stack_args(&args_stack, G.V);
        args = top_args(&args_stack);
        pop_args(&args_stack);
        int position = args.position;
        int * visit = args.visit;
        struct Stack * Path = args.path;
        // for(int i = 0; i < G.V; i++)
            // printf("x%d ", visit[i]);
        // print_args(args, G.V);
        if(position == G.V) {
            if(G.adj[top(Path)][0]) {
                push(Path, 0);
                show_stack(Path);
                pop(Path);
                num++;
            }
        } else {
            for(int i = 0; i < G.V; i++) {
                if((G.adj[top(Path)][i] && !visit[i])) {
                    push(Path, i);
                    visit[i] = 1;
                    int* visit_copy = (int*)malloc(sizeof(int)*G.V);
                    printf("\n");
                    struct Stack* Path_copy = (struct Stack*)malloc(sizeof(struct Stack));
                    Path_copy->top = -1;
                    Path_copy->arr = (int*)malloc(sizeof(int)*(G.V + 1));
                    copy_visit(visit, visit_copy, G.V);
                    copy_path(Path, Path_copy);
                    struct Args args_temp;
                    args_temp.position = position + 1 ;
                    args_temp.visit = visit_copy;
                    args_temp.path = Path_copy;
                    push_args(&args_stack, args_temp, G.V);
                    // num_cycles += num_hamiltonian_cycles(position + 1, visit, G, Path);
                    // show_stack_args(&args_stack, G.V);
                    visit[i] = 0;
                    pop(Path);
                }
            }
            // return num_cycles;
            // }
        }
    }
    return num;
}

void print_args(struct Args args, int V) {
    printf("pos: %d\n", args.position);
    show_stack(args.path);
    for(int i = 0; i < V; i++)
        printf("%d ", args.visit[i]);
    printf("\n");
}

void copy_path(struct Stack * st, struct Stack * copy_st) {
    copy_st->max_size = st->max_size;
    copy_st->top = st->top;
    copy_st->arr = (int*)malloc(sizeof(int)*(st->max_size));
    for(int i = 0; i < st->max_size; i++)
        copy_st->arr[i] = st->arr[i];
}

// copy array `visit` to `copy_visit`
void copy_visit(int * visit, int * copy_visit, int len) {
    for(int i = 0; i < len; i++)
        copy_visit[i] = visit[i];
}


// void print_args(struct Args args) {
//     printf("%d: ", args.position);

// }
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