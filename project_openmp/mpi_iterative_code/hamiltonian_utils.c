#include "utils.h"

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
    int num_cycles = num_hamiltonian_cycles(1, visit, G, &Path);
    printf("#Hamiltonian Cycles = %d\n", num_cycles); 
}

int num_hamiltonian_cycles(int pos, int * vis, struct Graph G, struct Stack *P) {
    int num_cycles = 0;
    struct Stack_Args args_stack = init_args_stack();
    struct Args args = init_args(pos, vis, P);
    push_args(&args_stack, args);
    while(top_args(&args_stack).position != EMPTY) {  // run the loop code stack.size times, these iterations don't have dependency
        args = top_args(&args_stack);
        pop_args(&args_stack);
        int position = args.position;
        int * visit = args.visit;
        struct Stack * Path = args.path;
        if(position == G.V) { // the base case when we finally get to know whether we came to the end of a path
            if(G.adj[top(Path)][0]) {
                push(Path, 0);
                show_stack(Path);
                pop(Path);
                num_cycles++;
            }
        } else {
            for(int i = 0; i < G.V; i++) {
                if((G.adj[top(Path)][i] && !visit[i])) { // for each of the unvisited adjacent vertex, of the vertex at the top of the stack 'Path'
                    push(Path, i);
                    visit[i] = 1;
                    int* visit_copy = (int*)malloc(sizeof(int)*G.V);
                    struct Stack* Path_copy = (struct Stack*)malloc(sizeof(struct Stack));
                    copy_visit(visit, visit_copy, G.V);
                    copy_path(Path, Path_copy);
                    args = init_args(position+1, visit_copy, Path_copy);
                    push_args(&args_stack, args);
                    visit[i] = 0; // backtracking step
                    pop(Path);
                }
            }
        }
    }
    return num_cycles;
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
