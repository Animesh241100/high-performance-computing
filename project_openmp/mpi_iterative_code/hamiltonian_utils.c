#include "utils.h"

// subroutine to check if there are hamiltoinian cycles present in the graph 'G'
void check_hamiltonian(struct Graph G, int my_rank) {
    int visit[G.V];
    for(int i = 0; i < G.V; i++)
        visit[i] = 0;
    struct Stack Path;
    Path.max_size = G.V + 1;
    Path.top = -1;
    Path.arr = (int*)malloc(sizeof(int)*(G.V + 1));
    push(&Path, 0);
    visit[0] = 1;
    int num_cycles = num_hamiltonian_cycles(1, visit, G, &Path, my_rank);
    
    if(my_rank == 0)
        printf("#Hamiltonian Cycles = %d\n", num_cycles); 
}

// returns the number of hamiltonian cycles present in the graph 'G'
int num_hamiltonian_cycles(int pos, int * vis, struct Graph G, struct Stack *P, int my_rank) {
    int num_cycles = 0;
    int temp_num_cycles = 0;
    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    int num_proc = 1;
    struct Stack_Args args_stack = init_args_stack();
    struct Stack_Args temp_args_stack = init_args_stack();
    struct Args args = init_args(pos, vis, P);
    push_args(&args_stack, args);
    BroadCastGraph(&G, my_rank);
    while((my_rank == 0 && size_stack_args(&args_stack) > 0) || (my_rank > 0)) {  // run the loop code stack.size times, these iterations don't have dependency
        MPI_Bcast(&num_proc, 1, MPI_INT, 0, MPI_COMM_WORLD);
        int size = size_stack_args(&args_stack);
        MPI_Bcast(&size, 1, MPI_INT, 0, MPI_COMM_WORLD);
        int i = 0;
        for(i = 0; i < (world_size - num_proc); i++) {
            if(size == 0)
                break;
            if(my_rank == 0){
                temp_num_cycles = 0;
                args = args_stack.arr[args_stack.top - i];
                SendArgs(&args, num_proc+i, G.V, 56);
                MPI_Status status;
                MPI_Recv(&temp_num_cycles, 1, MPI_INT, num_proc+i, 2, MPI_COMM_WORLD, &status);
                num_cycles += temp_num_cycles;
                RecvTempArgsStack(&temp_args_stack, num_proc+i, G.V);
            } else if(my_rank == num_proc+i){
                RecvArgs(&args, 0,  G.V, 56);
                temp_num_cycles = iterate_over_args(args, G, &temp_args_stack);
                MPI_Send(&temp_num_cycles, 1, MPI_INT, 0, 2, MPI_COMM_WORLD);
                SendTempArgsStack(&temp_args_stack, 0, G.V);
                return 0;
            }
            size--;
        }
        if(my_rank == 0) {
            num_proc += (i);
            if(size != 0) {
                while(size--){
                    struct Args args = args_stack.arr[args_stack.top - i];
                    num_cycles += iterate_over_args(args, G, &temp_args_stack);
                    i++;
                }
                // return num_cycles;
            }
            size = size_stack_args(&args_stack);
            while(size--)
                pop_args(&args_stack);
            push_temp_args_to_main_stack(&temp_args_stack, &args_stack);
        }
        
    }
    return num_cycles;
}

// subroutine to pop the content from 'temp_args_stack' and push them to 'args_stack'
void push_temp_args_to_main_stack(struct Stack_Args *temp_args_stack, struct Stack_Args *args_stack) {
    while(size_stack_args(temp_args_stack) > 0) {
        push_args(args_stack, top_args(temp_args_stack));
        pop_args(temp_args_stack);
    }
}

// suboroutine which returns the number of hamiltonian cycles w.r.t. the give 'args' object
int iterate_over_args(struct Args args, struct Graph G, struct Stack_Args *temp_args_stack) {
    int num_cycles = 0;
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
        iterate_over_unvisited_adjacent(args, G, temp_args_stack);
    }
    return num_cycles;
}

// literally iterates the backtracking algorithm over the adjacent nodes w.r.t. the lately added node in the path
void iterate_over_unvisited_adjacent(struct Args args, struct Graph G, struct Stack_Args *temp_args_stack) {
    int position = args.position;
    int * visit = args.visit;
    struct Stack * Path = args.path;
    
    for(int i = 0; i < G.V; i++) {
        if((G.adj[top(Path)][i] == 1 && visit[i] == 0)) { // for each of the unvisited adjacent vertex, of the vertex at the top of the stack 'Path'
            push(Path, i);
            visit[i] = 1;
            int* visit_copy = (int*)malloc(sizeof(int)*G.V);
            struct Stack* Path_copy = (struct Stack*)malloc(sizeof(struct Stack));
            copy_visit(visit, visit_copy, G.V);
            copy_path(Path, Path_copy);
            args = init_args(position+1, visit_copy, Path_copy);
            push_args(temp_args_stack, args);
            visit[i] = 0; // backtracking step
            pop(Path);
        }
    }
}

// print the 'args' for debugging purposes
void print_args(struct Args args, int V) {
    printf("pos: %d\n", args.position);
    show_stack(args.path);
    for(int i = 0; i < V; i++)
        printf("%d ", args.visit[i]);
    printf("\n");
}

// copy the path 'st' to 'copy_st'
void copy_path(struct Stack * st, struct Stack * copy_st) {
    copy_st->max_size = st->max_size;
    copy_st->top = st->top;
    copy_st->arr = (int*)malloc(sizeof(int)*(st->max_size));
    for(int i = 0; i < st->max_size; i++)
        copy_st->arr[i] = st->arr[i];
}

// copy the array `visit` to `copy_visit`
void copy_visit(int * visit, int * copy_visit, int len) {
    for(int i = 0; i < len; i++)
        copy_visit[i] = visit[i];
}

// MPI_Bcast the graph G to all the processes present in the MPI_COMM_WORLD 
void BroadCastGraph(struct Graph * G, int my_rank) {
    MPI_Bcast(&G->V, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&G->E, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if(my_rank > 0){
        init_graph_auto(G, G->V, G->E);
        for(int i = 0; i < G->V; i++) {
            for(int j = 0; j < G->V; j++)
                G->adj[i][j] = 0;
        }
    }
    MPI_Bcast(&(G->adj[0][0]), G->V*G->V, MPI_INT, 0, MPI_COMM_WORLD);

}

// MPI_Ssend(blocking send) the data 'args' to the 'dest' with tag 'tag' 
void SendArgs(struct Args *args, int dest, int V, int tag) {
    MPI_Ssend(&args->position, 1, MPI_INT, dest, tag+1, MPI_COMM_WORLD);
    MPI_Ssend(args->visit, V, MPI_INT, dest, tag+2, MPI_COMM_WORLD);
    MPI_Ssend(&args->path->top, 1, MPI_INT, dest, tag+3, MPI_COMM_WORLD);
    MPI_Ssend(&args->path->max_size, 1, MPI_INT, dest, tag+4, MPI_COMM_WORLD);
    MPI_Ssend(args->path->arr, args->path->max_size, MPI_INT, dest, tag+5, MPI_COMM_WORLD);
}

// MPI_Recv(blocking recv) the data 'args' from the 'src' with tag 'tag' 
void RecvArgs(struct Args * args, int src, int V, int tag) {
    struct Stack* p = (struct Stack*)malloc(sizeof(struct Stack));
    args->path = p;
    MPI_Status status;
    int top, position, max_size;
    int * visit = (int *)malloc(sizeof(int)*V);
    MPI_Recv(&position, 1, MPI_INT, src, tag+1, MPI_COMM_WORLD, &status);
    MPI_Recv(visit, V, MPI_INT, src, tag+2, MPI_COMM_WORLD, &status);
    MPI_Recv(&top, 1, MPI_INT, src, tag+3, MPI_COMM_WORLD, &status);
    MPI_Recv(&max_size, 1, MPI_INT, src, tag+4, MPI_COMM_WORLD, &status);
    args->visit = visit;
    args->position = position;
    args->path->top = top;
    args->path->max_size = max_size;
    int * arr = (int *)malloc(sizeof(int)*args->path->max_size);
    MPI_Recv(arr, args->path->max_size, MPI_INT, src, tag+5, MPI_COMM_WORLD, &status);
    args->path->arr = arr;
}

// MPI_Ssend(blocking send) the data 'temp_args_stack' to the 'dest'  
void SendTempArgsStack(struct Stack_Args *temp_args_stack, int dest, int V) {
    int size = temp_args_stack->top + 1;
    MPI_Ssend(&size, 1, MPI_INT, dest, 21, MPI_COMM_WORLD);
    while(size--) {
        struct Args args = top_args(temp_args_stack); 
        pop_args(temp_args_stack);
        SendArgs(&args, dest, V, size);
    }
}
// MPI_Recv(blocking recv) the data 'temp_args_stack' from the 'src'  
void RecvTempArgsStack(struct Stack_Args *temp_args_stack, int src, int V) {
    MPI_Status status;
    int size;
    MPI_Recv(&size, 1, MPI_INT, src, 21, MPI_COMM_WORLD, &status);
    while(size--) {
        struct Args args; 
        RecvArgs(&args, src, V, size);
        push_args(temp_args_stack, args);
    }
}
