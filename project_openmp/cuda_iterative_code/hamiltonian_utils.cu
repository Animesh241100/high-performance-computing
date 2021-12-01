#include "utils.cuh"

__managed__ struct Stack_Args temp_args_stack;

// prints the number of hamiltonian cycles in the undirected graph G
void check_hamiltonian(struct Graph G, int GRID_SIZE, int BLOCK_SIZE) {
    int num_procs = GRID_SIZE * BLOCK_SIZE;
    int * visit;
    cudaMallocManaged((void**)&visit, sizeof(int) * G.V);
    for(int i = 0; i < G.V; i++)
        visit[i] = 0;
    struct Stack * Path;
    cudaMallocManaged((void**)&Path, sizeof(struct Stack));
    Path->max_size = G.V + 1;
    Path->top = -1;
    cudaMallocManaged((void**)&Path->arr, sizeof(int)*(G.V + 1));
    push(Path, 0);
    visit[0] = 1;
    int num_cycles = num_hamiltonian_cycles(1, visit, G, Path, &num_procs);
    printf("#Hamiltonian Cycles = %d\n", num_cycles); 
}

// kernel function to iterate the arguments of the backtracking algorithm in parallel threads
__global__ void gpu_iterations(struct Graph G, struct Stack_Args args_stack, int * num_cycles, int *size) {
    // division of work among threads
    struct Args args = args_stack.arr[args_stack.top - threadIdx.x];
    int val = iterate_over_args(args, G);
    atomicAdd(num_cycles, val);
    __syncthreads();

    // load balancing by the thread #0
    if(threadIdx.x == 0) {
        for(int i = blockDim.x; i < *size; i++) {
            args = args_stack.arr[args_stack.top - i];
            val = iterate_over_args(args, G);
            atomicAdd(num_cycles, val);
        }
    }
}

// sets the grid dimension based upon the size of args_stack and number of processes available
void get_grid_dimensions(int size, int * num_procs, int * temp_grid_size, int * temp_block_size) {
    *temp_grid_size = 1;
    *temp_block_size = 1;
    if(*num_procs == 0)
        return;
    *temp_grid_size = 1;
    if(*num_procs >= size) {
        *temp_block_size = size;
        *num_procs -= size;
    } else {
        *temp_block_size = *num_procs;
        *num_procs = 0;
    }
}

// returns the number of hamiltonian cycles in the graph G
int num_hamiltonian_cycles(int pos, int * vis, struct Graph G, struct Stack *P, int * num_procs) {
    int temp_grid_size, temp_block_size;
    int * num_cycles;
    int * size;
    cudaMallocManaged((void**)&num_cycles, sizeof(int));
    cudaMallocManaged((void**)&size, sizeof(int));
    *num_cycles = 0;
    struct Stack_Args args_stack;
    init_args_stack(&args_stack);
    init_args_stack(&temp_args_stack);
    struct Args args;
    cudaMallocManaged((void**)&args, sizeof(struct Args));
    init_args(&args, pos, vis, P);
    push_args(&args_stack, args);
    while(size_stack_args(&args_stack) > 0) {  // run the loop code stack.size times, these iterations don't have dependency
        *size = size_stack_args(&args_stack);
        get_grid_dimensions(*size, num_procs, &temp_grid_size, &temp_block_size);
        struct Graph d_G;
        copy_graph_to_device(&d_G, &G);
        // printf("Grid size: %d, block size: %d\n", temp_grid_size, temp_block_size);
        gpu_iterations<<<temp_grid_size, temp_block_size>>>(d_G, args_stack, num_cycles, size);
        cudaError_t cudaerr = cudaDeviceSynchronize();
        if (cudaerr != cudaSuccess)
            printf("kernel launch failed with error \"%s\".\n", cudaGetErrorString(cudaerr));

        int size_stack = size_stack_args(&args_stack);
        while(size_stack--)
            pop_args(&args_stack);
        push_temp_args_to_main_stack(&temp_args_stack, &args_stack);
        cudaFree(d_G.adj);
    }
    cudaFree(num_cycles);
    cudaFree(size);
    cudaFree(&args);
    return *num_cycles;
}

// copy the graph G to the device memory d_G
void copy_graph_to_device(struct Graph * d_G, struct Graph * G) {
    d_G->E = G->E;
    d_G->V = G->V;
    int ** h_adj, ** d_adj;
    h_adj = (int **)malloc(G->V * sizeof(int*));
    cudaMalloc((void**)&d_adj, G->V * sizeof(int*));
    for(int i = 0; i < G->V; i++){
        cudaMalloc((void**)&h_adj[i], G->V * sizeof(int));
        cudaMemcpy(h_adj[i], G->adj[i], G->V * sizeof(int), cudaMemcpyHostToDevice);
    }
    cudaMemcpy(d_adj, h_adj, G->V * sizeof(int*), cudaMemcpyHostToDevice);
    d_G->adj = d_adj;
}


// subroutine to pop the content from 'temp_args_stack' and push them to 'args_stack'
__host__ __device__ void push_temp_args_to_main_stack(struct Stack_Args *temp_args_stack, struct Stack_Args *args_stack) {
    while(size_stack_args(temp_args_stack) > 0) {
        push_args(args_stack, top_args(temp_args_stack));
        pop_args(temp_args_stack);
    }
}


// suboroutine which returns the number of hamiltonian cycles w.r.t. the given 'args' object
__device__ int iterate_over_args(struct Args args, struct Graph G) {
    int num_cycles = 0;
    int position = args.position;
    struct Stack * Path = args.path;
    if(position == G.V) { // the base case when we finally get to know whether we came to the end of a path
        if(G.adj[top(Path)][0]) {
            push(Path, 0);
            for(int i = 0; i < blockDim.x; i++) {
                if(threadIdx.x == i) {
                    printf("[");
                    for(int i = 0; i <= Path->top; i++) {
                        printf("%d ", Path->arr[i]);
                    }
                    printf("\b]<---Top\n", threadIdx.x);
                }
                __syncthreads();
            }
            pop(Path);
            num_cycles++;
        }
    } else {
        iterate_over_unvisited_adjacent(args, G);
    }
    return num_cycles;
}

// iterate the 'args' over the unvisited adjacent nodes
__device__ void iterate_over_unvisited_adjacent(struct Args args, struct Graph G) {
    int position = args.position;
    int * visit = args.visit;
    struct Stack * Path = args.path;
    struct Stack_Args temp_args_stack2;
    temp_args_stack2.max_size = 2000;
    temp_args_stack2.top = -1;
    temp_args_stack2.arr = (struct Args*)malloc(sizeof(struct Args)*(temp_args_stack2.max_size));
    for(int i = 0; i < G.V; i++) {
        if((G.adj[top(Path)][i] && !visit[i])) { // for each of the unvisited adjacent vertex, of the vertex at the top of the stack 'Path'
            push(Path, i);
            visit[i] = 1;
            int* visit_copy = (int*)malloc(sizeof(int)*G.V);
            struct Stack* Path_copy = (struct Stack*)malloc(sizeof(struct Stack));
            copy_visit(visit, visit_copy, G.V);
            copy_path(Path, Path_copy);
            init_args(&args, position+1, visit_copy, Path_copy);
            push_args(&temp_args_stack2, args);
            visit[i] = 0; // backtracking step
            pop(Path);
        }
    }
    __syncthreads();

    for(int i = 0; i < blockDim.x; i++) {
        if(threadIdx.x == i) {
            push_temp_args_to_main_stack(&temp_args_stack2, &temp_args_stack);
        }
        __syncthreads();
    }
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