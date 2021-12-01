#include "utils.cuh"

__managed__ struct Stack_Args temp_args_stack;
__managed__ int is_locked = 0;
__managed__ int var_my = 0;

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
    // int num_cycles = 0;
    int num_cycles = num_hamiltonian_cycles(1, visit, G, Path, &num_procs);
    printf("#Hamiltonian2 Cycles = %d\n", num_cycles); 
}

// __global__ void gpu_iterations(struct Graph * G, struct Stack_Args * args_stack, int * num_cycles, struct Stack_Args * temp_args_stack) {
// __global__ void gpu_iterations(struct Graph G, struct Stack_Args args_stack) {
__global__ void gpu_iterations(struct Graph G, struct Stack_Args args_stack, int * d_num_cycles, int * size) {
    struct Args args = args_stack.arr[args_stack.top - threadIdx.x];
    printf("\n--------------------------------\n");
    for(int i = 0; i < blockDim.x; i++) {
        if(threadIdx.x == i) {
            printf("tid: %d, top: %d, dev_visit: ", threadIdx.x, top(args.path));
            for(int i = 0;  i < G.V; i++)
                printf("%d ", args_stack.arr[args_stack.top - threadIdx.x].visit[i]);
            printf("\n");
            show_stack(args_stack.arr[args_stack.top - threadIdx.x].path);
        }
        __syncthreads();
    }
    int val = iterate_over_args(args, G);
    atomicAdd(d_num_cycles, val);
    // __syncthreads();
    // if(threadIdx.x == 0) {
    //     printf("num threads %d size %d\n", blockDim.x, size);
    // }
    printf("I %d have done sum %d\n", threadIdx.x, val);
}

void get_grid_dimensions(int size, int * num_procs, int * temp_grid_size, int * temp_block_size) {
    *temp_grid_size = 1;
    *temp_block_size = 0;
    if(*num_procs == 0)
        return;
    if(*num_procs >= size) {
        *temp_block_size = size;
        *num_procs -= size;
    } else {
        *temp_block_size = *num_procs;
        *num_procs = 0;
    }
}

int num_hamiltonian_cycles(int pos, int * vis, struct Graph G, struct Stack *P, int * num_procs) {
    int temp_grid_size, temp_block_size;
    int * num_cycles,  * size;
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
        printf("Grid size: %d, block size: %d size: %d num_procs: %d\n", temp_grid_size, temp_block_size, *size, *num_procs);
        struct Graph d_G;
        struct Stack_Args d_args_stack;
        struct Stack_Args * d_temp_args_stack;
        copy_graph_to_device(&d_G, &G);
        // var_my = 0;

        gpu_iterations<<<temp_grid_size, temp_block_size>>>(d_G, args_stack, num_cycles, size);
        cudaError_t cudaerr = cudaDeviceSynchronize();
        if (cudaerr != cudaSuccess)
            printf("kernel launch failed with error \"%s\".\n", cudaGetErrorString(cudaerr));

        // size -= temp_grid_size * temp_block_size; 
        // if(size != 0) {
        //     int i = temp_grid_size * temp_block_size;
        //     printf("size %d size real %d size real temp %d\n", size, size_stack_args(&args_stack), size_stack_args(&temp_args_stack));
            // struct Args args = args_stack.arr[4];
            // printf("well %d\n", args_stack.arr[0].path->max_size);
            // while(size--){
                // struct Args args = args_stack.arr[args_stack.top - i];
            //     printf("hi size %d idx %d top%d\n", size, args_stack.top - i, 2);
            //     *num_cycles += host_iterate_over_args(args, G);
            //     i++;
            // }
        // }  
        *size = size_stack_args(&args_stack);
        printf("BEF: args stack size %d - temp args stack size %d, num: %d \n", size_stack_args(&args_stack), size_stack_args(&temp_args_stack), *num_cycles);
        while(*size--)
            pop_args(&args_stack);
        push_temp_args_to_main_stack(&temp_args_stack, &args_stack);
        printf("AFT: args stack size %d - temp args stack size %d\n", size_stack_args(&args_stack), size_stack_args(&temp_args_stack));
        cudaFree(d_G.adj);
        // execute_cuda_free(d_G, d_args_stack, d_temp_args_stack, d_num_cycles);
    }
    return *num_cycles;
}

void execute_cuda_free(struct Graph * d_G, struct Stack_Args * d_args_stack, struct Stack_Args * d_temp_args_stack, int * d_num_cycles) {
    for(int i = 0; i < d_G->V; i++)
        cudaFree(d_G->adj[i]);
    cudaFree(d_G);
    for(int i = 0; i < d_args_stack->max_size; i++)
        cudaFree(&d_args_stack->arr[i]);
    for(int i = 0; i < d_temp_args_stack->max_size; i++)
        cudaFree(&d_temp_args_stack->arr[i]);
    cudaFree(d_num_cycles);
}

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

void copy_stack_to_device(struct Stack_Args * d_args_stack, struct Stack_Args * args_stack, int V) {
    d_args_stack->max_size = args_stack->max_size;
    d_args_stack->top = args_stack->top;
    struct Args * h_args, * d_args;
    h_args = (struct Args *)malloc(args_stack->max_size * sizeof(struct Args));
    cudaMalloc((void**)&d_args, args_stack->max_size * sizeof(struct Args));
    struct Args a;
    for(int i = 0; i < args_stack->max_size; i++) {
        cudaMalloc((void **)&a, sizeof(struct Args));
        int x = 10;
        cudaMemcpy(&a.position, &x, sizeof(int), cudaMemcpyHostToDevice);
        // cudaMalloc((void**)&h_args[i].path, sizeof(int) * V);
        // cudaMemcpy(&h_args[i].path, &args_stack->arr[i].path, sizeof(int)*V, cudaMemcpyHostToDevice);
    }
    cudaMemcpy(d_args, h_args, args_stack->max_size * sizeof(struct Args), cudaMemcpyHostToDevice);
    d_args_stack->arr = d_args;
    // return a;
}

void copy_stack_to_host(struct Stack_Args * args_stack, struct Stack_Args * d_args_stack, int V) {
    cudaMemcpy(&args_stack->max_size, &d_args_stack->max_size, sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(&args_stack->top, &d_args_stack->top, sizeof(int), cudaMemcpyDeviceToHost);
    for(int i = 0; i < d_args_stack->max_size; i++) {
        cudaMemcpy(&args_stack->arr[i].position, &d_args_stack->arr[i].position, sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(&args_stack->arr[i].visit, &d_args_stack->arr[i].visit, sizeof(int)*V, cudaMemcpyDeviceToHost);
        cudaMemcpy(&args_stack->arr[i].path, &d_args_stack->arr[i].path, sizeof(int)*args_stack->arr[i].path->max_size, cudaMemcpyDeviceToHost);
    }
}


// subroutine to pop the content from 'temp_args_stack' and push them to 'args_stack'
__host__ __device__ void push_temp_args_to_main_stack(struct Stack_Args *temp_args_stack, struct Stack_Args *args_stack) {
    while(size_stack_args(temp_args_stack) > 0) {
        push_args(args_stack, top_args(temp_args_stack));
        pop_args(temp_args_stack);
    }
}

// suboroutine which returns the number of hamiltonian cycles w.r.t. the give 'args' object
__host__ __device__ int iterate_over_args(struct Args args, struct Graph G) {
    printf("iterate over args top %d, pretop %d, pos %d\n", top(args.path), args.path->arr[args.path->top - 1], args.position);
    int num_cycles = 0;
    int position = args.position;
    int * visit = args.visit;
    struct Stack * Path = args.path;
    if(position == G.V) { // the base case when we finally get to know whether we came to the end of a path
        if(G.adj[top(Path)][0]) {
            push(Path, 0);
            #ifdef __CUDA_ARCH__
            for(int i = 0; i < blockDim.x; i++) {
                if(threadIdx.x == i)
                    show_stack(Path);
                __syncthreads();
            }
            #else
            show_stack(Path);
            #endif
            pop(Path);
            num_cycles++;
        }
    } else {
        iterate_over_unvisited_adjacent(args, G);
    }
    // #ifdef _CUDA_ARCH__
    printf("done2 tid\n");
    // #endif
    return num_cycles;
}

__host__ __device__ void iterate_over_unvisited_adjacent(struct Args args, struct Graph G) {
    int position = args.position;
    int * visit = args.visit;
    struct Stack * Path = args.path;
    int top_val = top(args.path);
    struct Stack_Args temp_args_stack2;
    temp_args_stack2.max_size = 100;
    temp_args_stack2.top = -1;
    temp_args_stack2.arr = (struct Args*)malloc(sizeof(struct Args)*(100));
    for(int i = 0; i < G.V; i++) {
        if((G.adj[top(Path)][i] && !visit[i])) { // for each of the unvisited adjacent vertex, of the vertex at the top of the stack 'Path'
            push(Path, i);
            visit[i] = 1;
            int* visit_copy = (int*)malloc(sizeof(int)*G.V);
            struct Stack* Path_copy = (struct Stack*)malloc(sizeof(struct Stack));
            copy_visit(visit, visit_copy, G.V);
            copy_path(Path, Path_copy);
            // struct Args * args = (struct Args *)malloc(sizeof(struct Args));
            init_args(&args, position+1, visit_copy, Path_copy);
            #ifdef  __CUDA_ARCH__
            printf("tid: %d size: %d top: %d new_top: %d pos: %d\n", threadIdx.x, size_stack_args(&temp_args_stack), top_val, top(args.path), args.position);
            #else
            printf("size: %d top: %d new_top: %d pos: %d\n", size_stack_args(&temp_args_stack), top_val, top(args.path), args.position);
            #endif
            push_args(&temp_args_stack2, args);
            visit[i] = 0; // backtracking step
            pop(Path);
        }
    }
    #ifdef __CUDA_ARCH__
    __syncthreads();

    for(int i = 0; i < blockDim.x; i++) {
        if(threadIdx.x == i) {
            // var_my += 1;
            // printf("nahi ho gaya*************** temp2 %d temp %d var_my %d \n", size_stack_args(&temp_args_stack2), size_stack_args(&temp_args_stack), var_my);
            push_temp_args_to_main_stack(&temp_args_stack2, &temp_args_stack);
            // printf("ho gaya*************** temp2 %d temp %d  var_my %d \n", size_stack_args(&temp_args_stack2), size_stack_args(&temp_args_stack), var_my);
            // printf("thread %d did it to %d\n", threadIdx.x, var_my);
        }
        __syncthreads();
    }
    printf("done tid %d\n", threadIdx.x);

    #else
    push_temp_args_to_main_stack(&temp_args_stack2, &temp_args_stack);
    printf("hix\n");
    #endif

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

int host_iterate_over_args(struct Args args, struct Graph G) {
    int num_cycles = 0;
    int position = args.position;
    int * visit = args.visit;
    struct Stack * Path = args.path;
    // printf("top %d\n", args.path->top);
    if(position == G.V) { // the base case when we finally get to know whether we came to the end of a path
        if(G.adj[top(Path)][0]) {
            push(Path, 0);
            show_stack(Path);
            pop(Path);
            num_cycles++;
        }
    } else {
        host_iterate_over_unvisited_adjacent(args, G);
    }
    return num_cycles;
}

void host_iterate_over_unvisited_adjacent(struct Args args, struct Graph G) {
    int position = args.position;
    int * visit = args.visit;
    struct Stack * Path = args.path;
    // int top_val = top(args.path);
    // struct Stack_Args temp_args_stack2;
    // temp_args_stack2.max_size = 100;
    // temp_args_stack2.top = -1;
    // temp_args_stack2.arr = (struct Args*)malloc(sizeof(struct Args)*(100));
    // for(int i = 0; i < G.V; i++) {
    //     if((G.adj[top(Path)][i] && !visit[i])) { // for each of the unvisited adjacent vertex, of the vertex at the top of the stack 'Path'
    //         push(Path, i);
    //         visit[i] = 1;
    //         int* visit_copy = (int*)malloc(sizeof(int)*G.V);
    //         struct Stack* Path_copy = (struct Stack*)malloc(sizeof(struct Stack));
    //         copy_visit(visit, visit_copy, G.V);
    //         copy_path(Path, Path_copy);
    //         // struct Args * args = (struct Args *)malloc(sizeof(struct Args));
    //         init_args(&args, position+1, visit_copy, Path_copy);
    //         printf("size: %d top: %d new_top: %d pos: %d\n", size_stack_args(&temp_args_stack), top_val, top(args.path), args.position);
    //         push_args(&temp_args_stack2, args);
    //         visit[i] = 0; // backtracking step
    //         pop(Path);
    //     }
    // }
    // push_temp_args_to_main_stack(&temp_args_stack2, &temp_args_stack);
    printf("hix2\n");
}