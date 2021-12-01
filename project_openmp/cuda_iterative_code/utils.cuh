#include "types_header.cuh"

__host__ __device__ void push(struct Stack *S, int data);
__host__ __device__ int top(struct Stack *S);
__host__ __device__ int pop(struct Stack *S);
__host__ __device__ void show_stack(struct Stack *S);
__host__ __device__ void push_args(struct Stack_Args *S, struct Args data);
__host__ __device__ struct Args top_args(struct Stack_Args *S);
__host__ __device__ int pop_args(struct Stack_Args *S);
void show_stack_args(struct Stack_Args *S, int V);
__host__ __device__ int size_stack_args(struct Stack_Args *S);
void init_args_stack(struct Stack_Args *S);
__host__ __device__ void init_args(struct Args * args, int pos, int * visit, struct Stack* p);


void check_hamiltonian(struct Graph G, int GRID_SIZE, int BLOCK_SIZE);
int num_hamiltonian_cycles(int pos, int * vis, struct Graph G, struct Stack *P, int * num_procs);
void copy_graph_to_device(struct Graph * d_G, struct Graph * G);
__host__ __device__ void push_temp_args_to_main_stack(struct Stack_Args *temp_args_stack, struct Stack_Args *args_stack);
__device__ int iterate_over_args(struct Args args, struct Graph G);
__device__ void iterate_over_unvisited_adjacent(struct Args args, struct Graph G);
void print_args(struct Args args, int V);
__host__ __device__ void copy_path(struct Stack * st, struct Stack * copy_st);
__host__ __device__ void copy_visit(int * visit, int * copy_visit, int len);


void init_graph_auto(struct Graph *G, int V, int E);
void init_graph(struct Graph *G);
__device__ void print_graph(struct Graph *G);
int Allocate2DMemory(int ***array, int n, int m);
int Free2DMemory(int ***array);
