#include "types_header.h"


/***************************** For 'stacks.c'  ********************************/

void push(struct Stack *S, int data);
int top(struct Stack *S);
int pop(struct Stack *S);
void show_stack(struct Stack *S);
void push_args(struct Stack_Args *S, struct Args data);
struct Args top_args(struct Stack_Args *S);
int pop_args(struct Stack_Args *S);
void show_stack_args(struct Stack_Args *S, int V);
int size_stack_args(struct Stack_Args *S);
struct Stack_Args init_args_stack();
struct Args init_args(int pos, int * visit, struct Stack* p);


/************************** For 'hamiltonian_utils.c' *************************/

void check_hamiltonian(struct Graph G, int my_rank);
int num_hamiltonian_cycles(int pos, int * vis, struct Graph G, struct Stack *P, int my_rank);
void push_temp_args_to_main_stack(struct Stack_Args *temp_args_stack, struct Stack_Args *args_stack);
int iterate_over_args(struct Args args, struct Graph G, struct Stack_Args *temp_args_stack);
void iterate_over_unvisited_adjacent(struct Args args, struct Graph G, struct Stack_Args *temp_args_stack);
void print_args(struct Args args, int V);
void copy_path(struct Stack * st, struct Stack * copy_st);
void copy_visit(int * visit, int * copy_visit, int len);
void BroadCastGraph(struct Graph * G, int my_rank);
void SendArgs(struct Args *args, int dest, int V, int tag);
void RecvArgs(struct Args *args, int src, int V, int tag);
void SendTempArgsStack(struct Stack_Args *temp_args_stack, int dest, int V);
void RecvTempArgsStack(struct Stack_Args *temp_args_stack, int src, int V);

/******************************* For 'graphs.c'  ******************************/

void init_graph_auto(struct Graph *G, int V, int E);
void init_graph(struct Graph *G);
void print_graph(struct Graph *G);
int Allocate2DMemory(int ***array, int n, int m);
int Free2DMemory(int ***array);

