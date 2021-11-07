#include "types_header.h"

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


void check_hamiltonian(struct Graph G);
int num_hamiltonian_cycles(int pos, int * vis, struct Graph G, struct Stack *P);
int iterate_over_args(struct Stack_Args * args_stack, struct Args args, struct Graph G);
void iterate_over_unvisited_adjacent(struct Args args, struct Graph G, struct Stack_Args *args_stack);
void print_args(struct Args args, int V);
void copy_path(struct Stack * st, struct Stack * copy_st);
void copy_visit(int * visit, int * copy_visit, int len);


void init_graph_auto(struct Graph *G, int V, int E);
void init_graph(struct Graph *G);
void print_graph(struct Graph *G);
