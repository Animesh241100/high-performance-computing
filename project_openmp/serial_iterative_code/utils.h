#include "types_header.h"

void push(struct Stack *S, int data);
int top(struct Stack *S);
int pop(struct Stack *S);
void show_stack(struct Stack *S);
void push_args(struct Stack_Args *S, struct Args data);
struct Args top_args(struct Stack_Args *S);
int pop_args(struct Stack_Args *S);
void show_stack_args(struct Stack_Args *S, int V);


void check_hamiltonian(struct Graph G);
int num_hamiltonian_cycles(int pos, int * vis, struct Graph G, struct Stack *P);
void print_args(struct Args args, int V);
void copy_path(struct Stack * st, struct Stack * copy_st);
void copy_visit(int * visit, int * copy_visit, int len);


void init_graph_auto(struct Graph *G, int V, int E);
void init_graph(struct Graph *G);
void print_graph(struct Graph *G);
