#include<stdio.h>
#include<string.h>
#include<stdlib.h>
#include<time.h>
#include<chrono>

#define EMPTY -20

// Represents a stack of integers(nodes)
struct Stack {
    int * arr;
    int max_size;
    int top;
};

// Represents an undirected graph
struct Graph {
    int E;
    int V;
    int ** adj; // adjacency matrix
};

// Represents an argument for the iterative backtracking algorithm
struct Args {
    int position;
    int * visit;
    struct Stack *path;
};

// Represents a stack of a 'struct Args'
struct Stack_Args {
    struct Args * arr;
    int max_size;
    int top;
};

