#include<stdio.h>
#include<string.h>
#include<stdlib.h>
#include<time.h>
#include<mpi.h>

#define EMPTY -20  // Represents an empty stack

// Represents a Stack of 'max_size' integers
struct Stack {
    int * arr;
    int max_size;
    int top;
};

// Represents a Graph of 'E' edges and 'V' nodes stored as an adjancency matrix
struct Graph {
    int E;
    int V;
    int ** adj; // adjacency matrix
};

// Represents a Args object which is used to pack the arguments required to calculate the #hamiltonian paths w.r.t. the graph 'G'
struct Args {
    int position;
    int * visit;
    struct Stack *path;
};

// Represents a Stack object of 'max_size' of 'Args' objects
struct Stack_Args {
    struct Args * arr;
    int max_size;
    int top;
};

