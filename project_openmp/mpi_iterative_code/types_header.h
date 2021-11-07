#include<stdio.h>
#include<string.h>
#include<stdlib.h>
#include<time.h>

#define EMPTY -20


struct Stack {
    int * arr;
    int max_size;
    int top;
};

struct Graph {
    int E;
    int V;
    int ** adj; // adjacency matrix
};

struct Args {
    int position;
    int * visit;
    struct Stack *path;
};

struct Stack_Args {
    struct Args * arr;
    int max_size;
    int top;
};

