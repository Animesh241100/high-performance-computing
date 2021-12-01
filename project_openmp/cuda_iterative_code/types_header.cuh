#include<stdio.h>
#include<string.h>
#include<stdlib.h>
#include<time.h>

#define EMPTY -20

class Lock {
  public:
    int * mutex;
    // __device__ Lock() {
    //     mutex = (int*)malloc(sizeof(int));
    //     *mutex = 0;
    // }
    __device__ void lock() {
        while(atomicCAS(mutex, 0, 1) != 0);
    }

    __device__ void unlock() {
        atomicExch(mutex, 0);
    }

    // __device__ ~Lock() {
    //     free(mutex);
    // }
};

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

