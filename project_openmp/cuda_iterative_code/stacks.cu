#include "utils.cuh"


__host__ __device__ void push(struct Stack *S, int data) {
    if(S->top > S->max_size - 1)
        printf("Stack overflow push\n");
    else {
        S->top++;
        S->arr[S->top] = data;
    }
}

__host__ __device__ int top(struct Stack *S) {
    if(S->top > -1) 
        return S->arr[S->top];
    printf("Stack is empty!\n");
    return EMPTY;

}

__host__ __device__ int pop(struct Stack *S) {
    if(S->top > -1) {
        int val = S->arr[S->top];
        S->top--;
        return val;
    }
    printf("Popping out of empty stack\n");
    return EMPTY;
}

__host__ __device__ void show_stack(struct Stack *S) {
    printf("[");
    for(int i = 0; i <= S->top; i++) {
        printf("%d ", S->arr[i]);
    }
    printf("\b<---top\n");
}

/*********************************************************************************/


__host__ __device__ void push_args(struct Stack_Args *S, struct Args data) {
    if(S->top > S->max_size - 1)
        printf("Stack overflow push args\n");
    else {
        S->top++;
        S->arr[S->top] = data;
    }
}

__host__ __device__ struct Args top_args(struct Stack_Args *S) {
    if(S->top > -1) 
        return S->arr[S->top];
    printf("lol Stack is empty!\n");
    struct Args args;
    // cudaMallocManaged((void**)&args, sizeof(struct Args));
    args.position = EMPTY;
    return args;
}

__host__ __device__ int pop_args(struct Stack_Args *S) {
    if(S->top > -1) {
        S->top--;
        return 1;
    }
    printf("Popping out of empty stack\n");
    return EMPTY;
}

void show_stack_args(struct Stack_Args *S, int V) {
    printf("[");
    for(int i = 0; i <= S->top; i++) {
        print_args(S->arr[i], V);
        printf("___]\n");
    }
    printf("\b<-----------------top args\n");
}

__host__ __device__ int size_stack_args(struct Stack_Args *S) {
    return S->top + 1;
}


void init_args_stack(struct Stack_Args *S) {
    cudaMallocManaged((void**)S, sizeof(struct Stack_Args));
    S->max_size = 1000;
    S->top = -1;
    cudaMallocManaged((void**)S->arr, sizeof(struct Args)*(S->max_size));
}

__host__ __device__ void init_args(struct Args * args, int pos, int * visit, struct Stack* p) {
    args->path = p;
    args->visit = visit;
    args->position = pos;
}