#include "utils.h"

// push 'data' into the stack 'S'
void push(struct Stack *S, int data) {
    if(S->top > S->max_size - 1)
        printf("Stack overflow\n");
    else {
        S->top++;
        S->arr[S->top] = data;
    }
}

// lookup the top of the stack 'S'
int top(struct Stack *S) {
    if(S->top > -1) 
        return S->arr[S->top];
    printf("Stack is empty!\n");
    return EMPTY;

}

// pop out of the stack 'S'
int pop(struct Stack *S) {
    if(S->top > -1) {
        int val = S->arr[S->top];
        S->top--;
        return val;
    }
    printf("Popping out of empty stack\n");
    return EMPTY;
}

// print the stack 'S'
void show_stack(struct Stack *S) {
    printf("[");
    for(int i = 0; i <= S->top; i++) {
        printf("%d ", S->arr[i]);
    }
    printf("\b<---top\n");
}

/*********************************************************************************/

// push 'data' into the Stack_Args 'S'
void push_args(struct Stack_Args *S, struct Args data) {
    if(S->top > S->max_size - 1)
        printf("Stack overflow\n");
    else {
        S->top++;
        S->arr[S->top] = data;
    }
}

// lookup the top of the Stack_Args 'S'
struct Args top_args(struct Stack_Args *S) {
    if(S->top > -1) 
        return S->arr[S->top];
    printf("lol Stack is empty!\n");
    struct Args * args = (struct Args*)malloc(sizeof(struct Args));
    args->position = EMPTY;
    return *args;
}

// pop out of the Stack_Args 'S'
int pop_args(struct Stack_Args *S) {
    if(S->top > -1) {
        S->top--;
        return 1;
    }
    printf("Popping out of empty stack\n");
    return EMPTY;
}

// print the contents of the Stack_Args S
void show_stack_args(struct Stack_Args *S, int V) {
    printf("[");
    for(int i = 0; i <= S->top; i++) {
        print_args(S->arr[i], V);
        printf("___]\n");
    }
    printf("\b<-----------------top args\n");
}

// print the size of the Stack_Args S
int size_stack_args(struct Stack_Args *S) {
    return S->top + 1;
}

// return an intialized Stack_Args object
struct Stack_Args init_args_stack() {
    struct Stack_Args args_stack;
    args_stack.max_size = 100;
    args_stack.top = -1;
    args_stack.arr = (struct Args*)malloc(sizeof(struct Args)*(100));
    return args_stack;
}

// return an initialised Args object
struct Args init_args(int pos, int * visit, struct Stack* p) {
    struct Args args;
    args.path = p;
    args.visit = visit;
    args.position = pos;
    return args;
}