#include<stdio.h>
#include<string.h>
#include<stdlib.h>
#include<time.h>
#include <omp.h>


int main() {
    fun();
}


void fun(int x) {
    #pragma omp for
    for(int i = 0; i < 5; i++) {
        fun(i);
    }
}
