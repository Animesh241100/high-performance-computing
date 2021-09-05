
#include <stdio.h>
#include <stdlib.h>
#include<time.h>

#define n 100000
#define m 1000

int main(int argc, char* argv[])
{
    double start_time, end_time, running_time;
    int nthreads;
    int a[n], b[n], c[n];
    srand(time(0));
    for(int i = 0; i < n; i++) {
        a[i] = rand() % 1000;
        b[i] = rand() % 1000;
        for(int j = 0; j < m; j++)
            c[i] = a[i] + b[i];
        printf("i: %d, a[i]: %d, b[i]: %d, c[i]: %d\n" ,i, a[i], b[i], c[i]);
    }
    return 0;
}