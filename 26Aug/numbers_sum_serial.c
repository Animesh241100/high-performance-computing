#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include<time.h>

#define N 10

void init_arr(double * arr);
void print_arr(double * arr);

int main() {
    srand(time(0));
    double arr1[N];
    init_arr(arr1);

    double sum_val = 0;
    for(int i = 0; i < N; i++) {
        sum_val += arr1[i];
    }
    print_arr(arr1);
    printf("Sum value: %lf\n", sum_val);
    return 0;
}

void init_arr(double * arr) {
    for(int i = 0; i < N; i++)
        scanf("%lf", &arr[i]);
        // arr[i] = rand() % 100;
}



void print_arr(double * arr) {
    for(int i = 0; i < N; i++) {
        printf("%lf, ", arr[i]);
    }
    printf("\n");
}

