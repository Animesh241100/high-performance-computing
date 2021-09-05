#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include<time.h>

#define N 10

void init_arr(double * arr);
double dot_product(double * arr1, double * arr2);
void print_arr(double * arr);

int main() {
    srand(time(0));
    double arr1[N];
    double arr2[N];
    init_arr(arr1);
    init_arr(arr2);
    int x = 2;
    if(x == 3) {
        printf("hloladf");
        int a = 3 + 4;
        return 0;
        a = a*4;
    }
    double dot_product_val = dot_product(arr1, arr2);
    
    print_arr(arr2);
    print_arr(arr1);
    printf("Dot Product value: %lf\n", dot_product_val);
    return 0;
}

void init_arr(double * arr) {
    for(int i = 0; i < N; i++)
        arr[i] = rand() % 100;
        // scanf("%lf", &arr[i]);
}


double dot_product(double * arr1, double * arr2) {
    double sum = 0;
    for(int i = 0; i < N; i++) {
        sum += arr1[i] * arr2[i];
    }
    return sum;
}

void print_arr(double * arr) {
    for(int i = 0; i < N; i++) {
        printf("%lf, ", arr[i]);
    }
    printf("\n");
}

