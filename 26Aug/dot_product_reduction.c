#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include<time.h>

#define N 100000

void init_arr(double * arr);
void print_arr(double * arr);

int main() {
    srand(time(0));
    double arr1[N];
    double arr2[N];
    init_arr(arr1);
    init_arr(arr2);
    double start_time, end_time;
    int num_threads [] = {1, 2, 4, 6, 8, 10, 12, 14, 16, 32, 64, 80, 100, 128, 200, 256, 512};
    int num_threads_size = 17;
    double running_time_thread[num_threads_size];


    double sum;
    
    for(int j = 0; j < num_threads_size; j++) {
        start_time = omp_get_wtime();
        omp_set_num_threads(num_threads[j]);
        sum = 0;
        #pragma omp parallel for reduction(+:sum)
        for(int i = 0; i < N; i++) {
            sum += arr1[i] * arr2[i];
        }
        end_time = omp_get_wtime();
        running_time_thread[j] = (end_time - start_time);
        printf("#threads: %d, Dot Product value: %f\n", num_threads[j], sum);
    }

    for(int j = 0; j < num_threads_size; j++)
        printf("num_threads: %d Running time: %f\n", num_threads[j], running_time_thread[j]);
    return 0;

    // print_arr(arr2);
    // print_arr(arr1);
}

void init_arr(double * arr) {
    for(int i = 0; i < N; i++)
        arr[i] = (double)(rand() % 100000000) / (double)(10000);
        // scanf("%lf", &arr[i]);
}

void print_arr(double * arr) {
    for(int i = 0; i < N; i++) {
        printf("%f, ", arr[i]);
    }
    printf("\n");
}

