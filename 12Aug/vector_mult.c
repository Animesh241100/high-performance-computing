
#include <omp.h>
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
    int num_threads [] = {1, 2, 4, 6, 8, 10, 12, 14, 16, 32, 64, 80, 100, 128, 200, 256, 512};
    int num_threads_size = 17;
    float running_time_thread[num_threads_size];
    for(int j = 0; j < num_threads_size; j++) {
        start_time = omp_get_wtime();
        omp_set_num_threads(num_threads[j]);
        #pragma omp parallel
        {
            #pragma omp for 
            for(int i = 0; i < n; i++) {
                int tid = omp_get_thread_num();
                nthreads = omp_get_num_threads();
                a[i] = rand() % 1000;
                b[i] = rand() % 1000;
                for(int j = 0; j < m; j++)
                    c[i] = a[i] * b[i];
                // printf("ID: %d, i: %d, a[i]: %d, b[i]: %d, c[i]: %d\n", tid, i, a[i], b[i], c[i]);
            }
        end_time = omp_get_wtime();
        running_time_thread[j] = (end_time - start_time);
        }
    }
    for(int j = 0; j < num_threads_size; j++)
        printf("num_threads: %d Running time: %f\n", num_threads[j], running_time_thread[j]);
    return 0;
}