
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include<time.h>

#define n 300
#define m 1000

double get_value(double A[][n], int ptr_A, double B[][n], int ptr_B);

int main() {
    srand(time(0));
    double A[n][n], B[n][n], C[n][n];
    double start_time, end_time, running_time;
    int num_threads [] = {1, 2, 4, 6, 8, 10, 12, 14, 16, 32, 64, 80, 100, 128, 200, 256, 512};
    int num_threads_size = 17;
    float running_time_thread[num_threads_size];
    for(int i = 0; i < n; i++) {
        for(int j = 0; j < n; j++) {
            A[i][j] = (double)(rand() % 100000) / (double)1000;
            B[i][j] = (double)(rand() % 100000) / (double)1000;
        }
    }

    for(int j = 0; j < num_threads_size; j++) {
        start_time = omp_get_wtime();
        omp_set_num_threads(num_threads[j]);
        #pragma omp parallel
        {
            #pragma omp for 
            for(int i = 0; i < n; i++) {
                for(int j = 0; j < n; j++) {
                    C[i][j] = get_value(A, i, B, j);
                }
            }
        end_time = omp_get_wtime();
        running_time_thread[j] = (end_time - start_time);
        }
    }
    for(int j = 0; j < num_threads_size; j++)
        printf("num_threads: %d Running time: %f\n", num_threads[j], running_time_thread[j]);
    return 0;

    for(int i = 0; i < n; i++) {
        for(int j = 0; j < n; j++) {
            printf("a: (%f), i%d, j %d <--> ", A[i][j], i, j);
        }
        printf("\n");
    }

    for(int i = 0; i < n; i++) {
        for(int j = 0; j < n; j++) {
            printf("b: (%f), i%d, j %d <--> ", B[i][j], i, j);
        }
        printf("\n");
    }
    for(int i = 0; i < n; i++) {
        for(int j = 0; j < n; j++) {
            printf("c: (%f), i%d, j %d <--> ", C[i][j], i, j);
        }
        printf("\n");
    }
}



double get_value(double A[][n], int ptr_A, double B[][n], int ptr_B) {
    double sum = 0;
    for(int i = 0; i < n; i++) {
        sum = sum + A[ptr_A][i] * B[i][ptr_B];
    }
    return sum;
}
