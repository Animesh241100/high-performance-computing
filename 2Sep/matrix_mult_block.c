#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include<time.h>

#define n 456

double get_value(double A[][n], int ptr_A, double B[][n], int ptr_B);
void show_matrix(double A[][n]);

int main() {
    srand(time(0));
    double A[n][n], B[n][n], C[n][n];
    double start_time, end_time, running_time;
    int num_threads_size = 8;
    int num_threads [] = {1, 2, 3, 4, 5, 6, 8, 10};
    int block_sizes_size = 8;
    int block_sizes [] = {1, 2, 4, 8, 16, 20, 24, 32};
    float running_time_thread[num_threads_size][block_sizes_size];
    for(int i = 0; i < n; i++) {
        for(int j = 0; j < n; j++) {
            A[i][j] = (double)(rand() % 100000) / (double)1000;
            B[i][j] = (double)(rand() % 100000) / (double)1000;
        }
    }
 
    for(int b = 0; b < block_sizes_size; b++) {
        printf("--> Block size %d ---------\n", block_sizes[b]);
        for(int t = 0; t < num_threads_size; t++) {
            start_time = omp_get_wtime();
            printf("     |--> Num Threads %d --------- --", num_threads[t]);
            omp_set_num_threads(num_threads[t]);
            int block_size = block_sizes[b];
            for(int i = 0; i < n; i+=block_size) {
                for(int j = 0; j < n; j+=block_size) {
                    // block level computation
                    #pragma omp parallel for collapse(2)
                    for(int block_i = i; block_i < i + block_size; block_i++) {
                        for(int block_j = j; block_j < j + block_size; block_j++)
                            C[block_i][block_j] = get_value(A, block_i, B, block_j);
                    }
                }
            }
            end_time = omp_get_wtime();
            running_time_thread[t][b] = (end_time - start_time);
            printf("\b\b%f\n", running_time_thread[t][b]);
        }
    }

    return 0;
}


double get_value(double A[][n], int ptr_A, double B[][n], int ptr_B) {
    double sum = 0;
    for(int i = 0; i < n; i++) {
        sum = sum + A[ptr_A][i] * B[i][ptr_B];
    }
    return sum;
}

void show_matrix(double A[][n]) {
    for(int i = 0; i < n; i++) {
        for(int j = 0; j < n; j++) {
            printf("%f ", A[i][j]);
        }
        printf("\n");
    }
}

