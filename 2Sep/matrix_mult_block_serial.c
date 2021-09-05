

#include<stdio.h>
#include<stdlib.h>
#include<time.h>
#define n 9

double get_value(double A[][n], int ptr_A, double B[][n], int ptr_B);

int main() {
    srand(time(0));
    double A[n][n], B[n][n], C[n][n], C_Old[n][n];
    for(int i = 0; i < n; i++) {
        for(int j = 0; j < n; j++) {
            A[i][j] = (double)(rand() % 100000) / (double)1000;
            B[i][j] = (double)(rand() % 100000) / (double)1000;
        }
    }
    int block_size = 3;
    for(int i = 0; i < n; i+=block_size) {
        for(int j = 0; j < n; j+=block_size) {
            // block level computation
            for(int block_i = i; block_i < i + block_size; block_i++) {
                for(int block_j = j; block_j < j + block_size; block_j++)
                    C[block_i][block_j] = get_value(A, block_i, B, block_j);
            }
        }
    }
    for(int i = 0; i < n; i++) {
        for(int j = 0; j < n; j++) {
            // block level computation
            C_Old[i][j] = get_value(A, i, B, j);
        }
    }

    for(int i = 0; i < n; i++) {
        for(int j = 0; j < n; j++) {
            printf("%f ", A[i][j]);
        }
        printf("\n");
    }

    for(int i = 0; i < n; i++) {
        for(int j = 0; j < n; j++) {
            printf("%f ", B[i][j]);
        }
        printf("\n");
    }
    for(int i = 0; i < n; i++) {
        for(int j = 0; j < n; j++) {
            printf("%f ", C[i][j]-C_Old[i][j]);
        }
        printf("\n");
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