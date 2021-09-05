
#include<stdio.h>
#include<stdlib.h>
#include<time.h>

#define n 3

double get_value(double A[][n], int ptr_A, double B[][n], int ptr_B);

int main() {
    srand(time(0));
    double A[n][n], B[n][n], C[n][n];
    for(int i = 0; i < n; i++) {
        for(int j = 0; j < n; j++) {
            A[i][j] = (double)(rand() % 100000) / (double)1000;
            B[i][j] = (double)(rand() % 100000) / (double)1000;
        }
    }

    for(int i = 0; i < n; i++) {
        for(int j = 0; j < n; j++) {
            C[i][j] = A[i][j] + B[i][j];
        }
    }

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
