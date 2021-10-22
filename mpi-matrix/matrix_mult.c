// Program to Add two vectors

#include<mpi.h>
#include<stdio.h>
#include<stdlib.h>
#include<time.h>

#define MAX_SIZE 10
#define TRUE 1
#define FALSE 0

double get_value(double **A, int ptr_A, double **B, int ptr_B);
void PrintMatrix(double ** array, char * array_name);
int Allocate2DMemory(double ***array, int n, int m);
int Free2DMemory(double ***array);
void InitMatrix(double **array, _Bool is_empty);



int main(int argc, char** argv) {
    MPI_Init(NULL, NULL);

    int world_size, my_rank;
    double start, end;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    char processor_name[MPI_MAX_PROCESSOR_NAME];
    int name_len;
    MPI_Get_processor_name(processor_name, &name_len);
    
    double **matrix1;
    double **matrix2; 
    double **product_matrix;
    Allocate2DMemory(&matrix1, MAX_SIZE, MAX_SIZE);
    Allocate2DMemory(&matrix2, MAX_SIZE, MAX_SIZE);
    Allocate2DMemory(&product_matrix, MAX_SIZE, MAX_SIZE);
    InitMatrix(product_matrix, TRUE);

    MPI_Barrier(MPI_COMM_WORLD); /* IMPORTANT */
    start = MPI_Wtime();
    if(my_rank == 0) {
        srand(time(0));
        InitMatrix(matrix1, FALSE);
        InitMatrix(matrix2, FALSE);
        PrintMatrix(matrix1, "A");
        PrintMatrix(matrix2, "B");
        for(int i = 0; i < MAX_SIZE%world_size; i++) {
            for(int j = 0; j < MAX_SIZE; j++)
                product_matrix[MAX_SIZE - 1 - i][j] = get_value(matrix1, MAX_SIZE - 1 - i, matrix2, j);
        }
    }

    int len = MAX_SIZE / world_size;
    int * start_i_arr = (int *)malloc(sizeof(int)*world_size);
    for(int j = 0; j < world_size; j++) {
        start_i_arr[j] = j*len;
    }
    int start_i = 0;
    double **mult_mat;
    Allocate2DMemory(&mult_mat, len, MAX_SIZE);

    MPI_Bcast(&(matrix1[0][0]), MAX_SIZE*MAX_SIZE, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&(matrix2[0][0]), MAX_SIZE*MAX_SIZE, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Scatter(start_i_arr, 1, MPI_INT, &start_i, 1, MPI_INT, 0, MPI_COMM_WORLD);
    
    for(int i = 0; i < len; i++) {
        for(int j = 0; j < MAX_SIZE; j++)
            mult_mat[i][j] = get_value(matrix1, i + start_i, matrix2, j);
    }

    MPI_Gather(&(mult_mat[0][0]), len*MAX_SIZE, MPI_DOUBLE, &(product_matrix[0][0]), len*MAX_SIZE, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    Free2DMemory(&mult_mat);
    MPI_Barrier(MPI_COMM_WORLD); 
    end = MPI_Wtime();
    if(my_rank == 0) {
        PrintMatrix(product_matrix, "A * B");
        printf("Time taken: %f\n", end-start);
        double **test;
        Allocate2DMemory(&test, MAX_SIZE, MAX_SIZE);
        for(int i = 0; i < MAX_SIZE; i++) {
            for(int j = 0; j < MAX_SIZE; j++)
                test[i][j] = get_value(matrix1, i, matrix2, j);
        }
        PrintMatrix(test, "test");
    }
    Free2DMemory(&matrix1);
    Free2DMemory(&matrix2);
    Free2DMemory(&product_matrix);
    MPI_Finalize();
    return 0;
}

// Initializes the 2D matrix
void InitMatrix(double **array, _Bool is_empty) {
    for(int i = 0; i < MAX_SIZE; i++) {
        for(int j = 0; j < MAX_SIZE; j++) {
            if(is_empty)
                array[i][j] = -1;
            else    
                array[i][j] = (double)(rand() % 100) / (double)10;
        }
   } 
}

// allocates the memory for the pseudo 2D array
int Allocate2DMemory(double ***array, int n, int m) {
    double *p = (double *)malloc(n*m*sizeof(double));
    if (!p) return -1;

    (*array) = (double **)malloc(n*sizeof(double*));
    if (!(*array)) {
       free(p);
       return -1;
    }

    for (int i=0; i<n; i++) 
       (*array)[i] = &(p[i*m]);

    return 0;
}

// frees the memory of the pseudo 2D array
int Free2DMemory(double ***array) {
    free(&((*array)[0][0]));
    free(*array);
    return 0;
}

// prints the matrix
void PrintMatrix(double ** array, char * array_name) {
    printf("The Matrix %s:\n", array_name);
    for(int i = 0; i < MAX_SIZE; i++) {
        for(int j = 0; j < MAX_SIZE; j++) {
            printf("%f ", array[i][j]);
        }
        printf("\n");
    }
    printf("\n");
}

// returns the calculated value of dot product of row ptr_A of A and column ptr_B of B
double get_value(double **A, int ptr_A, double **B, int ptr_B) {
    double sum = 0;
    for(int i = 0; i < MAX_SIZE; i++) {
        sum = sum + A[ptr_A][i] * B[i][ptr_B];
    }
    return sum;
}