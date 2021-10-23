// Program to Add two matrices

#include<mpi.h>
#include<stdio.h>
#include<stdlib.h>
#include<time.h>

#define MAX_SIZE 100
#define TRUE 1
#define FALSE 0

void PrintMatrix(double ** array, char * array_name);
int Allocate2DMemory(double ***array, int n, int m);
int Free2DMemory(double ***array);
void InitMatrix(double **array, _Bool is_empty);
void TestSum(double **A, double **B, int len);


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
    double **final_sum_matrix;
    Allocate2DMemory(&matrix1, MAX_SIZE, MAX_SIZE);
    Allocate2DMemory(&matrix2, MAX_SIZE, MAX_SIZE);
    Allocate2DMemory(&final_sum_matrix, MAX_SIZE, MAX_SIZE);
    InitMatrix(final_sum_matrix, TRUE);

    MPI_Barrier(MPI_COMM_WORLD); /* IMPORTANT */
    start = MPI_Wtime();
    if(my_rank == 0) {
        srand(time(0));
        InitMatrix(matrix1, FALSE);
        InitMatrix(matrix2, FALSE);
        // PrintMatrix(matrix1, "A");  // uncomment to check the first randomly generated matrix
        // PrintMatrix(matrix2, "B");  // uncomment to check the second randomly generated matrix
        for(int i = 0; i < MAX_SIZE%world_size; i++) {
            for(int j = 0; j < MAX_SIZE; j++)
                final_sum_matrix[MAX_SIZE - 1 - i][j] = matrix1[MAX_SIZE - 1 - i][j] + matrix2[MAX_SIZE - 1 - i][j];
        }
    }

    int len = MAX_SIZE / world_size;
    double **scat_mat1;
    double **scat_mat2; 
    double **sum_mat;
    Allocate2DMemory(&scat_mat1, len, MAX_SIZE);
    Allocate2DMemory(&scat_mat2, len, MAX_SIZE);
    Allocate2DMemory(&sum_mat, len, MAX_SIZE);

    MPI_Scatter(&(matrix1[0][0]), len*MAX_SIZE, MPI_DOUBLE, &(scat_mat1[0][0]), len*MAX_SIZE, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Scatter(&(matrix2[0][0]), len*MAX_SIZE, MPI_DOUBLE, &(scat_mat2[0][0]), len*MAX_SIZE, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    for(int i = 0; i < len; i++) {
        for(int j = 0; j < MAX_SIZE; j++)
            sum_mat[i][j] = scat_mat1[i][j] + scat_mat2[i][j];
    }

    MPI_Gather(&(sum_mat[0][0]), len*MAX_SIZE, MPI_DOUBLE, &(final_sum_matrix[0][0]), len*MAX_SIZE, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    Free2DMemory(&scat_mat1);
    Free2DMemory(&scat_mat2);
    Free2DMemory(&sum_mat);
    MPI_Barrier(MPI_COMM_WORLD); 
    end = MPI_Wtime();
    if(my_rank == 0) {
        // PrintMatrix(final_sum_matrix, "A + B"); // uncomment to check the sum using MPI
        printf("Time taken: %f\n", end-start);
        // TestSum(matrix1, matrix2, MAX_SIZE); // uncomment to check the actual sum calculated 
    }
    Free2DMemory(&matrix1);
    Free2DMemory(&matrix2);
    Free2DMemory(&final_sum_matrix);
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
                array[i][j] = (double)(rand() % 100000) / (double)100;
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

// Calculates sum without using MPI for testing purpose
void TestSum(double **A, double **B, int len) {
    printf("This output is for test purposes only: \n");
    for(int i = 0; i < len; i++) {
        for(int j = 0; j < len; j++)
            printf("%f ", A[i][j] + B[i][j]);
        printf("\n");
    }
    printf("\n\n");
}