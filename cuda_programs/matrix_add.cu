// CUDA Program to add two matrices

#include<stdio.h>
#include<stdlib.h>
#include<chrono>

#define MAX_SIZE 5000
#define TRUE 1
#define FALSE 0
int GRID_SIZE;        // Total Number of blocks
int BLOCK_SIZE;       // Total Number of threads in one block

void InitMatrix(double *array, int is_empty);
int Allocate2DMemory(double ***array, int n, int m);
int Allocate2DMemoryDevice(double ***array, int n, int m);
int Free2DMemory(double ***array);
int Free2DMemoryDevice(double ***array);
void PrintMatrix(double * array, char * array_name);
void TestSum(double *A, double *B, int len);



__global__ void matrix_add(double *a, double *b, double *sum, int *dev_block_size, int *dev_grid_size) {
    long long idx = (blockIdx.x)*(*dev_block_size) + threadIdx.x;
    int num_procs = (*dev_block_size)*(*dev_grid_size);
    int len = MAX_SIZE/num_procs;
    if((long long)(len*idx + len - 1) < (long long)MAX_SIZE) {
        for(long long i = 0; i < len; i++)
            for(long long j = 0; j < MAX_SIZE; j++)
                sum[(i + len*idx)*MAX_SIZE + j] = a[(i + len*idx)*MAX_SIZE + j] + b[(i + len*idx)*MAX_SIZE + j];
    }
    if(blockIdx.x * threadIdx.x == 0) {
        for(long long i = 0; i < (MAX_SIZE%num_procs); i++) {
            for(long long j = 0; j < MAX_SIZE; j++)
                sum[(MAX_SIZE-1-i)*MAX_SIZE + j] = a[(MAX_SIZE-1-i)*MAX_SIZE + j] + b[(MAX_SIZE-1-i)*MAX_SIZE + j];
        }
    }
}



int main(void) {
    printf("Enter the grid size and the block size respectively:\n");
    scanf("%d %d", &GRID_SIZE, &BLOCK_SIZE);
    int size = sizeof(double) * MAX_SIZE * MAX_SIZE;
    
    // Initialising the matrices
    double *matrix1 = (double *)malloc(size);
    double *matrix2 = (double *)malloc(size);
    double *final_sum_matrix = (double *)malloc(size);
    srand(time(0));
    InitMatrix(matrix1, FALSE);
    InitMatrix(matrix2, FALSE);
    
    auto start = std::chrono::high_resolution_clock::now();
    // copying the data to the device
    double *dev_m1;
    double *dev_m2;
    double *dev_sum;
    int *dev_block_size, *dev_grid_size;
    cudaMalloc((void **)&dev_m1, size);
    cudaMalloc((void **)&dev_m2, size);
    cudaMalloc((void **)&dev_block_size, sizeof(int));
    cudaMalloc((void **)&dev_grid_size, sizeof(int));
    cudaMalloc((void **)&dev_sum, size);
    cudaMemcpy(dev_m1, matrix1, size, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_m2, matrix2, size, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_block_size, &BLOCK_SIZE, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_grid_size, &GRID_SIZE, sizeof(int), cudaMemcpyHostToDevice);

    matrix_add<<<GRID_SIZE,BLOCK_SIZE>>>(dev_m1, dev_m2, dev_sum, dev_block_size, dev_grid_size);
    cudaError err = cudaMemcpy(final_sum_matrix, dev_sum, size, cudaMemcpyDeviceToHost);
	if(err != cudaSuccess) {
		printf("CUDA error copying to Host: %s\n", cudaGetErrorString(err));
	}

    // PrintMatrix(matrix1, "A: ");  // uncomment to display the A matrix
    // PrintMatrix(matrix2, "B: ");  // uncomment to display the B matrix
    // PrintMatrix(final_sum_matrix, "Sum: "); // uncomment to display the final sum matrix
    auto stop = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
    printf("Time Taken: %ld\n", duration.count());
    // TestSum(matrix1, matrix2, MAX_SIZE); // uncomment to check the actual sum without using CUDA

    // Cleanup
    free(matrix1);
    free(matrix2);
    free(final_sum_matrix);
    cudaFree(dev_m1);
    cudaFree(dev_m2);
    cudaFree(dev_sum);
    return 0;

}


// Initializes the 2D matrix
void InitMatrix(double *matrix, int is_empty) {
    for(int i = 0; i < MAX_SIZE; i++) {
        for(int j = 0; j < MAX_SIZE; j++) {
            if(is_empty)
                matrix[i*MAX_SIZE + j] = -1;
            else    
                matrix[i*MAX_SIZE + j] = (double)(rand() % 100000) / (double)100;
        }
   } 
}

// prints the matrix 
void PrintMatrix(double * matrix, char * matrix_name) {
    printf("The Matrix %s:\n", matrix_name);
    for(int i = 0; i < MAX_SIZE; i++) {
        for(int j = 0; j < MAX_SIZE; j++) {
            printf("%f ", matrix[i*MAX_SIZE + j]);
        }
        printf("\n");
    }
    printf("\n");
}

// Calculates sum without using CUDA for testing purpose
void TestSum(double *A, double *B, int len) {
    printf("This output is without using CUDA : \n");
    for(int i = 0; i < len; i++) {
        for(int j = 0; j < len; j++)
            printf("%f ", A[i*len + j] + B[i*len + j]);
        printf("\n");
    }
    printf("\n\n");
}