// CUDA Program to multiply two matrices

#include<stdio.h>
#include<stdlib.h>
#include<chrono>

#define MAX_SIZE 500
#define TRUE 1
#define FALSE 0
int GRID_SIZE;        // Total Number of blocks
int BLOCK_SIZE;       // Total Number of threads in one block

void InitMatrix(double *array, int is_empty);
void PrintMatrix(double * array, char * array_name);
void TestProduct(double *A, double *B, int len);
double get_value(double *A, int ptr_A, double *B, int ptr_B);


__device__ double get_value2(double *A, int ptr_A, double *B, int ptr_B) {
    double product = 0;
    for(int i = 0; i < MAX_SIZE; i++) {
        product = product + A[ptr_A*MAX_SIZE + i] * B[i*MAX_SIZE + ptr_B];
    }
    return product;
}


__global__ void matrix_mult(double *a, double *b, double *product, int *dev_block_size, int *dev_grid_size) {
    long long idx = (blockIdx.x)*(*dev_block_size) + threadIdx.x;
    int num_procs = (*dev_block_size)*(*dev_grid_size);
    int len = MAX_SIZE/num_procs;
    if((long long)(len*idx + len - 1) < (long long)MAX_SIZE) {
        for(long long i = 0; i < len; i++)
            for(long long j = 0; j < MAX_SIZE; j++)
                product[(i + len*idx)*MAX_SIZE + j] = get_value2(a, i + len*idx, b, j);
    }
    if(blockIdx.x * threadIdx.x == 0) {
        for(long long i = 0; i < (MAX_SIZE%num_procs); i++) {
            for(long long j = 0; j < MAX_SIZE; j++)
                product[(MAX_SIZE-1-i)*MAX_SIZE + j] = get_value2(a, MAX_SIZE-1-i, b, j);
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
    double *final_product_matrix = (double *)malloc(size);
    srand(time(0));
    InitMatrix(matrix1, FALSE);
    InitMatrix(matrix2, FALSE);
    
    auto start = std::chrono::high_resolution_clock::now();
    
    // copying the data to the device
    double *dev_m1;
    double *dev_m2;
    double *dev_product;
    int *dev_block_size, *dev_grid_size;
    cudaMalloc((void **)&dev_m1, size);
    cudaMalloc((void **)&dev_m2, size);
    cudaMalloc((void **)&dev_block_size, sizeof(int));
    cudaMalloc((void **)&dev_grid_size, sizeof(int));
    cudaMalloc((void **)&dev_product, size);
    cudaMemcpy(dev_m1, matrix1, size, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_m2, matrix2, size, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_block_size, &BLOCK_SIZE, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_grid_size, &GRID_SIZE, sizeof(int), cudaMemcpyHostToDevice);

    matrix_mult<<<GRID_SIZE,BLOCK_SIZE>>>(dev_m1, dev_m2, dev_product, dev_block_size, dev_grid_size);
    cudaError err = cudaMemcpy(final_product_matrix, dev_product, size, cudaMemcpyDeviceToHost);
	if(err != cudaSuccess) {
		printf("CUDA error copying to Host: %s\n", cudaGetErrorString(err));
	}

    // PrintMatrix(matrix1, "A: ");  // uncomment to display the A matrix
    // PrintMatrix(matrix2, "B: ");  // uncomment to display the B matrix
    // PrintMatrix(final_product_matrix, "product: "); // uncomment to display the final product matrix
    auto stop = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
    printf("Time Taken: %ld\n", duration.count());
    // TestProduct(matrix1, matrix2, MAX_SIZE); // uncomment to check the actual product without using CUDA

    // Cleanup
    free(matrix1);
    free(matrix2);
    free(final_product_matrix);
    cudaFree(dev_m1);
    cudaFree(dev_m2);
    cudaFree(dev_product);
    return 0;

}

// returns the calculated value of dot product of row ptr_A of A and column ptr_B of B
double get_value(double *A, int ptr_A, double *B, int ptr_B) {
    double product = 0;
    for(int i = 0; i < MAX_SIZE; i++) {
        product = product + A[ptr_A*MAX_SIZE + i] * B[i*MAX_SIZE + ptr_B];
    }
    return product;
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

// Calculates the Product without using CUDA for testing purpose
void TestProduct(double *A, double *B, int len) {
    printf("This output is without using CUDA : \n");
    for(int i = 0; i < len; i++) {
        for(int j = 0; j < len; j++)
            printf("%f ",  get_value(A, i, B, j));
        printf("\n");
    }
    printf("\n\n");
}