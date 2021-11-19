// A CUDA Program to Add two vectors

#include<stdio.h>
#include<stdlib.h>
#include<sys/time.h>    
#include<chrono>

#define MAX_SIZE 10000000
#define TRUE 1
#define FALSE 0
int GRID_SIZE;        // Total Number of blocks
int BLOCK_SIZE;       // Total Number of threads in one block

void InitVector(double * vec, int len);
void PrintVector(double * vec, int len, char * array_name);
void TestSum(double *A, double *B, int len);

__global__ void vector_add(double *a, double *b, double *sum, int *dev_block_size, int *dev_grid_size) {
    int idx = (blockIdx.x)*(*dev_block_size) + threadIdx.x;
    int num_procs = (*dev_block_size)*(*dev_grid_size);
    int len = MAX_SIZE/num_procs;
    // printf("idx: %d, doing: %d to %d\n", idx, len*idx, len*idx+len-1);
    if((long long)(len*idx + len - 1) < (long long)MAX_SIZE*MAX_SIZE) {
        for(long long i = 0; i < len; i++)
            sum[i + len*idx] = a[i + len*idx] + b[i + len*idx];
    }
    if(blockIdx.x * threadIdx.x == 0) {
        for(long long i = 0; i < (MAX_SIZE%num_procs); i++) {
            sum[MAX_SIZE-1-i] = a[MAX_SIZE-1-i] + b[MAX_SIZE-1-i];
        }
    }
}



int main(void) {
    printf("Enter the grid size and the block size respectively:\n");
    scanf("%d %d", &GRID_SIZE, &BLOCK_SIZE);
    srand(time(0));
    int size = sizeof(double)*MAX_SIZE;

    // Initialise the vectors
    double *a = (double*)malloc(size);
    double *b = (double*)malloc(size);
    double *sum = (double*)malloc(size);
    InitVector(a, MAX_SIZE);
    InitVector(b, MAX_SIZE);
    InitVector(sum, MAX_SIZE);
    auto start = std::chrono::high_resolution_clock::now();
    double *dev_a, *dev_b, *dev_sum;
    
    // allocate memory on the device
    int *dev_block_size, *dev_grid_size;
    cudaMalloc((void **)&dev_a, size);
    cudaMalloc((void **)&dev_b, size);
    cudaMalloc((void **)&dev_sum, size);
    cudaMalloc((void **)&dev_block_size, sizeof(int));
    cudaMalloc((void **)&dev_grid_size, sizeof(int));

    // copy the data to the device
    cudaMemcpy(dev_a, a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_b, b, size, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_block_size, &BLOCK_SIZE, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_grid_size, &GRID_SIZE, sizeof(int), cudaMemcpyHostToDevice);

    vector_add<<<GRID_SIZE,BLOCK_SIZE>>>(dev_a, dev_b, dev_sum, dev_block_size, dev_grid_size);
    cudaError err = cudaMemcpy(sum, dev_sum, size, cudaMemcpyDeviceToHost);
	if(err != cudaSuccess) {
		printf("CUDA error copying to Host: %s\n", cudaGetErrorString(err));
	}

    // PrintVector(a, MAX_SIZE, "A: ");  // uncomment to see the A vector
    // PrintVector(b, MAX_SIZE, "B: ");  // uncomment to see the B vector  
    // PrintVector(sum, MAX_SIZE, "Final Sum(A+B): ");  // uncomment to see the output using CUDA
    auto stop = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
    printf("Time Taken: %ld\n", duration.count());

    // TestSum(a, b, MAX_SIZE);   // uncomment to see the actual output without using CUDA

    // Cleanup
    free(a);
    free(b);
    free(sum);
    cudaFree(dev_a);
    cudaFree(dev_b);
    cudaFree(dev_sum);
    return 0;

}


// initialises a vector
void InitVector(double * vec, int len) {
    for(int i = 0; i < MAX_SIZE; i++){
        vec[i] = (double)(rand() % 100000) / (double)100;
    }
}

//prints a vector to the host's screen
void PrintVector(double * vec, int len, char * array_name) {
    printf("The Vector %s:\n", array_name);
    for(int i = 0; i < len; i++)
        printf("%f ", vec[i]);
    printf("\n\n");
}


// Calculates sum without using CUDA for testing purpose
void TestSum(double *A, double *B, int len) {
    printf("This output is without using CUDA : \n");
    for(int i = 0; i < len; i++) {
        printf("%f ", A[i] + B[i]);
    }
    printf("\n\n");
}