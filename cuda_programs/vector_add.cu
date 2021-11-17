// Program to Add two vectors using CUDA Programming

#include<stdio.h>
#include<stdlib.h>
#include<time.h>

#define NUM_BLOCKS 10
#define NUM_THREADS_PER_BLOCK 1
#define MAX_SIZE 10  

void InitVector(double * vec, int len);
void PrintVector(double * vec, int len, char * array_name);


__global__ void vector_add(double *a, double *b, double *sum) {
    sum[blockIdx.x] = a[blockIdx.x] + b[blockIdx.x];
}


int main(void) {
    int size = sizeof(double)*MAX_SIZE;
    double *a = (double*)malloc(size);
    double *b = (double*)malloc(size);
    double *sum = (double*)malloc(size);
    InitVector(a, MAX_SIZE);
    InitVector(b, MAX_SIZE);
    InitVector(sum, MAX_SIZE);
    
    double *dev_a, *dev_b, *dev_sum;
    cudaMalloc((void **)&dev_a, size);
    cudaMalloc((void **)&dev_b, size);
    cudaMalloc((void **)&dev_sum, size);

    for(int i = 0; i < MAX_SIZE; i++) {
        cudaMemcpy(dev_a, a, size, cudaMemcpyHostToDevice);
        cudaMemcpy(dev_b, b, size, cudaMemcpyHostToDevice);
    }

    vector_add<<<NUM_BLOCKS,NUM_THREADS_PER_BLOCK>>>(dev_a, dev_b, dev_sum);
    cudaError err = cudaMemcpy(sum, dev_sum, size, cudaMemcpyDeviceToHost);
	if(err!=cudaSuccess) {
		printf("CUDA error copying to Host: %s\n", cudaGetErrorString(err));
	}
    PrintVector(a, MAX_SIZE, "A: ");  
    PrintVector(b, MAX_SIZE, "B: ");  
    PrintVector(sum, MAX_SIZE, "Final Sum(A+B): ");  
    // Cleanup
    free(a);
    free(b);
    free(sum);
    cudaFree(dev_a);
    cudaFree(dev_b);
    cudaFree(dev_sum);
    return 0;

}


void InitVector(double * vec, int len) {
    for(int i = 0; i < MAX_SIZE; i++){
        vec[i] = (double)(rand() % 100) / (double)1;
    }
}

void PrintVector(double * vec, int len, char * array_name) {
    printf("The Vector %s:\n", array_name);
    for(int i = 0; i < len; i++)
        printf("%f ", vec[i]);
    printf("\n\n");
}