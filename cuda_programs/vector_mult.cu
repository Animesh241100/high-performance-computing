// Program to mult two vectors using CUDA Programming

#include<stdio.h>
#include<stdlib.h>
#include<time.h>

#define NUM_BLOCKS 10
#define NUM_THREADS_PER_BLOCK 1
#define MAX_SIZE 10  

void InitVector(double * vec, int len);
void PrintVector(double * vec, int len, char * array_name);


__global__ void vector_mult(double *a, double *b, double *product) {
    product[blockIdx.x] = a[blockIdx.x] * b[blockIdx.x];
}


int main(void) {
    srand(time(0));
    int size = sizeof(double)*MAX_SIZE;
    double *a = (double*)malloc(size);
    double *b = (double*)malloc(size);
    double *product = (double*)malloc(size);
    InitVector(a, MAX_SIZE);
    InitVector(b, MAX_SIZE);
    InitVector(product, MAX_SIZE);
    
    double *dev_a, *dev_b, *dev_product;
    cudaMalloc((void **)&dev_a, size);
    cudaMalloc((void **)&dev_b, size);
    cudaMalloc((void **)&dev_product, size);

    for(int i = 0; i < MAX_SIZE; i++) {
        cudaMemcpy(dev_a, a, size, cudaMemcpyHostToDevice);
        cudaMemcpy(dev_b, b, size, cudaMemcpyHostToDevice);
    }

    vector_mult<<<NUM_BLOCKS,NUM_THREADS_PER_BLOCK>>>(dev_a, dev_b, dev_product);
    cudaError err = cudaMemcpy(product, dev_product, size, cudaMemcpyDeviceToHost);
	if(err!=cudaSuccess) {
		printf("CUDA error copying to Host: %s\n", cudaGetErrorString(err));
	}
    PrintVector(a, MAX_SIZE, "A: ");  
    PrintVector(b, MAX_SIZE, "B: ");  
    PrintVector(product, MAX_SIZE, "Final product of A and B: ");  
    // Cleanup
    free(a);
    free(b);
    free(product);
    cudaFree(dev_a);
    cudaFree(dev_b);
    cudaFree(dev_product);
    return 0;

}


void InitVector(double * vec, int len) {
    for(int i = 0; i < MAX_SIZE; i++){
        vec[i] = (double)(rand() % 100000) / (double)1000;
    }
}

void PrintVector(double * vec, int len, char * array_name) {
    printf("The Vector %s:\n", array_name);
    for(int i = 0; i < len; i++)
        printf("%f ", vec[i]);
    printf("\n\n");
}