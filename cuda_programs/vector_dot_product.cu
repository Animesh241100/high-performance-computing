// A CUDA Program to Find Dot product of two vectors

#include<stdio.h>
#include<stdlib.h>
#include<chrono>

#define MAX_SIZE 100000000
#define TRUE 1
#define FALSE 0
int GRID_SIZE;        // Total Number of blocks
int BLOCK_SIZE;       // Total Number of threads in one block

void InitVector(int * vec, int len);
void PrintVector(int * vec, int len, char * array_name);
void TestDotProduct(int *A, int *B, int len);

__global__ void vector_add(int *a, int *b, int *dot_product) {
    int idx = (blockIdx.x)*(blockDim.x) + threadIdx.x;
    int num_procs = (blockDim.x)*(gridDim.x);
    unsigned long long len = MAX_SIZE/num_procs;
    __shared__ int temp_sum[1000];      // size is only 'BLOCK_SIZE' amount
    unsigned long long min_i = len*idx;
    unsigned long long max_i = len*idx + len - 1;
    if(max_i < (unsigned long long)MAX_SIZE) {
        for(unsigned long long i = min_i; i <= max_i; i++)
            temp_sum[threadIdx.x] += a[i] * b[i];
    }
    if(idx == 0) {
        for(unsigned long long i = 0; i < (MAX_SIZE%num_procs); i++) 
            temp_sum[threadIdx.x] += a[MAX_SIZE-1-i] * b[MAX_SIZE-1-i];
    }
    __syncthreads();

    // dot_product calculation at the block level
    if(threadIdx.x == 0) {
        int sum_val = 0;
        for(int i = 0; i < blockDim.x; i++) {
            sum_val += temp_sum[i];
        }
        atomicAdd(dot_product, sum_val); // dot_product accumulation at the grid level
    }
}



int main(void) {
    printf("Enter the grid size and the block size respectively: ");
    scanf("%d %d", &GRID_SIZE, &BLOCK_SIZE);
    srand(time(0));
    int size = sizeof(int)*MAX_SIZE;

    // Initialise the vectors
    int *a = (int*)malloc(size);
    int *b = (int*)malloc(size);
    int *dot_product = (int*)malloc(sizeof(int));
    InitVector(a, MAX_SIZE);
    InitVector(b, MAX_SIZE);
    auto start = std::chrono::high_resolution_clock::now();
    int *dev_a; 
    int *dev_b; 
    int *dev_dot_product;
    
    // allocate memory on the device
    cudaMalloc((void **)&dev_a, size);
    cudaMalloc((void **)&dev_b, size);
    cudaMalloc((void **)&dev_dot_product, sizeof(int));

    // copy the data to the device
    cudaMemcpy(dev_a, a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_b, b, size, cudaMemcpyHostToDevice);

    vector_add<<<GRID_SIZE,BLOCK_SIZE>>>(dev_a, dev_b, dev_dot_product);
    cudaError err = cudaMemcpy(dot_product, dev_dot_product, sizeof(int), cudaMemcpyDeviceToHost);
	if(err != cudaSuccess) {
		printf("CUDA error copying to Host: %s\n", cudaGetErrorString(err));
	}

    // PrintVector(a, MAX_SIZE, "N numbers:");  // uncomment to see the N numbers
    printf("The dot_product of two vectors of size %d using CUDA is : %d\n", MAX_SIZE, *dot_product);  // uncomment to see the output using CUDA
    auto stop = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
    printf("Time Taken: %ld\n", duration.count());

    TestDotProduct(a, b, MAX_SIZE);   // uncomment to see the actual output without using CUDA

    // Cleanup
    free(a);
    free(b);
    free(dot_product);
    cudaFree(dev_a);
    cudaFree(dev_b);
    cudaFree(dev_dot_product);
    return 0;
}


// initialises a vector
void InitVector(int * vec, int len) {
    for(int i = 0; i < MAX_SIZE; i++){
        vec[i] = (int)(rand() % 10000) / (int)100;
    }
}

//prints a vector to the host's screen
void PrintVector(int * vec, int len, char * array_name) {
    printf("The Vector %s:\n", array_name);
    for(int i = 0; i < len; i++)
        printf("%d ", vec[i]);
    printf("\n\n");
}


// Calculates dot_product without using CUDA for testing purpose
void TestDotProduct(int *A, int *B, int len) {
    int temp_sum = 0;
    for(int i = 0; i < len; i++) 
        temp_sum += (A[i] * B[i]);
    printf("This actual output for testing purposes is (without using CUDA) : %d\n", temp_sum);
}