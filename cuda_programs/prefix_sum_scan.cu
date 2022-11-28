/*
 Implementation of Parallelized Scan(Prefix Sum) Operation for Arbitrarily Large Arrays with Arbitrary
 Block and Grid sizes in NVIDIA CUDA Programming.

 Written By: Animesh Kumar (CED18I065)

 See the Comment Section after the program for machine related details.
*/

#include <stdlib.h>
#include <stdio.h>
#include <time.h>

using namespace std;

// Exclusive Scan Algorithms
long sequential_scan(int* output, int* input, int length);  // Sequential
float scan(int *output, int *input, int length, int NUM_BLOCKS, int NUM_THREADS);  // Parallelized Version on NVIDIA GPUs

// GPU Kernel Functions
__global__ void GPU_kernel1(int* out, int* block_sums, int* in, int length, int EPT);
__global__ void GPU_kernel2(int *block_sums_scanned, int *block_sums, int length);
__global__ void GPU_kernel3(int* out, int* block_sums_scanned, int length, int EPT);

// utilities
bool verify_result(int* output, int* input, int length);
void printResult(const char* prefix, int result, long nanoseconds, int N);
void printResult(const char* prefix, int result, float milliseconds, int N);
void InitVector(int* in, int N);
long get_nanos();


void test(int N, int NUM_BLOCKS, int NUM_THREADS) {
	srand(time(0));
	int *in = new int[N];
	InitVector(in, N);

	printf("Count: %i , NUM_BLOCKS: %d, NUM_THREADS: %d\n", N, NUM_BLOCKS, NUM_THREADS);

	// sequential scan on CPU
	int *outSequential = new int[N]();
	long time_host = sequential_scan(outSequential, in, N);
	printResult("host    ", outSequential[N - 1], time_host, N);

	// parallel scan with GPU with Bank Conflict Optimizations
	int *outParallelGPU = new int[N]();
	float time_gpu = scan(outParallelGPU, in, N, NUM_BLOCKS, NUM_THREADS);
	printResult("gpu     ", outParallelGPU[N - 1], time_gpu, N);

	printf("\n");

	delete[] in;
	delete[] outSequential;
	delete[] outParallelGPU;
}

int main()
{
	int InputSizes[] = {
		(int)pow(2, 25),  // 20M (~30M)
		(int)pow(2, 24),  // 10M
		(int)pow(2, 23),  // 5M (~8M)
		(int)pow(2, 21),  // 2M
		(int)pow(2, 20),  // 1M
		(int)pow(2, 19),  // 500k
		(int)pow(2, 18),  // 200k,
		(int)pow(2, 17),  // 100k,
		(int)pow(2, 16),  // 50k,
		(int)pow(2, 14),  // 10k,
		(int)pow(2,10)    // 1k
	};
	int numElements = sizeof(InputSizes) / sizeof(InputSizes[0]);

	// for (int i = numElements - 1; i >= 0; i--) {
	// 	test(InputSizes[i], 16, 512);
	// }

	int blocks [] = {1, 2, 4, 8, 16, 32, 64, 128, 256, 512};
	int threads [] = {2, 4, 8, 16, 32, 64, 128, 256, 512, 1024};

	for(int i = 0; i < 9; i++)
		test(InputSizes[0], 16, threads[i]);

	for(int i = 0; i < 9; i++)
		test(InputSizes[0], 64, threads[i]);
	return 0;
}

void InitVector(int* in, int N) {
	for (int i = 0; i < N; i++) {
		in[i] = rand() % 100;
	}
}

// exclusive scan
long sequential_scan(int* output, int* input, int length) {
	long start_time = get_nanos();

	output[0] = 0; 
	for (int j = 1; j < length; ++j)
	{
		output[j] = input[j - 1] + output[j - 1];
	}

	long end_time = get_nanos();
	return end_time - start_time;
}

// parallelized Mark Harris Algorithm
float scan(int *output, int *input, int length, int NUM_BLOCKS, int NUM_THREADS) {
	int *d_out, *d_in;
	const int arraySize = length * sizeof(int);

  NUM_BLOCKS = min(length / NUM_THREADS, NUM_BLOCKS);
  int ELEMENTS_PER_THREAD = length / (NUM_THREADS * NUM_BLOCKS);

  int* block_sums = new int[NUM_BLOCKS];
  for(int i = 0; i < NUM_BLOCKS; i++)
    block_sums[i] = 0;

  int* d_block_sums;
	int* d_block_sums_scanned;

	cudaMalloc((void **)&d_out, arraySize);
	cudaMalloc((void **)&d_in, arraySize);
	cudaMalloc((void **)&d_block_sums, NUM_BLOCKS * sizeof(int));
	cudaMalloc((void **)&d_block_sums_scanned, NUM_BLOCKS * sizeof(int));
	cudaMemcpy(d_out, output, arraySize, cudaMemcpyHostToDevice);
	cudaMemcpy(d_in, input, arraySize, cudaMemcpyHostToDevice);
	cudaMemcpy(d_block_sums, block_sums, NUM_BLOCKS * sizeof(int), cudaMemcpyHostToDevice);

	// start timer
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start);

	GPU_kernel1<<<NUM_BLOCKS, NUM_THREADS, 4*NUM_THREADS*sizeof(int)>>> (d_out, d_block_sums, d_in, length, ELEMENTS_PER_THREAD);
	
	GPU_kernel2<<<1, NUM_BLOCKS/2, 2*NUM_BLOCKS*sizeof(int)>>> (d_block_sums_scanned, d_block_sums, NUM_BLOCKS) ;
	
	GPU_kernel3<<<NUM_BLOCKS, NUM_THREADS>>> (d_out, d_block_sums_scanned, length, ELEMENTS_PER_THREAD);

	// end timer
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	float elapsedTime = 0;
	cudaEventElapsedTime(&elapsedTime, start, stop);

	cudaMemcpy(output, d_out, arraySize, cudaMemcpyDeviceToHost);
	cudaMemcpy(block_sums, d_block_sums, NUM_BLOCKS * sizeof(int), cudaMemcpyDeviceToHost);
  
	if(verify_result(output, input, length))
		printf("TEST PASSED\n");
	else
		printf("TEST FAILED\n");

  // printf("input: ");
  // for(int i = 0; i < length; i++)
  //   printf("%d ", input[i]);
  // printf("\n");

  // printf("output: ");
  // for(int i = 0; i < length; i++)
  //   printf("%d ", output[i]);
  // printf("\n");

	// printf("block_sums: ");
	// for(int i = 0; i < NUM_BLOCKS; i++)
	// 	printf("%d ", block_sums[i]);
	// printf("\n");


	cudaFree(d_out);
	cudaFree(d_in);
	cudaFree(d_block_sums);
	cudaEventDestroy(start);
	cudaEventDestroy(stop);

	return elapsedTime;
}

// Compute Thread level prefix sums
__global__ void GPU_kernel1(int* out, int* block_sums, int* in, int length, int EPT) {
  extern __shared__ int Shared_Chunk[]; // Chunk of Shared Memory
  int B = blockIdx.x; int T = threadIdx.x;
  int NUM_THREADS = blockDim.x;
  int I = B*NUM_THREADS + T;
  
  // Actual Shared Data
  int* thread_partial_sums = Shared_Chunk;             // Length: NUM_THREADS
  int* temp = (int*)&thread_partial_sums[NUM_THREADS]; // Length: 2*NUM_THREADS
  int* scanned_partials = (int*)&temp[2*NUM_THREADS];  // Length: NUM_THREADS

  // Do serial computation per thread in this block
  if(I*EPT < length) {
    int i = I*EPT;
    out[i++] = 0; // exclusive scan
    for(; i < (I+1)*EPT; i++)
      out[i] = in[i-1] + out[i-1];
    thread_partial_sums[T] = in[i-1] + out[i-1];
  } else
    thread_partial_sums[T] = 0;

  __syncthreads();  // Threads sync after calculating sequential partial sums
	
	if (T < NUM_THREADS / 2) {
		temp[2*T] = thread_partial_sums[2*T]; // load input into shared memory
		temp[2*T + 1] = thread_partial_sums[2*T + 1];
	}
	else {
		temp[2*T] = 0;
		temp[2*T + 1] = 0;
	}

	int offset = 1;
	for (int d = NUM_THREADS >> 1; d > 0; d >>= 1) // build sum in place up the tree
	{
		__syncthreads();
		if (T < d)
		{
			int ai = offset * (2*T + 1) - 1;
			int bi = offset * (2*T + 2) - 1;
			temp[bi] += temp[ai];
		}
		offset *= 2;
	}

	if (T == 0) { 
		block_sums[B] = temp[NUM_THREADS - 1];
    temp[NUM_THREADS - 1] = 0; // clear the last element 
  } 

	for (int d = 1; d < NUM_THREADS; d *= 2) // traverse down tree & build scan
	{
		offset >>= 1;
		__syncthreads();
		if (T < d)
		{
			int ai = offset * (2*T + 1) - 1;
			int bi = offset * (2*T + 2) - 1;
			int t = temp[ai];
			temp[ai] = temp[bi];
			temp[bi] += t;
		}
	}
	__syncthreads();

	if(T < NUM_THREADS / 2) {
		scanned_partials[2*T] = temp[2*T]; // write results to device memory
		scanned_partials[2*T + 1] = temp[2*T + 1];
	}

  __syncthreads();

  if(I*EPT < length) {
    for(int i = I*EPT; i < (I+1)*EPT; i++)
      out[i] += scanned_partials[T];
  }
}

// Compute Block level prefix sums 
__global__ void GPU_kernel2(int *block_sums_scanned, int *block_sums, int length) {
	extern __shared__ int temp[];// allocated on invocation
	int threadID = threadIdx.x;

	if (threadID < length) {
		temp[2 * threadID] = block_sums[2 * threadID]; // load input into shared memory
		temp[2 * threadID + 1] = block_sums[2 * threadID + 1];
	}
	else {
		temp[2 * threadID] = 0;
		temp[2 * threadID + 1] = 0;
	}


	int offset = 1;
	for (int d = length >> 1; d > 0; d >>= 1) // build sum in place up the tree
	{
		__syncthreads();
		if (threadID < d)
		{
			int ai = offset * (2 * threadID + 1) - 1;
			int bi = offset * (2 * threadID + 2) - 1;
			temp[bi] += temp[ai];
		}
		offset *= 2;
	}

	if (threadID == 0) { temp[length - 1] = 0; } // clear the last element

	for (int d = 1; d < length; d *= 2) // traverse down tree & build scan
	{
		offset >>= 1;
		__syncthreads();
		if (threadID < d)
		{
			int ai = offset * (2 * threadID + 1) - 1;
			int bi = offset * (2 * threadID + 2) - 1;
			int t = temp[ai];
			temp[ai] = temp[bi];
			temp[bi] += t;
		}
	}
	__syncthreads();

	if (threadID < length) {
		block_sums_scanned[2 * threadID] = temp[2 * threadID]; // write results to device memory
		block_sums_scanned[2 * threadID + 1] = temp[2 * threadID + 1];
	}
}

// Add Block level results to the out[] vector to get final scan output
__global__ void GPU_kernel3(int* out, int* block_sums_scanned, int length, int EPT) {
  int B = blockIdx.x;  int T = threadIdx.x;
  int NUM_THREADS = blockDim.x;
  int I = B*NUM_THREADS + T;
  if(I*EPT < length) {
    for(int i = I*EPT; i < (I+1)*EPT; i++)
      out[i] += block_sums_scanned[B];
  }
}


// checks the result of parallel code with the sequential output
bool verify_result(int* output, int* input, int length) {
	int* correct = new int[length];
	for (int j = 0; j < length; ++j) {
		correct[j] = j ? input[j - 1] + correct[j - 1] : 0;
		if(correct[j] != output[j]) {
			printf("Failed at j: %d : correct=%d, output=%d\n", j, correct[j], output[j]);
			return false;
		}
	}
	return true;
}


void printResult(const char* prefix, int result, long nanoseconds, int N) {
	printf("  ");
	printf("%s ", prefix);
	printf(" : output: %i in %7.6lf ms, [%9.3lf M/s, %7.3lf GB/s] \n", result, (double)nanoseconds / 1e6, (double)N * 1e3 / nanoseconds, (double)sizeof(int) * N / nanoseconds);
}


void printResult(const char* prefix, int result, float milliseconds, int N) {
	printf("  ");
	printf("%s ", prefix);
	printf(" : output: %i in %f ms, [%9.3lf M/s, %7.3lf GB/s] \n", result, milliseconds, (double)N / (1e3 * milliseconds), (double)sizeof(int) * N / (1e6 * milliseconds));
	// printf(" : %i in %f ms \n", result, milliseconds);
}


// Get the current time in nanoseconds
long get_nanos() {
	struct timespec ts;
	timespec_get(&ts, TIME_UTC);
	return (long)ts.tv_sec * 1000000000L + ts.tv_nsec;
}


/*

Sample Output:-

	Count: 33554432 , NUM_BLOCKS: 16, NUM_THREADS: 2
		host      : output: 1661174866 in 143.557989 ms, [  233.734 M/s,   0.935 GB/s] 
	TEST PASSED
		gpu       : output: 1661174866 in 229.438431 ms, [  146.246 M/s,   0.585 GB/s] 

	Count: 33554432 , NUM_BLOCKS: 16, NUM_THREADS: 4
		host      : output: 1661082478 in 132.591711 ms, [  253.066 M/s,   1.012 GB/s] 
	TEST PASSED
		gpu       : output: 1661082478 in 119.429123 ms, [  280.957 M/s,   1.124 GB/s] 

	Count: 33554432 , NUM_BLOCKS: 16, NUM_THREADS: 8
		host      : output: 1660971753 in 137.440945 ms, [  244.137 M/s,   0.977 GB/s] 
	TEST PASSED
		gpu       : output: 1660971753 in 64.964607 ms, [  516.503 M/s,   2.066 GB/s] 

	Count: 33554432 , NUM_BLOCKS: 16, NUM_THREADS: 16
		host      : output: 1660961505 in 140.191455 ms, [  239.347 M/s,   0.957 GB/s] 
	TEST PASSED
		gpu       : output: 1660961505 in 39.067650 ms, [  858.880 M/s,   3.436 GB/s] 

	Count: 33554432 , NUM_BLOCKS: 16, NUM_THREADS: 32
		host      : output: 1661077135 in 136.800484 ms, [  245.280 M/s,   0.981 GB/s] 
	TEST PASSED
		gpu       : output: 1661077135 in 26.560513 ms, [ 1263.320 M/s,   5.053 GB/s] 

	Count: 33554432 , NUM_BLOCKS: 16, NUM_THREADS: 64
		host      : output: 1661233049 in 120.636617 ms, [  278.145 M/s,   1.113 GB/s] 
	TEST PASSED
		gpu       : output: 1661233049 in 15.775744 ms, [ 2126.963 M/s,   8.508 GB/s] 

	Count: 33554432 , NUM_BLOCKS: 16, NUM_THREADS: 128
		host      : output: 1660906707 in 135.455001 ms, [  247.716 M/s,   0.991 GB/s] 
	TEST PASSED
		gpu       : output: 1660906707 in 13.519872 ms, [ 2481.860 M/s,   9.927 GB/s] 

	Count: 33554432 , NUM_BLOCKS: 16, NUM_THREADS: 256
		host      : output: 1661147273 in 130.051074 ms, [  258.010 M/s,   1.032 GB/s] 
	TEST PASSED
		gpu       : output: 1661147273 in 13.276160 ms, [ 2527.420 M/s,  10.110 GB/s] 

	Count: 33554432 , NUM_BLOCKS: 16, NUM_THREADS: 512
		host      : output: 1660927143 in 144.171149 ms, [  232.740 M/s,   0.931 GB/s] 
	TEST PASSED
		gpu       : output: 1660927143 in 12.809216 ms, [ 2619.554 M/s,  10.478 GB/s] 

	Count: 33554432 , NUM_BLOCKS: 64, NUM_THREADS: 2
		host      : output: 1660940893 in 129.892868 ms, [  258.324 M/s,   1.033 GB/s] 
	TEST PASSED
		gpu       : output: 1660940893 in 54.326271 ms, [  617.647 M/s,   2.471 GB/s] 

	Count: 33554432 , NUM_BLOCKS: 64, NUM_THREADS: 4
		host      : output: 1660670255 in 140.314965 ms, [  239.137 M/s,   0.957 GB/s] 
	TEST PASSED
		gpu       : output: 1660670255 in 30.603264 ms, [ 1096.433 M/s,   4.386 GB/s] 

	Count: 33554432 , NUM_BLOCKS: 64, NUM_THREADS: 8
		host      : output: 1661218370 in 123.032470 ms, [  272.728 M/s,   1.091 GB/s] 
	TEST PASSED
		gpu       : output: 1661218370 in 17.157120 ms, [ 1955.715 M/s,   7.823 GB/s] 

	Count: 33554432 , NUM_BLOCKS: 64, NUM_THREADS: 16
		host      : output: 1660673153 in 147.213374 ms, [  227.931 M/s,   0.912 GB/s] 
	TEST PASSED
		gpu       : output: 1660673153 in 10.637312 ms, [ 3154.409 M/s,  12.618 GB/s] 

	Count: 33554432 , NUM_BLOCKS: 64, NUM_THREADS: 32
		host      : output: 1661005815 in 110.236050 ms, [  304.387 M/s,   1.218 GB/s] 
	TEST PASSED
		gpu       : output: 1661005815 in 7.384064 ms, [ 4544.169 M/s,  18.177 GB/s] 

	Count: 33554432 , NUM_BLOCKS: 64, NUM_THREADS: 64
		host      : output: 1660829028 in 135.588942 ms, [  247.472 M/s,   0.990 GB/s] 
	TEST PASSED
		gpu       : output: 1660829028 in 5.532672 ms, [ 6064.779 M/s,  24.259 GB/s] 

	Count: 33554432 , NUM_BLOCKS: 64, NUM_THREADS: 128
		host      : output: 1661194096 in 134.173647 ms, [  250.082 M/s,   1.000 GB/s] 
	TEST PASSED
		gpu       : output: 1661194096 in 5.538816 ms, [ 6058.051 M/s,  24.232 GB/s] 

	Count: 33554432 , NUM_BLOCKS: 64, NUM_THREADS: 256
		host      : output: 1661172669 in 111.615326 ms, [  300.626 M/s,   1.203 GB/s] 
	TEST PASSED
		gpu       : output: 1661172669 in 5.302272 ms, [ 6328.312 M/s,  25.313 GB/s] 

	Count: 33554432 , NUM_BLOCKS: 64, NUM_THREADS: 512
		host      : output: 1661138554 in 116.023965 ms, [  289.203 M/s,   1.157 GB/s] 
	TEST PASSED
		gpu       : output: 1661138554 in 13.921280 ms, [ 2410.298 M/s,   9.641 GB/s] 


GPU Device Used:-
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 450.51.06    Driver Version: 450.51.06    CUDA Version: 11.0     |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|                               |                      |               MIG M. |
|===============================+======================+======================|
|   0  Tesla V100-PCIE...  On   | 00000000:41:00.0 Off |                  Off |
| N/A   63C    P0    34W / 250W |      0MiB / 16160MiB |      0%      Default |
|                               |                      |                  N/A |
+-------------------------------+----------------------+----------------------+

CPU Details:-

	Architecture:                    x86_64
	CPU op-mode(s):                  32-bit, 64-bit
	Byte Order:                      Little Endian
	Address sizes:                   43 bits physical, 48 bits virtual
	CPU(s):                          32
	On-line CPU(s) list:             0-31
	Thread(s) per core:              2
	Core(s) per socket:              16
	Socket(s):                       1
	NUMA node(s):                    1
	Vendor ID:                       AuthenticAMD
	CPU family:                      23
	Model:                           1
	Model name:                      AMD Ryzen Threadripper 1950X 16-Core Processor
	Stepping:                        1
	Frequency boost:                 enabled
	CPU MHz:                         1889.447
	CPU max MHz:                     3400.0000
	CPU min MHz:                     2200.0000
	BogoMIPS:                        6786.12
	Virtualization:                  AMD-V
	L1d cache:                       512 KiB
	L1i cache:                       1 MiB
	L2 cache:                        8 MiB
	L3 cache:                        32 MiB
	NUMA node0 CPU(s):               0-31

OS:
	Linux ronl-linux 5.4.0-125-generic #141-Ubuntu SMP Wed Aug 10 13:42:03 UTC 2022 
	x86_64 x86_64 x86_64 GNU/Linux


*/
