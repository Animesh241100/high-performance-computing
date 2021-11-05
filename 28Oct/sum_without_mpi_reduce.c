// Program to add N numbers without using mpi reduce

#include<mpi.h>
#include<stdio.h>
#include<stdlib.h>
#include<time.h>

// maximum vector size
#define MAX_SIZE 100000

void PrintVector(double * vec, int len, char * array_name);
void InitVector(double * vec, int len);
double TestActualSum(double * vec, int len);

int main(int argc, char** argv) {
    MPI_Init(NULL, NULL);
    int world_size, my_rank;
    double start, end;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    char processor_name[MPI_MAX_PROCESSOR_NAME];
    int name_len;
    MPI_Get_processor_name(processor_name, &name_len);   
    double vector[MAX_SIZE];
    double final_sum = 0;
    MPI_Barrier(MPI_COMM_WORLD); /* IMPORTANT */
    start = MPI_Wtime();
    
    if(my_rank == 0) {
        srand(time(0));
        InitVector(vector, MAX_SIZE);
        // PrintVector(vector, MAX_SIZE, "vec"); // uncomment to check the randomly generated vector
    }

    int len = MAX_SIZE / world_size;
    double scat_vec[len];
    double local_sum = 0;

    MPI_Scatter(vector, len, MPI_DOUBLE, scat_vec, len, MPI_DOUBLE, 0, MPI_COMM_WORLD); // scatter the parts of the vector to all the threads
    for(int i = 0; i < len; i++) 
        local_sum += scat_vec[i];
    // printf("rank %d, sum %f", my_rank, local_sum);
    double local_sum_vec[world_size];
    MPI_Gather(&local_sum, 1, MPI_DOUBLE, local_sum_vec, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    if(my_rank == 0) {
        for(int i = 0; i < MAX_SIZE%world_size; i++)
            final_sum += vector[MAX_SIZE - 1 - i];
        for(int i = 0; i < world_size; i++)
            final_sum += local_sum_vec[i];
    }
    MPI_Barrier(MPI_COMM_WORLD); 
    end = MPI_Wtime();
    MPI_Finalize();
    if(my_rank == 0) {
        printf("Final Sum: %f\n",final_sum);
        printf("Time taken: %f\n", end-start);
        printf("Actual Sum: %f\n",TestActualSum(vector, MAX_SIZE));
    }
    return 0;
}

// prints the vector `vec` with name `array_name`
void PrintVector(double * vec, int len, char * array_name) {
    printf("The Vector %s:\n", array_name);
    for(int i = 0; i < len; i++)
        printf("%f ", vec[i]);
    printf("\n\n");
}

// initializes the vector `vec` with random values
void InitVector(double * vec, int len) {
    for(int i = 0; i < MAX_SIZE; i++){
        vec[i] = (double)(rand() % 1000000) / (double)100;
    }
}

// purely for testing purposes
double TestActualSum(double * vec, int len) {
    double sum = 0;
    for(int i = 0; i < len; i++)
        sum += vec[i];
    return sum;
}
