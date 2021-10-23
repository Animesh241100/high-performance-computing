// Program to Add two vectors

#include<mpi.h>
#include<stdio.h>
#include<stdlib.h>
#include<time.h>

// maximum vector size
#define MAX_SIZE 100000  

void PrintVector(double * vec, int len, char * array_name);
void InitVector(double * vec, int len);
void TestActualSum(double * vec1, double * vec2, int len);

int main(int argc, char** argv) {
    MPI_Init(NULL, NULL);

    int world_size, my_rank;
    double start, end;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    char processor_name[MPI_MAX_PROCESSOR_NAME];
    int name_len;
    MPI_Get_processor_name(processor_name, &name_len);   
    double vector1[MAX_SIZE];
    double vector2[MAX_SIZE];
    double final_sum_vector[MAX_SIZE];
    MPI_Barrier(MPI_COMM_WORLD); /* IMPORTANT */
    start = MPI_Wtime();
    
    if(my_rank == 0) {
        srand(time(0));
        InitVector(vector1, MAX_SIZE);
        InitVector(vector2, MAX_SIZE);
        // PrintVector(vector1, MAX_SIZE, "A"); // uncomment to check the first randomly generated vector
        // PrintVector(vector2, MAX_SIZE, "B"); // uncomment to check the second randomly generated vector
        for(int i = 0; i < MAX_SIZE%world_size; i++)
            final_sum_vector[MAX_SIZE - 1 - i] = vector1[MAX_SIZE - 1 - i] + vector2[MAX_SIZE - 1 - i];
    }

    int len = MAX_SIZE / world_size;
    double scat_vec1[len];
    double scat_vec2[len];
    double sum_vec[len];

    MPI_Scatter(vector1, len, MPI_DOUBLE, scat_vec1, len, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Scatter(vector2, len, MPI_DOUBLE, scat_vec2, len, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    for(int i = 0; i < len; i++) {
        sum_vec[i] = scat_vec1[i] + scat_vec2[i];
    }

    MPI_Gather(sum_vec, len, MPI_DOUBLE, final_sum_vector, len, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Barrier(MPI_COMM_WORLD); 
    end = MPI_Wtime();
    MPI_Finalize();
    if(my_rank == 0) {
        // PrintVector(final_sum_vector, MAX_SIZE, "A + B"); // uncomment to check the sum using MPI
        printf("Time taken: %f\n", end-start);
        // TestActualSum(vector1, vector2, MAX_SIZE); // uncomment to check the actual sum calculated
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
void TestActualSum(double * vec1, double * vec2, int len) {
    printf("This is a testing output: \n");
    for(int i = 0; i < len; i++)
        printf("%f ", vec1[i]+vec2[i]);
    printf("\n\n");   
}