// Program to calculate dot product of two vectors

#include<mpi.h>
#include<stdio.h>
#include<stdlib.h>
#include<time.h>

// maximum vector size
#define MAX_SIZE 100000  

void PrintVector(double * vec, int len, char * array_name);
void InitVector(double * vec, int len);
double TestActualDotProduct(double * vec1, double * vec2, int len);

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
    double final_dot_product = 0;
    MPI_Barrier(MPI_COMM_WORLD); /* IMPORTANT */
    start = MPI_Wtime();
    
    if(my_rank == 0) {
        srand(time(0));
        InitVector(vector1, MAX_SIZE);
        InitVector(vector2, MAX_SIZE);
        // PrintVector(vector1, MAX_SIZE, "A"); // uncomment to check the first randomly generated vector
        // PrintVector(vector2, MAX_SIZE, "B"); // uncomment to check the second randomly generated vector
    }

    int len = MAX_SIZE / world_size;
    double scat_vec1[len];
    double scat_vec2[len];

    MPI_Scatter(vector1, len, MPI_DOUBLE, scat_vec1, len, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Scatter(vector2, len, MPI_DOUBLE, scat_vec2, len, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    double local_dot_product = 0;
    for(int i = 0; i < len; i++)
        local_dot_product += scat_vec1[i] * scat_vec2[i];
    MPI_Reduce(&local_dot_product, &final_dot_product, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);  // reducing their sum down to a single number
    if(my_rank == 0) {
        for(int i = 0; i < MAX_SIZE%world_size; i++)
            final_dot_product += vector1[MAX_SIZE - 1 - i] * vector2[MAX_SIZE - 1 - i];
    }
    MPI_Barrier(MPI_COMM_WORLD); 
    end = MPI_Wtime();
    MPI_Finalize();
    if(my_rank == 0) {
        printf("Calculated final dot product: %f\n", final_dot_product);
        printf("Time taken: %f\n", end-start);
        printf("Actual dot product: %f\n", TestActualDotProduct(vector1, vector2, MAX_SIZE));
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
        vec[i] = (double)(rand() % 10000) / (double)100;
    }
}

// purely for testing purposes
double TestActualDotProduct(double * vec1, double * vec2, int len) {
    double product = 0;
    for(int i = 0; i < len; i++)
        product += vec1[i]*vec2[i];
    return product;   
}