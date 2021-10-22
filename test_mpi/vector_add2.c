// Program to Add two vectors

#include<mpi.h>
#include<stdio.h>
#include<stdlib.h>
#include<time.h>

#define MAX_SIZE 10000

void print_vec(double * vec, int len);

int main(int argc, char** argv) {
    MPI_Init(NULL, NULL);

    int world_size, my_rank;
    double start, end;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    char processor_name[MPI_MAX_PROCESSOR_NAME];
    int name_len;
    MPI_Get_processor_name(processor_name, &name_len);
    
    int vec_size = 10000;
    double vector1[vec_size];
    double vector2[vec_size];
    double final_mult_vector[vec_size];
    MPI_Barrier(MPI_COMM_WORLD); /* IMPORTANT */
    start = MPI_Wtime();
    if(my_rank == 0) {
        srand(time(0));
        for(int i = 0; i < vec_size; i++){
            vector1[i] = (rand() % 1000000) / 100;
            vector2[i] = (rand() % 1000000) / 100;
        }
        // printf("\n\nFirst vector: ");
        // print_vec(vector1, vec_size);
        // printf("\n\nSecond vector: ");
        // print_vec(vector2, vec_size);
        for(int i = 0; i < vec_size%world_size; i++)
            final_mult_vector[vec_size - 1 - i] = vector1[vec_size - 1 - i] * vector2[vec_size - 1 - i];
    }
    int len = vec_size / world_size;
    double scat_vec1[len];
    double scat_vec2[len];
    double mult_vec[len];

    MPI_Scatter(vector1, len, MPI_DOUBLE, scat_vec1, len, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Scatter(vector2, len, MPI_DOUBLE, scat_vec2, len, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    for(int i = 0; i < len; i++) {
        mult_vec[i] = scat_vec1[i] * scat_vec2[i];
    }

    MPI_Gather(mult_vec, len, MPI_DOUBLE, final_mult_vector, len, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Barrier(MPI_COMM_WORLD); 
    end = MPI_Wtime();
    MPI_Finalize();
    if(my_rank == 0) {
        // printf("\n\nFinal vector: ");
        // print_vec(final_mult_vector, vec_size);
        printf("Time taken: %f\n", end-start);
    }
    return 0;
}

void print_vec(double * vec, int len) {
    for(int i = 0; i < len; i++)
        printf("%f ", vec[i]);
    printf("\n");
}
