// Program to Add two vectors

#include<mpi.h>
#include<stdio.h>
#include<stdlib.h>

#define MAX_SIZE 1000

void print_vec(int * vec, int len);

int main(int argc, char** argv) {
    MPI_Init(NULL, NULL);

    int world_size, my_rank;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    char processor_name[MPI_MAX_PROCESSOR_NAME];
    int name_len;
    MPI_Get_processor_name(processor_name, &name_len);
    
    int vec_size = 19;
    int vector1[vec_size];
    int vector2[vec_size];
    int final_sum_vector[vec_size];
    if(my_rank == 0) {
        for(int i = 0; i < vec_size; i++){
            vector1[i] = 2*i;
            vector2[i] = 3*i;
        }
        printf("First vector: ");
        print_vec(vector1, vec_size);
        printf("Second vector: ");
        print_vec(vector2, vec_size);
        for(int i = 0; i < vec_size%world_size; i++)
            final_sum_vector[vec_size - 1 - i] = vector1[vec_size - 1 - i] + vector2[vec_size - 1 - i];
    }
    int len = vec_size / world_size;
    int scat_vec1[len];
    int scat_vec2[len];
    int sum_vec[len];

    MPI_Scatter(vector1, len, MPI_INT, scat_vec1, len, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Scatter(vector2, len, MPI_INT, scat_vec2, len, MPI_INT, 0, MPI_COMM_WORLD);
    for(int i = 0; i < len; i++) {
        sum_vec[i] = scat_vec1[i] + scat_vec2[i];
    }

    MPI_Gather(sum_vec, len, MPI_INT, final_sum_vector, len, MPI_INT, 0, MPI_COMM_WORLD);
    if(my_rank == 0) {
        printf("Final vector: ");
        print_vec(final_sum_vector, vec_size);
    }
    MPI_Finalize();
    return 0;
}

void print_vec(int * vec, int len) {
    for(int i = 0; i < len; i++)
        printf("%d ", vec[i]);
    printf("\n");
}
