// using bcast

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
    
    int vec_size = 15;
    // scanf("%d", &vec_size);
    printf("loda mera\n");
    int vector1[vec_size];
    int vector2[vec_size];
    int final_sum_vector[vec_size];
    if(my_rank == 0) {
        for(int i = 0; i < vec_size; i++){
            vector1[i] = 2*i;
            vector2[i] = 3*i;
        }
    }
    int len = vec_size / world_size;
    int scat_vec1[len];
    int scat_vec2[len];
    int sum_vec[len];

    MPI_Scatter(vector1, len, MPI_INT, scat_vec1, len, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Scatter(vector2, len, MPI_INT, scat_vec2, len, MPI_INT, 0, MPI_COMM_WORLD);
    printf("BEFORE: rank: %d : ", my_rank);
    print_vec(scat_vec1, len);
    print_vec(scat_vec2, len);
    for(int i = 0; i < len; i++) {
        sum_vec[i] = scat_vec1[i] + scat_vec2[i];
    }

    MPI_Gather(sum_vec, len, MPI_INT, final_sum_vector, len, MPI_INT, 0, MPI_COMM_WORLD);
    if(my_rank == 0) {
        printf("Final vector: ");
        print_vec(final_sum_vector, vec_size);
    }
    // int final_sum = 0;
    // int local_sum = data_el[0] + data_el[1];
    // MPI_Reduce(&local_sum, &final_sum, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
    // printf("AFTER: rank: %d, local_sum: %d, sum: %d\n", my_rank, local_sum, final_sum);
    MPI_Finalize();
    return 0;
}

void print_vec(int * vec, int len) {
    for(int i = 0; i < len; i++)
        printf("%d ", vec[i]);
    printf("\n");
}


/*

int MPI_Recv(void *buf, int count, MPI_Datatype datatype, int source, int tag, MPI_Comm comm, MPI_Status *status)
int MPI_Send(const void *buf, int count, MPI_Datatype datatype, int dest, int tag, MPI_Comm comm);


*/