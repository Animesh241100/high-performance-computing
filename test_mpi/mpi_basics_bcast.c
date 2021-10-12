// using bcast

#include<mpi.h>
#include<stdio.h>
#include<stdlib.h>

int main(int argc, char** argv) {
    MPI_Init(NULL, NULL);

    int world_size, my_rank;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    char processor_name[MPI_MAX_PROCESSOR_NAME];
    int name_len;
    MPI_Get_processor_name(processor_name, &name_len);
    
    char* data;
    int len;
    if(my_rank == 0) {
        data = "AdvaitaXP";
        len = 9; 
        printf("%s with rank %d sent %s\n", processor_name, my_rank, data);
    }

    printf("BEFORE: rank %d: data: %s, len :%d\n", my_rank, data, len);
    MPI_Bcast(&len, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if(my_rank != 0)
        data = malloc(sizeof(char)*len);
    MPI_Bcast(data, len, MPI_CHAR, 0, MPI_COMM_WORLD);
    printf("AFTER: rank %d: data: %s, len :%d\n", my_rank, data, len);

    MPI_Finalize();
    return 0;
}


/*

int MPI_Recv(void *buf, int count, MPI_Datatype datatype, int source, int tag, MPI_Comm comm, MPI_Status *status)
int MPI_Send(const void *buf, int count, MPI_Datatype datatype, int dest, int tag, MPI_Comm comm);


*/