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
    if(my_rank == 0) {
        char* data = "Advaita";
        int len = 7; 
        MPI_Send(&len, 1, MPI_INT, 2, 1, MPI_COMM_WORLD);
        MPI_Send(data, len, MPI_CHAR, 2, 2, MPI_COMM_WORLD);
        printf("%s with rank %d sent %s\n", processor_name, my_rank, data);
    }
    else if(my_rank == 2){
        int len_data;
        MPI_Recv(&len_data, 1, MPI_INT, MPI_ANY_SOURCE, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        char* recv_data = malloc(sizeof(char)*len_data);
        MPI_Recv(recv_data, len_data, MPI_CHAR, MPI_ANY_SOURCE, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        printf("%s with rank %d recieved %s\n", processor_name, my_rank, recv_data);
    } else {
        printf("Bello world from Slave %s, rank %d.\n", processor_name, my_rank);
    }

    MPI_Finalize();
    return 0;
}


/*

int MPI_Recv(void *buf, int count, MPI_Datatype datatype, int source, int tag, MPI_Comm comm, MPI_Status *status)
int MPI_Send(const void *buf, int count, MPI_Datatype datatype, int dest, int tag, MPI_Comm comm);


*/