// using bcast

#include<mpi.h>
#include<stdio.h>
#include<stdlib.h>


void print_vec(int * vec, int len);

int main(int argc, char** argv) {
    MPI_Init(NULL, NULL);

    int world_size, my_rank;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    char processor_name[MPI_MAX_PROCESSOR_NAME];
    int name_len;
    MPI_Get_processor_name(processor_name, &name_len);
    
    int vector[10];
    int data_el[2];
    if(my_rank == 0) {
        for(int i = 0; i < 10; i++)
            vector[i] = 2*i;
        printf("%s with rank %d sent: ", processor_name, my_rank);
        print_vec(vector, 10);
    }
    MPI_Scatter(vector, 2, MPI_INT, data_el, 2, MPI_INT, 0, MPI_COMM_WORLD);
    printf("AFTER: rank: %d : ", my_rank);
    print_vec(data_el, 2);

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