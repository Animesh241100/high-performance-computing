#include<mpi.h>
#include<stdio.h>


int main() {
    printf("Serial code\n");
    MPI_Init(NULL, NULL);
    printf("Parallel code\n");
    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size); // define #processors(threads)
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    char pro_name[MPI_MAX_PROCESSOR_NAME];
    int name_len;
    MPI_Get_processor_name(pro_name, &name_len);
    printf("Proc: %s, rank %d, num_proc: %d\n", pro_name, rank, world_size);
    MPI_Finalize();
}