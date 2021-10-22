#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

#define NRA 100               /* number of elements in matrix A */
#define MASTER 0               /* taskid of first task */
#define FROM_MASTER 1          /* setting a message type */
#define FROM_WORKER 2          /* setting a message type */

int main (int argc, char *argv[])
{
int	numtasks,              /* number of tasks in partition */
	taskid,                /* a task identifier */
	numworkers,            /* number of worker tasks */
	source,                /* task id of message source */
	dest,                  /* task id of message destination */
	mtype,                 /* message type */
	rows,                  /* rows of matrix A sent to each worker */
	averow, extra, offset, /* used to determine rows sent to each worker */
	i, j, k, rc;           /* misc */
int	a[NRA],           /* matrix A to be added */
	b[NRA],           /* matrix B to be added */
	c[NRA];           /* result matrix C */
MPI_Status status;

MPI_Init(&argc,&argv);
MPI_Comm_rank(MPI_COMM_WORLD,&taskid);
MPI_Comm_size(MPI_COMM_WORLD,&numtasks);
double start, end;
MPI_Barrier(MPI_COMM_WORLD); /* IMPORTANT */
start = MPI_Wtime();
if (numtasks < 2 ) {
  printf("Need at least two MPI tasks. Quitting...\n");
  MPI_Abort(MPI_COMM_WORLD, rc);
  exit(1);
  }
numworkers = numtasks-1;


/* master task */
   if (taskid == MASTER)
   {
    //   printf("Mpi has started with %d tasks.\n",numtasks);
      // printf("Initializing arrays...\n");
      for (i=0; i<NRA; i++)
            a[i]= i+1;
      for (i=0; i<NRA; i++)
            b[i]= i+1;


      /* Send matrix data to the worker tasks */
      averow = NRA/numworkers;
      extra = NRA%numworkers;
      offset = 0;
      mtype = FROM_MASTER;
      for (dest=1; dest<=numworkers; dest++)
      {
         rows = (dest <= extra) ? averow+1 : averow;   	
         MPI_Send(&offset, 1, MPI_INT, dest, mtype, MPI_COMM_WORLD);
         MPI_Send(&rows, 1, MPI_INT, dest, mtype, MPI_COMM_WORLD);
         MPI_Send(&a[offset], rows, MPI_DOUBLE, dest, mtype,MPI_COMM_WORLD);
         MPI_Send(&b[offset], rows, MPI_DOUBLE, dest, mtype, MPI_COMM_WORLD);
         offset = offset + rows;
      }

      /* Receive results from worker tasks */
      mtype = FROM_WORKER;
      for (i=1; i<=numworkers; i++)
      {
         source = i;
         MPI_Recv(&offset, 1, MPI_INT, source, mtype, MPI_COMM_WORLD, &status);
         MPI_Recv(&rows, 1, MPI_INT, source, mtype, MPI_COMM_WORLD, &status);
         MPI_Recv(&c[offset], rows, MPI_DOUBLE, source, mtype, MPI_COMM_WORLD, &status);
        //  printf("Received results from task %d\n",source);
      }

      /* Print results */
      
    //   printf("\n");
      char processor_name[MPI_MAX_PROCESSOR_NAME];
      int name_len;
    
    //   printf("Result Matrix:\n");
      for (i=0; i<NRA; i++)
      {	MPI_Get_processor_name(processor_name, &name_len);
            // printf("%s 	%d\n", processor_name, c[i]);
      }
    //   printf("\n\n");
	exit(0);	
   }


/*worker task */
   if (taskid > MASTER)
   {
      mtype = FROM_MASTER;
      MPI_Recv(&offset, 1, MPI_INT, MASTER, mtype, MPI_COMM_WORLD, &status);
      MPI_Recv(&rows, 1, MPI_INT, MASTER, mtype, MPI_COMM_WORLD, &status);
      MPI_Recv(&a, rows, MPI_DOUBLE, MASTER, mtype, MPI_COMM_WORLD, &status);
      MPI_Recv(&b, rows, MPI_DOUBLE, MASTER, mtype, MPI_COMM_WORLD, &status);
         for (i=0; i<rows; i++)
         {
               c[i] = a[i] + b[i];
         }
      mtype = FROM_WORKER;
      MPI_Send(&offset, 1, MPI_INT, MASTER, mtype, MPI_COMM_WORLD);
      MPI_Send(&rows, 1, MPI_INT, MASTER, mtype, MPI_COMM_WORLD);
      MPI_Send(&c, rows, MPI_DOUBLE, MASTER, mtype, MPI_COMM_WORLD);	
   }
   MPI_Barrier(MPI_COMM_WORLD); /* IMPORTANT */
    end = MPI_Wtime();  
   if(taskid == MASTER) {
       printf("Runtime = %f\n", end-start);
   }
   MPI_Finalize();
   

}