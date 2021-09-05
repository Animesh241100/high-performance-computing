
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
  
int main(int argc, char* argv[])
{
    int nthreads, tid;
  
    #pragma omp parallel private(tid)
    {
        tid = omp_get_thread_num();
        nthreads = omp_get_num_threads();
        printf("ID: %d, Number of threads = %d\n",
                   tid, nthreads);
    }
    return 0;
}