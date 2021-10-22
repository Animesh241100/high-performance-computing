#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include<time.h>
#include<math.h>

#define N 100000

void print_arr(double * arr);

int main() {
    double arr[N];
    double start_time, end_time;

    // Already calculated the value of 1/root(2)
    double gp_factor = 0.707106;   

    int num_threads [] = {1, 2, 4, 6, 8, 10, 12, 14, 16, 32, 64, 80, 100, 128, 200, 256, 512};
    int num_threads_size = 17;
    
    for(int j = 0; j < num_threads_size; j++) {    // for various number of threads
        start_time = omp_get_wtime();   // record the starting time
        omp_set_num_threads(num_threads[j]);   // set the number of threads required
        #pragma omp parallel for
        for(int i = 0; i < N; i++) {   // there are so many iterations - this is the hotspot for parallelisation
            arr[i] = 20482048.00 * pow(gp_factor, i);  // General Formula - A[n] = E * gp_factor^n     where E = 0th element and ^ represents the power function
        }
        end_time = omp_get_wtime();    // record the ending time
        printf("#threads: %d     Running time: %f\n", num_threads[j], end_time - start_time);
        // print_arr(arr);      // Uncomment if you wish to print the output sequence
    }
    return 0;
}

// A utility function to print the array (sequence) generated
void print_arr(double * arr) {
    for(int i = 0; i < N; i++) {
        printf("%f, ", arr[i]);
    }
    printf("\n");
}

