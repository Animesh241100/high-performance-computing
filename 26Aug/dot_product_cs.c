#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include<time.h>

#define N 100000

void init_arr(double * arr);
// void print_arr(int * arr);

int main() {
    srand(time(0));
    double arr1[N];
    double arr2[N];
    init_arr(arr1);
    init_arr(arr2);

    double start_time, end_time;
    int num_threads [] = {1, 2, 4, 6, 8, 10, 12, 14, 16, 32, 64, 80, 100, 128, 200, 256, 512};
    int num_threads_size = 17;
    double running_time_thread[num_threads_size];

    double sum;

    for(int j = 0; j < num_threads_size; j++) {
        start_time = omp_get_wtime();
        omp_set_num_threads(num_threads[j]);
        sum = 0;
        double private_sum;
        #pragma omp parallel shared(arr1, arr2, sum) private(private_sum)
        {
            private_sum = 0;
            #pragma omp for
            for(int i = 0; i < N; i++) {
                private_sum += arr1[i] * arr2[i];
            }
            #pragma omp critical(calculateSum)
            {
                sum = sum + private_sum;
            }
        }
        end_time = omp_get_wtime();
        running_time_thread[j] = (end_time - start_time);
        printf("#threads: %d, Dot Product value: %lf\n", num_threads[j], sum);
    }

    for(int j = 0; j < num_threads_size; j++)
        printf("num_threads: %d Running time: %lf\n", num_threads[j], running_time_thread[j]);
    return 0;
    // print_arr(arr2);
    // print_arr(arr1);
}

void init_arr(double * arr) {
    for(int i = 0; i < N; i++)
        arr[i] = (double)(rand() % 100000000) / (double)(10000);
        // scanf("%lf", &arr[i]);
}


/*

int dot_product(int * arr1, int * arr2) {
    int sum = 0;
    int private_sum;
    #pragma omp parallel shared(arr1, arr2, sum) private(private_sum)
    {
        int private_sum = 0;
        #pragma omp for
        for(int i = 0; i < N; i++) {
            private_sum += arr1[i] * arr2[i];
        }
        #pragma omp critical(calculateSum)
        {
            sum = sum + private_sum;
        }

    }
    return sum;
}

*/

// void print_arr(int * arr) {
//     for(int i = 0; i < N; i++) {
//         printf("%d, ", arr[i]);
//     }
//     printf("\n");
// }

