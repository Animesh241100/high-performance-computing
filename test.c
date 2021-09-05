#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include<time.h>

int main() {
    // omp_set_num_threads(3);
    #pragma omp parallel num_threads(3) 
    {
        int num = omp_get_thread_num();
        printf("hello world %d \n", num);
        #pragma omp for collapse(2)
        for(int i = 0; i < 5; i++) {
            // int num = omp_get_thread_num();
            // printf("h1 %d %d\n", i, num);
            for(int j = 0; j < 4; j++)
                printf("h2 %d %d %d\n", i, j, num);
            // printf("\n");
        }
    }
    // #pragma omp parallel for schedule(static)
    // for(int i = 0; i < 8; i++){
    //     int num = omp_get_thread_num();
    //     printf("h %d %d\n", i, num);
    // }
    // printf("\n");

    printf("completed\n");
    return 0;
}