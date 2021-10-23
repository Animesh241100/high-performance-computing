#include<stdio.h>
#include<stdlib.h>


int malloc2dfloat(float ***array, int n, int m);
int free2dfloat(float ***array);

void main() {
    
    float **array;
    int i, j;
    /* ... */
    malloc2dfloat(&array, 10, 10);
    for(i=0;i<10;i++)
        for(j=0;j<10;j++)
            array[i][j]=i+j;
    printf("%f\n", array[1][2]);
    
    free2dfloat(&array);
}



int malloc2dfloat(float ***array, int n, int m) {

    /* allocate the n*m contiguous items */
    float *p = (float *)malloc(n*m*sizeof(float));
    if (!p) return -1;

    /* allocate the row pointers into the memory */
    (*array) = (float **)malloc(n*sizeof(float*));
    if (!(*array)) {
       free(p);
       return -1;
    }

    /* set up the pointers into the contiguous memory */
    for (int i=0; i<n; i++) 
       (*array)[i] = &(p[i*m]);

    return 0;
}

int free2dfloat(float ***array) {
    /* free the memory - the first element of the array is at the start */
    free(&((*array)[0][0]));

    /* free the pointers into the memory */
    free(*array);

    return 0;
}
