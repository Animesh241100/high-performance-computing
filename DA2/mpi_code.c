#include<mpi.h>
#include<stdio.h>
#include<stdlib.h>
#include<time.h>


const int width = 512;
const int height = 512;
int *input_image;
int *output_image;


int get_average(int i, int j);
void read_input_image();
void get_row(int i, char* line, size_t len);
void print_input_image();
void numfy(char * num);
void write_output_image();


int main(int argc, char** argv) {
    MPI_Init(NULL, NULL);
    input_image = (int*)malloc(sizeof(int)*height*width);
    output_image = (int*)malloc(sizeof(int)*height*width);
    int world_size, my_rank;
    double start, end;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    MPI_Barrier(MPI_COMM_WORLD); 
    start = MPI_Wtime();
    if(my_rank == 0) {    
        read_input_image();
        for(int i = 0; i < height%world_size; i++) {
            for(int j = 0; j < height; j++) {
                output_image[(height - 1 - i)*width + j] = get_average(height - 1 - i, j);
            } 
        }
    }
    int len = height/world_size;
    int * output_vec = (int*)malloc(sizeof(int)*len*width);
    MPI_Bcast(input_image, width*height, MPI_INT, 0, MPI_COMM_WORLD);
    for(int i = 0; i < len; i++) {
        for(int j = 0; j < width; j++) {
            output_vec[i*width + j] = get_average(my_rank*len+i, j);
        }
    }
    MPI_Gather(output_vec, len*width, MPI_INT, output_image, len*width, MPI_INT, 0, MPI_COMM_WORLD);

    MPI_Barrier(MPI_COMM_WORLD); 
    end = MPI_Wtime();
    MPI_Finalize();
    if(my_rank == 0) {
        write_output_image();
        printf("Time taken: %f\n", end-start);
    }
    return 0;
}

// returns the average expected in the question
int get_average(int i, int j) {
    int max_val = input_image[i*width + j];
    int min_val = (i == 0) || (j == 0) || (i == height-1) || (j == width-1) ? 0 : max_val;
    for(int x = i-1; x <= i+1 ; x++) {
        for(int y = j-1; y <= j+1; y++) {
            if(x >= 0 && x < height && y >= 0 && y < width){
                int val = input_image[x*width+y];
                if(val > max_val) 
                    max_val = val; 
                if(val < min_val) 
                    min_val = val;
            }
        }
    } 
    return (max_val+min_val)*0.5;
}


// read the input image present in the given file
void read_input_image() {
    FILE* fptr = fopen("lena.txt", "r");
    char* line = NULL;
    size_t len = 0;
    ssize_t read;
    if (fptr == NULL) {
        printf("Error Opening File lena.txt\n");
        exit(EXIT_FAILURE);
    }
    int i = 0;
    while ((read = getline(&line, &len, fptr)) != -1) {
        get_row(i, line, len);
        i++;
    }
    fclose(fptr);
    if (line)
        free(line);
}

// return the value to put in the i'th row of the output_image
void get_row(int i, char* line, size_t len) {
    int j = 0;
    int k = 0;
    while(k < len) {
        if(line[k] >= 48 && line[k] <= 57){
            char num[3] = "xxx";
            int p = 0;
            while(line[k] != ',' && line[k] != '\n') {
                num[p] = line[k];
                p++;
                k++;
            }
            numfy(num);
            input_image[i*width + j] = atoi(num);
            j++;
        }
        k++;
    }
}

// print the pixel values of the input_image
void print_input_image() {
    for(int i = 0; i < height; i++) {
        for(int j = 0; j < width; j++) {
            printf("%d ", input_image[i*width + j]);
        }
        printf("\n");
    }
}

// write the output image pixel intensity values to lena_out.txt file
void write_output_image() {
    FILE* fptr = fopen("lena_out_mpi.txt", "w");
    if (fptr == NULL) {
        printf("Error Opening File lena_out.txt\n");
        exit(EXIT_FAILURE);
    }
    for(int i = 0; i < height; i++) {
        for(int j = 0; j < width; j++) {
            if(j==width-1) 
                fprintf(fptr, "%d\n", output_image[i*width + j]);
            else
                fprintf(fptr, "%d,", output_image[i*width + j]);
        }
    }
}

// modify the string 'num' like a number
void numfy(char * num) {
    if(num[1] == 'x') {
        num[1] = num[0];
        num[0] = '0';
    }
    if(num[2] == 'x') {
        num[2] = num[1];
        num[1] = num[0];
        num[0] = '0';
    }
}