// #include<mpi.h>
#include<stdio.h>
#include<stdlib.h>
#include<time.h>


int *input_image;
int *output_image;
int width;
int height;


int get_average(int i, int j);
void read_input_image();
void get_row(int i, char* line, size_t len);
void print_input_image();
void numfy(char * num);
void write_output_image();


int main(int argc, char** argv) {
    read_input_image();
    // print_input_image();
    output_image = (int*)malloc(sizeof(int) * height * width);
    
    for(int i = 0; i < height; i++) {
        for(int j = 0; j < width; j++) {
            output_image[i*width + j] = get_average(i, j);
        }
    }
    write_output_image();
    return 0;
}


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



void read_input_image() {
    width = 512;
    height = 512;
    input_image = (int*)malloc(sizeof(int)*height*width);
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


void print_input_image() {
    for(int i = 0; i < height; i++) {
        for(int j = 0; j < width; j++) {
            printf("%d ", input_image[i*width + j]);
        }
        printf("\n");
    }
}

void write_output_image() {
    FILE* fptr = fopen("lena_out.txt", "w");
    if (fptr == NULL) {
        printf("Error Opening File lena_out.txt\n");
        exit(EXIT_FAILURE);
    }
    for(int i = 0; i < height; i++) {
        for(int j = 0; j < width; j++) {
            fprintf(fptr, "%d,", output_image[i*width + j]);
        }
        fprintf(fptr, "\n");
    }
}


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