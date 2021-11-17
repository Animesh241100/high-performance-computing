__global__ void mykernel(void) {

}


int main(void) {
    mykernel<<<1,1>>>();
    printf("hello world\n");
    return 0;
}