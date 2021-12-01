nvcc -dc main.cu graphs.cu stacks.cu hamiltonian_utils.cu 
nvcc main.o graphs.o stacks.o hamiltonian_utils.o
./a.out
