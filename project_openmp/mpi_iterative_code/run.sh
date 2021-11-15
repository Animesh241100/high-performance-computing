#!/bin/bash
rm a.out utils.h.gch types_header.h.gch
mpicc main.c utils.h types_header.h graphs.c stacks.c hamiltonian_utils.c
mpirun -n $1 ./a.out
