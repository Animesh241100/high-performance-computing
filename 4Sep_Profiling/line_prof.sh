gcc -fprofile-arcs -ftest-coverage $1 -o $2
./$2
gcov $1