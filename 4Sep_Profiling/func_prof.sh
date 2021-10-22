gcc -pg $1 -o $2
./$2
gprof -b $2 
