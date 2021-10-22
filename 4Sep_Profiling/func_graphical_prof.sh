gcc -pg $1 -o $2
./$2
gprof $2 > $2.gprof
python gprof2dot.py < $2.gprof | dot -Tsvg -o new_$2.svg
xdg-open new_$2.svg
