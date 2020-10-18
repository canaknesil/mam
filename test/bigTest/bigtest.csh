
exec=bigtest.out

truncate -s 0 bigtestoutput.txt

for s in 1 10 100 1000 10000 100000 1000000 10000000
do
	./$exec $s 5 >> bigtestoutput.txt
done

./$exec 100000000 5 >> bigtestoutput.txt
./$exec 1000000000 5 >> bigtestoutput.txt
