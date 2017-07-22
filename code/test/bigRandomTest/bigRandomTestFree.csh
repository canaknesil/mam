
exec=bigRandomTestFree.out

output=bigRandomTestFreeOutput.txt
truncate -s 0 $output

max=15
for i in `seq 1 $max`
do
	./$exec 500 malloc >> $output
done

for i in `seq 1 $max`
do
	./$exec 500 cudaMalloc >> $output
done

for i in `seq 1 $max`
do
	./$exec 500 MAM_CudaMalloc >> $output
done


