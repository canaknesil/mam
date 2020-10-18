
exec=bigGeneralTest.out

output=bigGeneralTestOutput.txt
truncate -s 0 $output

for s in 1 10 100 1000 10000 100000 1000000 10000000
do
	./$exec $s 10 malloc >> $output
	./$exec $s 10 cudaMalloc >> $output
	./$exec $s 10 MAM_CudaMalloc >> $output
done

./$exec 100000000 10 malloc >> $output
./$exec 100000000 10 cudaMalloc >> $output
./$exec 100000000 10 MAM_CudaMalloc >> $output

./$exec 1000000000 10 malloc >> $output
./$exec 1000000000 10 cudaMalloc >> $output
./$exec 1000000000 10 MAM_CudaMalloc >> $output
