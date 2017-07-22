
exec=bigTrueTest.out

output=bigTrueTestOutput.txt
truncate -s 0 $output


for i in 1 10 100 1000 10000 100000 1000000 10000000 100000000 1000000000
do
./$exec $i cudaMalloc >> $output
#./$exec $i MAM_CudaMalloc >> $output
done

