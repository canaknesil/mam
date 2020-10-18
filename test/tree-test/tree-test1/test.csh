#!/bin/bash
#$ -N RBTreeTest
#$ -q short.q
#$ -cwd
#$ -o RBTreeOutput.txt



exec=test.out

#for s in 1 10 100 1000 10000 100000 1000000 10000000
#do
	./$exec
#done

#./$exec 100000000 10
#./$exec 1000000000 1
