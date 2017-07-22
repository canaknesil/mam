#!/bin/bash
#$ -N mamtest
#$ -q short.q
#$ -cwd
#$ -o somonmamoutput.txt
#$ -l hostname=parcore-6-0


./somonmamtest.out

