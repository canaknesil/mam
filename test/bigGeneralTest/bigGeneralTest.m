[m, cm, mcm]=textread('/Volumes/somonscratch/mam/PROGRAM/src/bigGeneralTestOutput.txt','%f\n%f\n%f\n');
s=[1 10 100 1000 10000 100000 1000000 10000000 100000000 1000000000];
m
cm
mcm
loglog(s, m, 'b*-')
hold
loglog(s, cm, 'r*-')
%hold
loglog(s, mcm, 'g*-')
legend('malloc', 'cudaMalloc', 'MAM-CudaMalloc')