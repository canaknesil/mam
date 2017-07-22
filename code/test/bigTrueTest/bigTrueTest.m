

cuda=textread('/Volumes/somonscratch/mam/PROGRAM/test/bigTrueTest/bigTrueTestOutput-cuda.txt','%f');
n_cuda=[1 10 100 1000 10000 100000 1000000];

mam=textread('/Volumes/somonscratch/mam/PROGRAM/test/bigTrueTest/bigTrueTestOutput-mam.txt','%f');
n_mam=[1 10 100 1000 10000 100000 1000000 10000000];



loglog(n_cuda, cuda, 'r*-');
hold
loglog(n_mam, mam, 'b*-');

set(gca, 'FontSize', 16);

l = legend('cudaMalloc()', 'MAM-CudaMalloc()');
set(l, 'FontSize',16)

t = title('cudaMalloc() vs MAM-CudaMalloc()');
set(t, 'FontSize',20)

xl = xlabel('number of previous allocations');
set(xl, 'FontSize',16)

yl = ylabel('Allocation duration (s)');
set(yl, 'FontSize',16)




