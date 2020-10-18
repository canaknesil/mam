

raw=textread('/Volumes/somonscratch/mam/PROGRAM/test/bigRandomTest/bigRandomTestOutput-MATLAB.txt','%f');
raw = raw';

nRange=9;
n=15;

mallocAll=raw(1 : nRange*n);
cudaMallocAll=raw(nRange*n+1 : 2*nRange*n);
MAM_CudaMallocAll=raw(2*nRange*n+1 : 3*nRange*n);

mallocAv=zeros(1, nRange);
cudaMallocAv=zeros(1, nRange);
MAM_CudaMallocAv=zeros(1, nRange);

for k=1:n-1
    (k+1)*nRange
    mallocAv = mallocAv + mallocAll(k*nRange+1:(k+1)*nRange);
    cudaMallocAv = cudaMallocAv + cudaMallocAll(k*nRange+1:(k+1)*nRange);
    MAM_CudaMallocAv = MAM_CudaMallocAv + MAM_CudaMallocAll(k*nRange+1:(k+1)*nRange);
end

mallocAv = mallocAv ./ n;
cudaMallocAv = cudaMallocAv ./ n;
MAM_CudaMallocAv = MAM_CudaMallocAv ./ n;

ranges = 5.5 * 10.^(0:nRange-1);

%loglog(ranges, mallocAv, 'g*-');
%hold
loglog(ranges, cudaMallocAv, 'r*-');
%hold
%loglog(ranges, MAM_CudaMallocAv, 'b*-');

set(gca, 'FontSize',16)

%l = legend('cudaMalloc()', 'MAM-CudaMalloc()');
%set(l, 'FontSize',16)

%t = title('cudaMalloc() vs MAM-CudaMalloc()');
t = title('cudaMalloc()');
set(t, 'FontSize',20)

xl = xlabel('Allocation Size (byte)');
set(xl, 'FontSize',16)

yl = ylabel('Allocation duration (s)');
set(yl, 'FontSize',16)




