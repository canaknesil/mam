#include <sys/time.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "mam.h"


double getTime()
{
	
	const double kMicro = 1.0e-6;
	struct timeval TV;
	
	const int RC = gettimeofday(&TV, NULL);
	
	if(RC == -1)
	{
		printf("ERROR: Bad call to gettimeofday\n"); return(-1);
	}
	
	return( ((double)TV.tv_sec) + kMicro * ((double)TV.tv_usec) );
	
}


int main(int argc, char *argv[]) {
	
	//arguments
	if (argc != 3) {
		printf("Invalid number of argument. Exitting...\n");
		exit(1);
	}
	int size = atoi(argv[1]);
	int n = atoi(argv[2]);
	
	
	//wake up the device
	char *chWake;
	cudaMallocHost( (void **) &chWake, 1);
	cudaFreeHost( (void *) chWake);
	
	
	//prepare
	int mamerr = MAM_Create_auto();
	if (mamerr == 0) {
		printf("unable to create mam\n");
		return 0;
	}

	void *chv = malloc(sizeof(char *) * n);
	char **ch = (char **) chv;
	
	
	
	int i;
	
	double begin, end, timeSpentPerAlloc;
	
	//begin experience
	begin = getTime();
	
	for (i=0; i<n; i++) {
		//cudaError_t err = cudaMalloc( (void **) &ch[i], size);
		int err = MAM_CudaMalloc( (void **) &ch[i], size);
		
		if (err == 0) {
			for (i-=1; i>=0; i--) {
				MAM_CudaFree(ch[i]);
			}
			free(ch);
			printf("cudaErrorMemoryAllocation. Exitting...\n");
			exit(1);
		}
		
	}
	
	end = getTime();
	
	timeSpentPerAlloc = (end - begin) / n;
	
	printf("%e\n", timeSpentPerAlloc);
	
	
	
	for (i=0; i<n; i++) {
		MAM_CudaFree( (void *) ch[i]);
	}
	MAM_Destroy();
	free(ch);
	
	
	return 0;
}
