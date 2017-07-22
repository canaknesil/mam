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

int allocate(char *type, void **ptr, size_t size) {
	if (strcmp(type, "malloc") == 0) {
		*ptr = malloc(size);
		if (*ptr == NULL) return 0;
	}
	else if (strcmp(type, "cudaMalloc") == 0) {
		int err = cudaMalloc(ptr, size);
		if (err == cudaErrorMemoryAllocation) return 0;
	}
	else if (strcmp(type, "MAM_CudaMalloc") == 0) {
		int err = MAM_CudaMalloc(ptr, size);
		if (err == 0) return 0;
	}
	else {
		return 0;
	}
	return 1;
}

void deallocate (char *type, void *ptr) {
	if (strcmp(type, "malloc") == 0) {
		free(ptr);
	}
	else if (strcmp(type, "cudaMalloc") == 0) {
		cudaFree(ptr);
	}
	else if (strcmp(type, "MAM_CudaMalloc") == 0) {
		MAM_CudaFree(ptr);
	}
}


int main(int argc, char *argv[]) {
	
	//arguments
	if (argc != 4) {
		printf("Invalid number of argument. Exitting...\n");
		exit(1);
	}
	int size = atoi(argv[1]);
	int n = atoi(argv[2]);
	char *type = argv[3];
	
	
	//wake up the device
	char *chWake;
	cudaMalloc( (void **) &chWake, 1);
	cudaFree( (void *) chWake);
	
	
	//prepare
	if (strcmp(type, "MAM_CudaMalloc") == 0) {
		int mamerr = MAM_Create_auto();
		if (mamerr == 0) {
			printf("unable to create mam\n");
			return 0;
		}
	}
	

	void *chv = malloc(sizeof(char *) * n);
	char **ch = (char **) chv;
	
	
	
	int i;
	
	double begin, end, timeSpentPerAlloc;
	
	//begin experience
	begin = getTime();
	
	for (i=0; i<n; i++) {
		
		int err = allocate(type, (void **) &ch[i], size);
		
		if (err == 0) {
			for (i-=1; i>=0; i--) {
				deallocate(type, ch[i]);
			}
			free(ch);
			printf("Allocation error. Exitting...\n");
			exit(1);
		}
		
	}
	
	end = getTime();
	
	timeSpentPerAlloc = (end - begin) / n;
	
	printf("%e\n", timeSpentPerAlloc);
	
	
	
	for (i=0; i<n; i++) {
		deallocate(type, (void *) ch[i]);
	}
	MAM_Destroy();
	free(ch);
	
	
	return 0;
}
