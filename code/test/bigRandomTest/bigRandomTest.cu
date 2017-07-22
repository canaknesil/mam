#include <sys/time.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "mam.h"

#define histSize 9


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

int allocate(char *type, void **ptr, size_t size, double *begin, double *end) {
	if (strcmp(type, "malloc") == 0) {
		
		*begin = getTime();
		*ptr = malloc(size);
		*end = getTime();
		
		if (*ptr == NULL) return 0;
	}
	else if (strcmp(type, "cudaMalloc") == 0) {
		
		*begin = getTime();
		int err = cudaMalloc(ptr, size);
		*end = getTime();
		
		if (err == cudaErrorMemoryAllocation) return 0;
	}
	else if (strcmp(type, "MAM_CudaMalloc") == 0) {
		
		*begin = getTime();
		int err = MAM_CudaMalloc(ptr, size);
		*end = getTime();
		
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



void wakeUp() {
	char *chWake;
	cudaMalloc( (void **) &chWake, 1);
	cudaFree( (void *) chWake);
}

void prepareMAM(char *type) {
	if (strcmp(type, "MAM_CudaMalloc") == 0) {
		int mamerr = MAM_Create_auto();
		if (mamerr == 0) {
			printf("unable to create mam\n");
			exit(0);
		}
	}
}

void finalizeMAM(char *type) {
	if (strcmp(type, "MAM_CudaMalloc") == 0) {
		MAM_Destroy();
	}
}

size_t getRandomSize(int range) {
	
	int min = pow(10, range);
	int max = pow(10, range+1);
	int size = rand() % (max-min) + min;
	return (size_t) size;
	
}

int getRandomRange() {
	int r = rand() % 100;
	if (r < 1) return 8;
	if (r < 3) return 7;
	if (r < 6) return 6;
	if (r < 10) return 5;
	if (r < 25) return 4;
	if (r < 40) return 3;
	if (r < 55) return 2;
	if (r < 70) return 1;
	if (r < 85) return 0;
	return 0;
}


int main(int argc, char *argv[]) {
	
	//arguments
	int n;
	char *type;
	if (argc != 3) {
		printf("Invalid number of argument. Exitting...\n");
		exit(1);
	}
	n = atoi(argv[1]);
	type = argv[2];
	
	
	//wake up the device
	wakeUp();
	
	
	//prepare
	prepareMAM(type);
	
	void *chv = malloc(sizeof(char *) * n);
	char **ch = (char **) chv;
	
	int histCount[histSize];
	double histTime[histSize];
	
	int i;
	for (i=0; i<histSize; i++) {
		histCount[i] = 0;
		histTime[i] = 0;
	}
	
	
	srand(time(NULL));
	
	
	
	//begin experience
	double begin, end;
	
	for (i=0; i<n; i++) {
		
		int range = getRandomRange();
		//printf("range = %d\n", range);
		size_t size = getRandomSize(range);
		//printf("size = %lu\n", size);
		
		
		int err = allocate(type, (void **) &ch[i], size, &begin, &end);
		
		
		if (err == 0) {
			for (i-=1; i>=0; i--) {
				deallocate(type, ch[i]);
			}
			finalizeMAM(type);
			free(ch);
			printf("Allocation error. Exitting...\n");
			exit(1);
		}
		
		histCount[range]++;
		histTime[range] += end - begin;
		
	}
	
	for (i=0; i<histSize; i++) {
		
		int count = histCount[i];
		double time = histTime[i] / (count == 0 ? 1 : count);
		
		//printf("count = %d\n", count);
		printf("%e\n", time);
		
	}
	
	
	
	
	//finalize
	for (i=0; i<n; i++) {
		deallocate(type, (void *) ch[i]);
	}
	
	finalizeMAM(type);
	free(ch);
	
	
	return 0;
}
