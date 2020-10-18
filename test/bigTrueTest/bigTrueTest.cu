#include <sys/time.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "mam.h"

#define N_EXTRA 20


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

size_t getRandomSize() {
	
	return (size_t) rand() % 9 + 1;
	
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
	
	void *chv = malloc(sizeof(char *) * (n+N_EXTRA));
	char **ch = (char **) chv;
	
	
	srand(time(NULL));
	
	
	
	//begin experience
	double begin, end;
	int i;
	int k = 0;
	for (i=0; i<n; i++) {
		
		size_t size = getRandomSize();
		//printf("size = %lu\n", size);
		int err = allocate(type, (void **) &ch[k++], size, &begin, &end);
		//printf("err = %d\n", err);
		if (err == 0) {
			for (k-=2; k>=0; k--) {
				deallocate(type, ch[k]);
			}
			finalizeMAM(type);
			free(ch);
			printf("Allocation error. Exitting...\n");
			exit(1);
		}
		
	}
	
	double totalTime = 0;;
	
	for (i=0; i<N_EXTRA; i++) {
		
		size_t size = getRandomSize();
		int err = allocate(type, (void **) &ch[k++], size, &begin, &end);
		
		if (err == 0) {
			for (k-=2; k>=0; k--) {
				deallocate(type, ch[k]);
			}
			finalizeMAM(type);
			free(ch);
			printf("Allocation error. Exitting...\n");
			exit(1);
		}
		
		totalTime += end - begin;
		
	}
	
	printf("%e\n", totalTime / N_EXTRA);
	
	
	
	//finalize
	for (i=0; i<k; i++) {
		deallocate(type, (void *) ch[i]);
	}
	
	finalizeMAM(type);
	free(ch);
	
	
	return 0;
}
