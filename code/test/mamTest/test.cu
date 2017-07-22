#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "mam.h"

#define N 50000
#define M 20

int main() {
	
	//int err = MAM_Create(1000000000);
	MAM_Create_auto();
	printf("chunk size: %lu\n", MAM_getChunkSize());
	
	//srand(time(NULL));
	//int num = 0;
	int i, j, k;
	for (k=0; k<M; k++) {
		
		float *arr[N];
		printf("k=%d\n", k);
		for (i=0; i<N; i++) {
			int r = rand()%100 + 10;
			
			int err = MAM_CudaMalloc((void **) &arr[i], r*sizeof(float));
			if (err == 0) {
				printf("CUDA MALLOC ERROR\n");
				exit(1);
			}
			//arr[i] = malloc(r*sizeof(float));
			for (j=0; j<r; j++) {
				//*(arr[i]+j) = num++;
			}
			for (j=0; j<r; j++) {
				//printf("%f\n", *(arr[i]+j));
			}
		}
		
		for (i=0; i<N; i++) {
			if (rand()%2 == 1) {
				MAM_CudaFree(arr[i]);
				//free(arr[i]);
				arr[i] = NULL;
			}
		}
		
	}
	
	
	
	
	
	
	
//	float *arr[N];
//	
//	int t;
//	for (t=0; t<3; t++) {
//		printf("err\n");
//		int i, j;
//		for (i=0; i<N; i++) {
//			
//			int r = rand()%100+10;
//			int err = MAM_CudaMalloc((void **) &arr[i], r * sizeof(float));
//			if (!err) {
//				printf("err\n");
//				return 1;
//			}
//			for (j=0; j<r; j++) {
//				*(arr[i]+j) = 1.1;
//			}
//		}
//		
//		for (i=0; i<N; i++) {
//			if (rand()%2) {
//				MAM_CudaFree(arr[i]);
//			}
//		}
//		
//		
//	}
	
	

	
	MAM_Destroy();
	
	return 0;
}





















