


int MAM_Create(size_t size);
int MAM_Create_auto();

void MAM_Destroy();

size_t MAM_getChunkSize();

int MAM_CudaMalloc(void **ptr, size_t size);
void MAM_CudaFree(void *ptr);