// Copyright (c) 2017 Can Aknesil
// 
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
// 
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
// 
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>

#include "mam.h"
#include "tree/tree.h"


//segment structure representing all segments in the chunk
struct segment {
	
	void *base;         //pointer to the begining of the segment
	size_t size;        //size of the segment
	char isEmpty;       //boolean value: true if it is assigned to user
	
	RB_ENTRY(segment) ptrTreeEntry;      //entry for the pointer tree
	RB_ENTRY(segment) sizeTreeDicEntry;  //entry for the size tree dictionary
	
	struct segment *next;  //next and previous segment with same size in size tree dictionary
	struct segment *prev;
	
};

//used to create segment
struct segment *createSegment() {
	struct segment *seg = (struct segment *) malloc(sizeof(*seg));
	seg->base = NULL;
	seg->size = 0;
	seg->isEmpty = 1;
	seg->next = NULL;
	seg->prev = NULL;
	return seg;
}

//used to free segment
void destroySegment(struct segment *seg) {
	free(seg);
	seg = NULL;
}



//pointer tree generation
int ptrTreeCmp(struct segment *s1, struct segment *s2) {
	if (s1->base < s2->base) {
		return -1;
	} else if (s1->base > s2->base) {
		return 1;
	} else {
		return 0;
	}
}
RB_HEAD(ptrTree, segment) ptrTreeHead = RB_INITIALIZER(&ptrTreeHead);
RB_GENERATE(ptrTree, segment, ptrTreeEntry, ptrTreeCmp);



//size tree dictionary generation
int sizeTreeDicCmp(struct segment *s1, struct segment *s2) {
	if (s1->size < s2->size) {
		return -1;
	} else if (s1->size > s2->size) {
		return 1;
	} else {
		return 0;
	}
}
RB_HEAD(sizeTreeDic, segment) sizeTreeDicHead = RB_INITIALIZER(&sizeTreeDicHead);
RB_GENERATE(sizeTreeDic, segment, sizeTreeDicEntry, sizeTreeDicCmp);

//used to insert to size tree dictionary
void sizeTreeDicInsert(struct segment *seg) {
	struct segment *node = RB_FIND(sizeTreeDic, &sizeTreeDicHead, seg);
	if (node != NULL) {
		RB_REMOVE(sizeTreeDic, &sizeTreeDicHead, node);
		seg->next = node;
		node->prev = seg;
	}
	RB_INSERT(sizeTreeDic, &sizeTreeDicHead, seg);
}

//used to remove from size tree dictionary
void sizeTreeDicRemove(struct segment *seg) {
	if (seg->prev == NULL) {
		RB_REMOVE(sizeTreeDic, &sizeTreeDicHead, seg);
		struct segment *nextSeg = seg->next;
		seg->next = NULL;
		if (nextSeg != NULL) {
			nextSeg->prev = NULL;
			RB_INSERT(sizeTreeDic, &sizeTreeDicHead, nextSeg);
		}
	} else {
		seg->prev->next = seg->next;
		if (seg->next != NULL) {
			seg->next->prev = seg->prev;
		}
		seg->prev = NULL;
		seg->next = NULL;
	}
}

//used to find a best fitting segment
struct segment *findBestFit(struct segment *find) {
	struct segment *node = RB_ROOT(&sizeTreeDicHead);
	if (node == NULL) {
		return NULL;
	}
	
	struct segment *old;
	
	while (node != NULL) {
		int cmpResult = sizeTreeDicCmp(find, node);
		if (cmpResult == 0) {
			return node;
		} else if (cmpResult == -1) {
			old = node;
			node = RB_LEFT(node, sizeTreeDicEntry);
		} else {
			old = node;
			node = RB_RIGHT(node, sizeTreeDicEntry);
		}
	}
	
	if (sizeTreeDicCmp(old, find) == -1) {
		return RB_NEXT(sizeTreeDic, &sizeTreeDicHead, old);
	} else {
		return old;
	}
	
}


//pointer of the big chunk of memory allocated at the initialization
void *chunk = NULL;
size_t chunkSize = 0;
pthread_mutex_t mam_mutex;


//used to get the chunk size
size_t MAM_getChunkSize() {
	return chunkSize;
}


//used for debugging
void printAllSegments() {
	struct segment *seg;
	int i=0;
	RB_FOREACH(seg, ptrTree, &ptrTreeHead) {
		printf("Segment %d: base=%p, size=%zu, isEmpty=%d\n", i++, seg->base, seg->size, seg->isEmpty);
	}
	printf("\n");
}


//called at the begining of code to initialize
int MAM_Create(size_t size) {
	
	int err = cudaMalloc(&chunk, size);
	if (err == cudaErrorMemoryAllocation) {
		printf("CUDA ALLOCATION ERROR\n");
		return 0;
	}
	
	chunkSize = size;
	
	//initialize mutex lock
	pthread_mutex_init(&mam_mutex, NULL);
	
	//allocate chunk
//	chunk = malloc(size);
//	if (chunk == NULL) {
//		return 0;
//	}
	
	
	
	//create initial empty segment
	struct segment *seg = (struct segment *) malloc(sizeof(*seg));
	seg->base = chunk;
	seg->size = size;
	seg->isEmpty = 1;
	
	//initialize pointer tree
	RB_INSERT(ptrTree, &ptrTreeHead, seg);
	
	//initialize size tree dictionary
	RB_INSERT(sizeTreeDic, &sizeTreeDicHead, seg);
	
	return 1;
}

//called at the begining of code to initialize without specifiing the size
int MAM_Create_auto() {
	
	//find maximum possible allocation size
	size_t free;
	size_t total;
	int err = cudaMemGetInfo(&free, &total);
	if (err != cudaSuccess) {
		printf("CUDA MEMGETINFO ERROR\n");
		return 0;
	}
	
	int waste;
	for (waste=1000000; waste < 1000000000; waste *= 2) {
		//create
		size_t size = free - waste;
		if (size <= 0) {
			continue;
		}
		if (MAM_Create(size)) {
			return 1;
		}
		printf("cant allocate %lu\n", size);
	}
	printf("UNABLE TO CREATE\n");
	return 0;
	
}


//called at the end of the code to finalize
void MAM_Destroy() {
	
	//destroy mutex lock
	pthread_mutex_destroy(&mam_mutex);
	
	//reset size tree by removing all elements
	struct segment *seg;
	struct segment *temp;
	RB_FOREACH_SAFE(seg, sizeTreeDic, &sizeTreeDicHead, temp) {
		
		RB_REMOVE(sizeTreeDic, &sizeTreeDicHead, seg);
	}
	
	//reset pointer tree by removing and freeing all segments
	RB_FOREACH_SAFE(seg, ptrTree, &ptrTreeHead, temp) {
		
		RB_REMOVE(ptrTree, &ptrTreeHead, seg);
		free(seg);
	}
	
	//free the chunk
	cudaFree(chunk);
	chunk = NULL;
	chunkSize = 0;
	
}

//called to allocate memory
int MAM_CudaMalloc(void **ptr, size_t size) {
	
	if (chunk == NULL) {
		if (!MAM_Create_auto()) {
			return 0;
		}
	}
	
//	//check wheather MAM is created.
//	if (chunk == NULL) {
//		printf("MAM IS NOT CREATED\n");
//		return 0;
//	}
	
	//lock
	pthread_mutex_lock(&mam_mutex);
	
	//Find a best-fitting empty segment using tree-dictionary. O(log(n))
	struct segment find, *found;
	find.size = size;
	found = findBestFit(&find);
	if (found == NULL) {
		//this happens in case there is no bigger segment
		printf("THERE IS NO BIG ENOUGH EMPTY SEGMENT\n");
		return 0;
	}
	if (found->size < size) {
		//just to be sure
		return 0;
	}
	
	//Mark the segment as filled. O(1)
	found->isEmpty = 0;
	
	//Remove it from tree-dictionary. O(log(n))
	RB_REMOVE(sizeTreeDic, &sizeTreeDicHead, found);
	struct segment *next = found->next;
	found->next = NULL;
	if (next != NULL) {
		next->prev = NULL;
		RB_INSERT(sizeTreeDic, &sizeTreeDicHead, next);
	}
	
	//If do not perfectly fits, O(1)
	if (found->size > size) {
		
		//Resize it. O(1)
		size_t oldSize = found->size;
		found->size = size;
		
		//Create a new empty segment with corresponding base pointer and size. O(1)
		struct segment *newSeg = createSegment();
		newSeg->base = (char *) found->base + size;
		newSeg->size = oldSize - size;
		newSeg->isEmpty = 1;
		
		//Insert it in pointer-tree. O(log(n))
		RB_INSERT(ptrTree, &ptrTreeHead, newSeg);
		
		//Insert it in tree-dictionary. O(log(n))
		sizeTreeDicInsert(newSeg);
		
	}
	
	//Return base pointer of filled segment. O(1)
	*ptr = found->base;
	
	//printAllSegments();
	
	//unlock
	pthread_mutex_unlock(&mam_mutex);
	
	return 1;
}

//called to deallocate memory
void MAM_CudaFree(void *ptr) {
	
	//check wheather MAM is created.
	if (chunk == NULL) {
		printf("MAM IS NOT CREATED\n");
		return;
	}
	
	//lock
	pthread_mutex_lock(&mam_mutex);
	
	//Find the segment with pointer-tree. O(log(n))
	struct segment find;
	find.base = ptr;
	struct segment *found = RB_FIND(ptrTree, &ptrTreeHead, &find);
	if (found == NULL) {
		//this should never happen
		return;
	}
	
	//Mark it as empty and insert it in tree dictionary. O(1)
	found->isEmpty = 1;
	sizeTreeDicInsert(found);
	
	//Get previous and next segments. O(log(n))
	struct segment *prev = RB_PREV(ptrTree, &ptrTreeHead, found);
	struct segment *next = RB_NEXT(ptrTree, &ptrTreeHead, found);
	struct segment *upper = found;
	//If previous segment is empty, O(1)
	if (prev != NULL && prev->isEmpty) {
		//Remove the found segment from the pointer-tree and tree dictionary then free it. O(log(n))
		RB_REMOVE(ptrTree, &ptrTreeHead, found);
		sizeTreeDicRemove(found);
		size_t foundSize = found->size;
		destroySegment(found);
		
		//Resize previous segment with corresponding size. O(1)
		prev->size += foundSize;
		
		//Replace previous segment in tree-dictionary. O(log(n))
		
		//remove
		sizeTreeDicRemove(prev);
		
		//insert
		sizeTreeDicInsert(prev);
		
		upper = prev;
	}
	
	//If next segment is empty, O(1)
	if (next != NULL && next->isEmpty) {
		//Remove the next segment from pointer-tree and tree-dictionary then free it . O(log(n))
		
		//remove from pointer tree
		RB_REMOVE(ptrTree, &ptrTreeHead, next);
		
		//remove from tree dictionary
		sizeTreeDicRemove(next);
		size_t nextSize = next->size;
		destroySegment(next);
		
		//Resize upper segment with corresponding size. O(1)
		upper->size += nextSize;
		
		//Replace it in tree-dictionary. O(log(n))
		sizeTreeDicRemove(upper);
		sizeTreeDicInsert(upper);
		
		
	}
	
	//printAllSegments();
	
	//unlock
	pthread_mutex_unlock(&mam_mutex);
	
}

































