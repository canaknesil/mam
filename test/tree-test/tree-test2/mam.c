#include <stdio.h>
#include <stdlib.h>

#include "mam.h"
#include "tree/tree.h"



struct segment {
	
	void *base;
	size_t size;
	char isEmpty;
	
	RB_ENTRY(segment) ptrTreeEntry;
	
};


//pointer tree generation
int ptrTreeCmp(struct segment *s1, struct segment *s2) {
	return (s1->base < s2->base ? -1 : 1);
}
RB_HEAD(ptrTree, segment) ptrTreeHead = RB_INITIALIZER(&ptrTreeHead);
RB_GENERATE(ptrTree, segment, ptrTreeEntry, ptrTreeCmp);




void insert(void *ptr) {
	struct segment *seg = malloc(sizeof(*seg));
	seg->base = ptr;
	RB_INSERT(ptrTree, &ptrTreeHead, seg);
}

void printTree() {
	struct segment *seg;
	struct segment *temp;
	RB_FOREACH_SAFE(seg, ptrTree, &ptrTreeHead, temp) {
		printf("%d\n", (int) seg->base);
	}
}




































