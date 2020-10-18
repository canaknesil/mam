#include <stdio.h>
#include <stdlib.h>

#include "../tree.h"



struct segment {
	
	size_t size;
	
	RB_ENTRY(segment) entry;
	
};


//pointer tree generation
int treeCmp(struct segment *s1, struct segment *s2) {
	if (s1->size < s2->size) {
		return -1;
	} else if (s1->size > s2->size) {
		return 1;
	} else {
		return 0;
	}
}
RB_HEAD(tree, segment) treeHead = RB_INITIALIZER(&treeHead);
RB_GENERATE(tree, segment, entry, treeCmp);


int main() {
	
	struct segment seg1;
	struct segment seg2;
	struct segment seg3;
	seg1.size = 5;
	seg2.size = 3;
	seg3.size = 1;
	
	RB_INSERT(tree, &treeHead, &seg1);
	RB_INSERT(tree, &treeHead, &seg2);
	RB_INSERT(tree, &treeHead, &seg3);
	
	struct segment find, *res;
	find.size = 6;
	res = RB_NFIND(tree, &treeHead, &find);
	
	if (res == NULL) {
		printf("it is null\n");
	} else {
		printf("%d\n", (int) res->size);
	}
	
	res = RB_NEXT(tree, &treeHead, res);
	
	if (res == NULL) {
		printf("it is null\n");
	} else {
		printf("%d\n", (int) res->size);
	}
	
	res = RB_NEXT(tree, &treeHead, res);
	
	if (res == NULL) {
		printf("it is null\n");
	} else {
		printf("%d\n", (int) res->size);
	}
	
	
	return 0;
}


































