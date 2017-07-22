#include <stdio.h>
#include <stdlib.h>

#include "../tree.h"


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
	struct segment *seg = malloc(sizeof(*seg));
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

void printTreeDic();

//used to insert to size tree dictionary
void sizeTreeDicInsert(struct segment *seg) {
	struct segment *node = RB_FIND(sizeTreeDic, &sizeTreeDicHead, seg);
	if (node != NULL) {
		RB_REMOVE(sizeTreeDic, &sizeTreeDicHead, node);
		seg->next = node;
		node->prev = seg;
		seg->prev = NULL;
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


void printTreeDic() {
	struct segment *seg;
	RB_FOREACH(seg, sizeTreeDic, &sizeTreeDicHead) {
		printf("Size=%zu ", seg->size);
		struct segment *temp;
		temp = seg;
		while (temp->next != NULL) {
			temp = temp->next;
			printf("%zu ", temp->size);
		}
		printf("\n");
	}
	printf("\n");
}


int main() {
	
	struct segment *seg1 = createSegment();
	seg1->size = 2;
	struct segment *seg2 = createSegment();
	seg2->size = 4;
	struct segment *seg3 = createSegment();
	seg3->size = 6;
	struct segment *seg4 = createSegment();
	seg4->size = 8;
	struct segment *seg5 = createSegment();
	seg5->size = 10;
	struct segment *seg6 = createSegment();
	seg6->size = 8;
	struct segment *seg7 = createSegment();
	seg7->size = 10;
	struct segment *seg8 = createSegment();
	seg8->size = 6;
	struct segment *seg9 = createSegment();
	seg9->size = 2;
	
	
	
	sizeTreeDicInsert(seg1);
	printTreeDic();
	
	sizeTreeDicInsert(seg2);
	printTreeDic();
	
	sizeTreeDicInsert(seg3);
	printTreeDic();
	
	sizeTreeDicInsert(seg4);
	printTreeDic();
	printf("PREV OF 6 = %p\n", seg3->prev);
	sizeTreeDicInsert(seg5);
	printTreeDic();
	printf("PREV OF 6 = %p\n", seg3->prev);
	sizeTreeDicInsert(seg6);
	printTreeDic();
	printf("PREV OF 6 = %p\n", seg3->prev);
	sizeTreeDicInsert(seg7);
	printTreeDic();
	printf("PREV OF 6 = %p\n", seg3->prev);
	sizeTreeDicInsert(seg8);
	printTreeDic();
	printf("PREV OF 6 = %p\n", seg3->prev);
	sizeTreeDicInsert(seg9);
	printTreeDic();
	
	
	
	printf("PREV OF 6 = %p\n", seg3->prev);
	sizeTreeDicRemove(seg2);
	printTreeDic();
	
	printf("PREV OF 6 = %p\n", seg3->prev);
	
	sizeTreeDicRemove(seg3);
	printTreeDic();
	sizeTreeDicRemove(seg6);
	printTreeDic();
	
	
}











































