#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include "tree.h"

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

struct segment {
	RB_ENTRY(segment) entry;
	int num;
};

int segTreeCmp(struct segment *s1, struct segment *s2)
{
	return (s1->num < s2->num ? -1 : 1);
}

RB_HEAD(segtree, segment) head = RB_INITIALIZER(&head);
RB_GENERATE(segtree, segment, entry, segTreeCmp);

double test(int treeSize) {
	
	int n = treeSize / 40;
	int total = treeSize + n;
	
	struct segment * segPtrArr = malloc((total) * sizeof(struct segment));
	
	int i;
	for (i=0; i<treeSize; i++) {
		segPtrArr[i].num = i;
		RB_INSERT(segtree, &head, &segPtrArr[i]);
	}
	
	double begin, end, timeSpent;
	begin = getTime();
	
	for ( ; i<total; i++) {
		segPtrArr[i].num = i;
		RB_INSERT(segtree, &head, &segPtrArr[i]);
	}
	
	end = getTime();
	
	
	struct segment * seg;
	struct segment * temp;
	RB_FOREACH_SAFE(seg, segtree, &head, temp) {
		//printf("%d\n", seg->num);
		RB_REMOVE(segtree, &head, seg);
	}
	
	free(segPtrArr);
	
	timeSpent = end - begin;
	return timeSpent / n;
	
}


int main() {
	
	float i;
	for (i=108; i<1000000; i *= 1.1) {
		printf("%e\n", test(i));
	}
	
	
	return 0;
}































