#include <stdio.h>
#include <stdlib.h>

#include "mam.h"



int main() {
	int i;
	for (i=0; i<12; i++) {
		insert((void *) i);
	}
	printTree();
	
	
	
	return 0;
}





















