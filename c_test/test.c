#include <stdio.h>
#include <stdbool.h>

void agglomerate(bool* a, double* p, int* ei, int* e, int* b, int s, int np, int ne, int nbp, int stages, int factor) {

  printf("C agglomerate");

  for (int i=0; i<np; i++) {
    a[i] = true;
  }

  return;
}
