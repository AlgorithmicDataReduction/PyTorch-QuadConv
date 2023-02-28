#include <stdio.h>
#include <stdbool.h>

void agglomerate(bool* a, double* p, int* ei, int* e, int* b, int s, int np, int ne, int nbp, int stages, int factor) {

  printf("C agglomerate");

  for (int i=0; i<np; i++) {
    for (int j=0; j<stages; j++)
    {
      a[i*stages+j] = true;
    }
  }

  return;
}
