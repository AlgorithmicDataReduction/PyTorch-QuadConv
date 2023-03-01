#include <stdio.h>
#include <stdbool.h>

void agglomerate(bool* a, double* p, int* ep, int* ei, int* bp, int s, int np, int ne, int nbp, int stages, int factor) {

  printf("C agglomerate test\n");

  for (int i=0; i<np; i++) {
    for (int j=0; j<stages; j++)
    {
      a[i*stages+j] = true;
    }
  }

  printf("%i\n", ne);
  printf("%i\n", ep[ne-1]);
  printf("%i\n", ep[ne]);
  printf("%i\n", ep[ne+1]);
  printf("\n");
  printf("%i\n", ei[12684]);
  printf("%i\n", ei[12685]);
  printf("%i\n", ei[12686]);

  for (int i=ep[ne-1]; i<ep[ne]; i++)
  {
    printf("%f, %f\n", p[s*ei[i]], p[s*ei[i]+1]);
  }


  return;
}
