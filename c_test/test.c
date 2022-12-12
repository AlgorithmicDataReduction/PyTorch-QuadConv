#include <stdio.h>

void test(float* input, float* output, int n) {
  for (int i=0; i<n; i++) {
    output[i] = input[i];
  }
}
