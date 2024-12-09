#include "error.cuh"
#include <stdio.h>

#ifdef USE_DP
    typedef double real;
#else
    typedef float real;
#endif

const int NUM_REPEATS = 100;
const int N = 100000000;
const int nBytes = N * sizeof(real);
const int BLOCK_SIZE = 512;

void timing(real *h_x, real *d_x, const int method);



int main(void) {

}