#include <stdio.h>
#include "error.cuh"

// real may cause precision loss
#ifdef USE_DP
    typedef double real;
#else
    typedef float real;
#endif

const int NUM_REPEAT = 20;
void timing(const real *x, const int N);
real reduce(const real *x, const int N);

int main(void) {
    const int N = 10000000;
    const int nBytes = N * sizeof(real);
    real *x = (real *)malloc(nBytes);
    for (int i = 0; i < N; i++) {
        x[i] = 1.23;
    }
    timing(x, N);
    free(x);
    return 0;
}

void timing(const real *x, const int N) {
    real sum = 0;

    for (int i = 0; i < NUM_REPEAT; i++) {
        cudaEvent_t start, stop;
        CHECK(cudaEventCreate(&start));
        CHECK(cudaEventCreate(&stop));
        CHECK(cudaEventRecord(start));
        cudaEventQuery(start);

        sum = reduce(x, N);

        CHECK(cudaEventRecord(stop));
        CHECK(cudaEventSynchronize(stop));
        real elapsed_time = 0;
        CHECK(cudaEventElapsedTime(&elapsed_time, start, stop));
        printf("execution time of reduce kernel: %f (ms)\n", elapsed_time);
        CHECK(cudaEventDestroy(start));
        CHECK(cudaEventDestroy(stop));
    }

    printf("sum: %f\n", sum);
}

real reduce(const real *x, const int N) {
    real sum = 0;
    for (int i = 0; i < N; i++) {
        sum += x[i];
    }
    return sum;
}