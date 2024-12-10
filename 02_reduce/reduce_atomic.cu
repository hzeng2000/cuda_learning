#include "error.cuh"
#include <stdio.h>

#ifdef USE_DP
    typedef double real;
#else
    typedef float real;
#endif

const int NUM_REPEATS = 10;
const int N = 100000000;
const int nBytes = N * sizeof(real);
const int BLOCK_SIZE = 512;

void timing(real *h_x, real *d_x);

int main(void) {
    real *h_x = (real *)malloc(nBytes);
    for (int i = 0; i < N; i++) {
        h_x[i] = 1.23;
    }
    real *d_x;
    CHECK(cudaMalloc((void **)&d_x, nBytes));

    printf("\natomic dynamic shared memory:\n");
    timing(h_x, d_x);

    free(h_x);
    CHECK(cudaFree(d_x));
    return 0;
}

/**
 * @brief 动态共享内存 原子操作
*/
void __global__ reduce(real *d_x, real *d_y) {
    const int tid = threadIdx.x;
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    extern __shared__ real s_y[];
    s_y[tid] = (idx < N) ? d_x[idx] : 0.0;
    __syncthreads();
    for (int offset = blockDim.x >> 1; offset > 0; offset >>= 1) {
        if(tid < offset)
        {
            s_y[tid] += s_y[tid + offset];
        }
        __syncthreads();
    }
    if (tid == 0) {
        atomicAdd(d_y, s_y[0]);
    }
}

real reduce(real *d_x) {
    int girdSize = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    real *d_y;
    CHECK(cudaMalloc((void **)&d_y, girdSize * sizeof(real)));
    real *h_y = (real *)malloc(girdSize * sizeof(real));

    reduce<<<girdSize, BLOCK_SIZE, sizeof(real)*BLOCK_SIZE>>>(d_x, d_y);

    CHECK(cudaMemcpy(h_y, d_y, girdSize * sizeof(real), cudaMemcpyDeviceToHost));
    real result = 0.0;
    for (int i = 0; i < girdSize; i++) {
        result += h_y[i];
    }
    free(h_y);
    CHECK(cudaFree(d_y));
    return result;
}

void timing(real *h_x, real *d_x) {
    real sum = 0;

    for (int i = 0; i < NUM_REPEATS; i++) {
        CHECK(cudaMemcpy(d_x, h_x, nBytes, cudaMemcpyHostToDevice));
        cudaEvent_t start, stop;
        CHECK(cudaEventCreate(&start));
        CHECK(cudaEventCreate(&stop));
        CHECK(cudaEventRecord(start));
        cudaEventQuery(start);

        sum = reduce(d_x);

        CHECK(cudaEventRecord(stop));
        CHECK(cudaEventSynchronize(stop));
        float elapsedTime;
        CHECK(cudaEventElapsedTime(&elapsedTime, start, stop));
        printf("Average time: %f ms\n", elapsedTime / NUM_REPEATS);
        
        CHECK(cudaEventDestroy(start));
        CHECK(cudaEventDestroy(stop));
    }
    printf("Sum: %f\n", sum);
}