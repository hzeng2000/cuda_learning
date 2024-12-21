#include "error.cuh"
#include <stdio.h>
// cooperative groups
#include <cooperative_groups.h>
using namespace cooperative_groups;

#ifdef USE_DP
    typedef double real;
#else
    typedef float real;
#endif

const int NUM_REPEATS = 10;
const int N = 100000000;
const int nBytes = N * sizeof(real);
const int BLOCK_SIZE = 512;
const unsigned FULL_MASK = 0xffffffff;

void timing(real *h_x, real *d_x, const int method);

int main(void) {
    real *h_x = (real *)malloc(nBytes);
    for (int i = 0; i < N; i++) {
        h_x[i] = 1.23;
    }
    real *d_x;
    CHECK(cudaMalloc((void **)&d_x, nBytes));

    printf("\nusing parallel process more data and call kernel func twice conduct final reduction:\n");
    timing(h_x, d_x, 0);

    free(h_x);
    CHECK(cudaFree(d_x));
    return 0;
}

/**
 * @brief 使用协作组(cooperative groups)进行规约计算
          首先通过stride把所有数据规约到grid*block的线程中
          通过两次调用核函数完成最终的规约
*/
void __global__ reduce_cg(real *d_x, real *d_y, const int N) {
    const int tid = threadIdx.x;
    const int bid = blockIdx.x;
    extern __shared__ real s_y[];

    real y = 0.0;
    const int stride = blockDim.x * gridDim.x;
    for (int n = bid * blockDim.x + tid; n < N; n += stride) {
        y += d_x[n];
    }
    s_y[tid] = y;
    __syncthreads();

    for (int offset = blockDim.x >> 1; offset >= 32; offset >>= 1) 
    {
        if (tid < offset) {
            s_y[tid] += s_y[tid + offset];
        }
        __syncthreads();
    }

    y = s_y[0];

    thread_block_tile<32> g = tiled_partition<32>(this_thread_block());
    for (int i = g.size() >> 1; i > 0; i >>= 1) 
    {
        y += g.shfl_down(y, i);
    }

    if(tid == 0) 
    {
        d_y[bid] = y;
    }
}

real reduce(real *d_x, const int method) {
    int girdSize = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    real *d_y;
    CHECK(cudaMalloc((void **)&d_y,  sizeof(real)));
    real h_y[1] = {0};
    CHECK(cudaMemcpy(d_y, h_y, sizeof(real), cudaMemcpyHostToDevice));

    switch (method)
    {
        case 0:
            reduce_cg<<<girdSize, BLOCK_SIZE, sizeof(real)*BLOCK_SIZE>>>(d_x, d_y, N);
            reduce_cg<<<1, 1024, sizeof(real)*1024>>>(d_y, d_y, girdSize);
            break;
        default:
            break;
    }

    CHECK(cudaMemcpy(h_y, d_y, sizeof(real), cudaMemcpyDeviceToHost));
    CHECK(cudaFree(d_y));
    return h_y[0];
}

void timing(real *h_x, real *d_x, const int method) {
    real sum = 0;

    for (int i = 0; i < NUM_REPEATS; i++) {
        CHECK(cudaMemcpy(d_x, h_x, nBytes, cudaMemcpyHostToDevice));
        cudaEvent_t start, stop;
        CHECK(cudaEventCreate(&start));
        CHECK(cudaEventCreate(&stop));
        CHECK(cudaEventRecord(start));
        cudaEventQuery(start);

        sum = reduce(d_x, method);

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