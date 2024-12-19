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

    printf("\nusing warp sync func :\n");
    timing(h_x, d_x, 0);
    printf("\nusing shfl_down_sync:\n");
    timing(h_x, d_x, 1);
    printf("\nusing cooperative_groups sync:\n");
    timing(h_x, d_x, 2);

    free(h_x);
    CHECK(cudaFree(d_x));
    return 0;
}

/**
 * @brief 使用线程束同步函数
*/
void __global__ reduce_syncwarp(real *d_x, real *d_y, const int N) {
    const int tid = threadIdx.x;
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    extern __shared__ real s_y[];
    s_y[tid] = (idx < N) ? d_x[idx] : 0.0;
    __syncthreads();
    for (int offset = blockDim.x >> 1; offset >= 32; offset >>= 1) {
        if(tid < offset)
        {
            s_y[tid] += s_y[tid + offset];
        }
        __syncthreads();
    }
    for (int offset = 16; offset > 0; offset >>= 1) {
        if(tid < offset)
        {
            s_y[tid] += s_y[tid + offset];
        }
        __syncwarp();
    }
    if (tid == 0) {
        atomicAdd(d_y, s_y[0]);
    }
}
/**
 * @brief 使用线程束洗牌函数进行规约计算
*/
void __global__ reduce_shfl(real *d_x, real *d_y, const int N) {
    const int tid = threadIdx.x;
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    extern __shared__ real s_y[];
    s_y[tid] = (idx < N) ? d_x[idx] : 0.0;
    __syncthreads();
    for (int offset = blockDim.x >> 1; offset >= 32; offset >>= 1) {
        if(tid < offset)
        {
            s_y[tid] += s_y[tid + offset];
        }
        __syncthreads();
    }
    real y = s_y[tid];
    thread_block_tile<32> g = tiled_partition<32>(this_thread_block());
    for (int i = g.size() >> 1; i > 0; i >>= 1)
    {
        y += g.shfl_down(y, i);
    }
    if (tid == 0) {
        atomicAdd(d_y, y);
    }
}
/**
 * @brief 使用协作组(cooperative groups)进行规约计算
*/
void __global__ reduce_cg(real *d_x, real *d_y, const int N) {
    const int tid = threadIdx.x;
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    extern __shared__ real s_y[];
    s_y[tid] = (idx < N) ? d_x[idx] : 0.0;
    __syncthreads();
    for (int offset = blockDim.x >> 1; offset >= 32; offset >>= 1) {
        if(tid < offset)
        {
            s_y[tid] += s_y[tid + offset];
        }
        __syncthreads();
    }
    real y = s_y[tid];
    for (int offset = 16; offset > 0; offset >>= 1) {
        y += __shfl_down_sync(FULL_MASK, y, offset);
    }
    if (tid == 0) {
        atomicAdd(d_y, y);
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
            reduce_syncwarp<<<girdSize, BLOCK_SIZE, sizeof(real)*BLOCK_SIZE>>>(d_x, d_y, N);
            break;
        case 1:
            reduce_shfl<<<girdSize, BLOCK_SIZE, sizeof(real)*BLOCK_SIZE>>>(d_x, d_y, N);
            break;
        case 2:
            reduce_cg<<<girdSize, BLOCK_SIZE, sizeof(real)*BLOCK_SIZE>>>(d_x, d_y, N);
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