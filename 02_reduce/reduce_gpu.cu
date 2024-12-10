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

void timing(real *h_x, real *d_x, const int method);

int main(void) {
    real *h_x = (real *)malloc(nBytes);
    for (int i = 0; i < N; i++) {
        h_x[i] = 1.23;
    }
    real *d_x;
    CHECK(cudaMalloc((void **)&d_x, nBytes));

    printf("\nUsing global memory only:\n");
    timing(h_x, d_x, 0);
    printf("\nUsing shared memory only:\n");
    timing(h_x, d_x, 1);
    printf("\nUsing dynamic memory only:\n");
    timing(h_x, d_x, 2);

    free(h_x);
    CHECK(cudaFree(d_x));
    return 0;
}

/**
 * @brief 该核函数实现把一个 数组规约为一个长度为 len(x) / blockSize 的数组，对每一个block规约结果到x[0]，然后再转移到d_y[blockIdx.x]
    注意循环里每一次规约后都要等所有线程完成这一次offset的规约，保证数据正确
    [1, 2, 3, 4, 5, 6, 7, 8]
    block1: [1, 2, 3, 4], block2: [5, 6, 7, 8] blockDim = 4
    offset = 2:
        block1: [4, 6, 3, 4], block2: [12, 14, 7, 8]
    offset = 1:
        block1: [10, 6, 3, 4], block2: [26, 14, 7, 8]
    dy[] = []10, 26
 */
void __global__ reduce_global(real *d_x, real *d_y) {
    const int tid = threadIdx.x;
    real *x = d_x + blockIdx.x * blockDim.x;
    
    for (int offset = blockDim.x >> 1; offset > 0; offset >>= 1) {
        if(tid < offset)
        {
            x[tid] += x[tid + offset];
        }
        __syncthreads();
    }
    if (tid == 0) {
        d_y[blockIdx.x] = x[0];
    }
}

/**
 * @brief 对共享内存访问次数越多，该核函数加速效果越明显
          同时
          1. 避免了对全局内存中数据的修改
          2. 不再要求全局内存数组长度N是线程块大小整数倍 
*/
void __global__ reduce_shared(real *d_x, real *d_y) {
    const int tid = threadIdx.x;
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    __shared__ real s_y[BLOCK_SIZE];
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
        d_y[blockIdx.x] = s_y[0];
    }
}
/**
 * @brief 动态共享内存
          核函数的执行配置除了grid block还有第三个参数。即核函数中每个线程块需要定义的动态共享内存的字节数，默认为0
          这个参数的动态性指的是可以在运行时传入，然后加载
          还需要改变共享内存变量的声明方式 
*/
void __global__ reduce_dynamic(real *d_x, real *d_y) {
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
        d_y[blockIdx.x] = s_y[0];
    }
}

real reduce(real *x, const int method) {
    int girdSize = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    real *d_y;
    CHECK(cudaMalloc((void **)&d_y, girdSize * sizeof(real)));
    real *h_y = (real *)malloc(girdSize * sizeof(real));

    switch (method)
    {
    case 0:
        reduce_global<<<girdSize, BLOCK_SIZE>>>(x, d_y);
        break;
    case 1:
        reduce_shared<<<girdSize, BLOCK_SIZE>>>(x, d_y);
        break;
    case 2:
        reduce_dynamic<<<girdSize, BLOCK_SIZE, sizeof(real)*BLOCK_SIZE>>>(x, d_y);
        break;
    default:
        break;
    }

    CHECK(cudaMemcpy(h_y, d_y, girdSize * sizeof(real), cudaMemcpyDeviceToHost));
    real result = 0.0;
    for (int i = 0; i < girdSize; i++) {
        result += h_y[i];
    }
    free(h_y);
    CHECK(cudaFree(d_y));
    return result;
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