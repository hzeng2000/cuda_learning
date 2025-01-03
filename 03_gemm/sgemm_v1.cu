#include <stdio.h>
#include <stdlib.h>
#include <float.h>
#include <cuda_runtime.h>

const int BLOCK_SIZE = 64;
#define OFFSET(row, col, ld) ((row) * (ld) + (col))
#define FLOAT4(pointer) (reinterpret_cast<float4*>(&(pointer))[0])

float testError(void);
float testPerformance(
    void (*gpuSgemm) (float *, float *, float *, const int, const int, const int),
    dim3 gridDim, dim3 blockDim, const int M, const int N, const int K, const int repeat);

void cpuSgemm(
    float *a, float *b, float *c, const int M, const int N, const int K)
{
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            float sum = 0.0f;
            for (int k = 0; k < K; k++) {
                sum += a[OFFSET(i, k, K)] * b[OFFSET(k, j, N)];
            }
            c[OFFSET(i, j, N)] = sum;
        }
    }
}

/**
 * @brief
 *  利用 share memory 优化 naiveSgemm
 * @param a
 * @param b
 * @param c
 * @param M
 * @param N
 * @param K
 */
__global__ void Sgemm_v1(
    float *a, float *b, float *c, const int M, const int N, const int K) {
    int nCol = blockIdx.x * blockDim.x + threadIdx.x;
    int nRow = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (nRow < M && nCol < N) {
        __shared__ float shTileA[BLOCK_SIZE][BLOCK_SIZE];
        __shared__ float shTileB[BLOCK_SIZE][BLOCK_SIZE];
        const int nIter = (K + BLOCK_SIZE - 1) / BLOCK_SIZE;
        float sum = 0.0f;
        for (int k = 0; k < nIter; k++) 
        {
            // load data from global memory to shared memory
            // attention for shTileA[threadIdx.y][threadIdx.x]
            if (k * BLOCK_SIZE + threadIdx.x < K) {
                shTileA[threadIdx.y][threadIdx.x] = a[OFFSET(nRow, k * BLOCK_SIZE + threadIdx.x, K)];
            }
            else {
                shTileA[threadIdx.y][threadIdx.x] = 0.0f;
            }
            if (k * BLOCK_SIZE + threadIdx.y < K) {
                shTileB[threadIdx.y][threadIdx.x] = b[OFFSET(k * BLOCK_SIZE + threadIdx.y, nCol, N)];
            }
            else {
                shTileB[threadIdx.y][threadIdx.x] = 0.0f;
            }
            // sync to wait for all threads to load data
            __syncthreads();
            // sub-matrix multiplication
            for (int n = 0; n < BLOCK_SIZE; n++) 
            {
                sum += shTileA[threadIdx.y][n] * shTileB[n][threadIdx.x];
            }
            // sync to wait for all threads to finish computation
            __syncthreads();
        }
        c[OFFSET(nRow, nCol, N)] = sum;
    }
}

int main(void) 
{
    float maxError = testError();
    printf("Max error: %f\n", maxError);

    printf("\nKernel = Sgemm_v1\n");
    const int M_list[15] = {128, 192, 256, 384, 512, 768, 1024, 1536, 2048, 3072, 4096, 6144, 8192, 12288, 16384};
    const int N_list[15] = {128, 192, 256, 384, 512, 768, 1024, 1536, 2048, 3072, 4096, 6144, 8192, 12288, 16384};
    const int K_list[15] = {1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024};

    const int outer_repeat = 10, inner_repeat = 1;
    const int BM = 32, BN = 32;
    void (*gpuSgemm) (float*, float*, float*, const int, const int, const int) = Sgemm_v1;
    const int TESTNUM = 15;
    for (int i = 0; i < TESTNUM; i++) 
    {
        const int M = M_list[i], N = N_list[i], K = K_list[i];
        dim3 blockDim(BM, BN);
        dim3 gridDim((M + BM -1) / BM, (N + BN -1) / BN);

        double max_sec = 0.0;
        double min_sec = DBL_MAX;
        double total_sec = 0.0;

        for (int j = 0; j < outer_repeat; j++) 
        {
            double this_sec = testPerformance(gpuSgemm, gridDim, blockDim, M, N, K, inner_repeat);
            max_sec = max(max_sec, this_sec);
            min_sec = min(min_sec, this_sec);
            total_sec += this_sec;
        }

        double avg_sec = total_sec / outer_repeat;
        double avg_GFLOPS = (double)M * N * K * 2.0 / avg_sec / 1024 / 1024 / 1024;
        printf("M N K = %6d %6d %6d, Time = %12.8lf %12.8lf %12.8lf s, AVG Performance = %10.4lf GFLOPS\n", M, N, K, min_sec, avg_sec, max_sec, avg_GFLOPS);
    }
    return 0;
}

float testError(void) {
    const int BM = 32, BN = 32;
    const int M = 1024, N = 1024, K = 1024;
    dim3 blockDim(BM, BN);
    dim3 gridDim((M + BM -1) / BM, (N + BN -1) / BN);

    size_t size_a = M * K * sizeof(float);
    size_t size_b = K * N * sizeof(float);
    size_t size_c = M * N * sizeof(float);

    float *h_a, *h_b, *h_c;
    float *d_a, *d_b, *d_c;
    float *cpu_c;
    h_a = (float *)malloc(size_a);
    h_b = (float *)malloc(size_b);
    h_c = (float *)malloc(size_c);
    cpu_c = (float *)malloc(size_c);
    cudaMalloc((void **)&d_a, size_a);
    cudaMalloc((void **)&d_b, size_b);
    cudaMalloc((void **)&d_c, size_c);

    srand(time(0));
    for (int i = 0; i < M * K; i++) {
        h_a[i] = rand() / float(RAND_MAX);
    }
    for (int i = 0; i < K * N; i++) {
        h_b[i] = rand() / float(RAND_MAX);
    }
    cudaMemset(d_c, 0, size_c);
    cpuSgemm(h_a, h_b, cpu_c, M, N, K);

    cudaMemcpy(d_a, h_a, size_a, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, size_b, cudaMemcpyHostToDevice);
    Sgemm_v1<<<gridDim, blockDim>>>(d_a, d_b, d_c, M, N, K);
    cudaDeviceSynchronize();
    cudaMemcpy(h_c, d_c, size_c, cudaMemcpyDeviceToHost);

    float max_error = 0.0;
    for (int i = 0; i < M * N; i++) {
        // printf("%f %f\n", cpu_c[i], h_c[i]);
        float error = abs(cpu_c[i] - h_c[i]);
        max_error = max(error, max_error);
    }
    free(h_a);
    free(h_b);
    free(h_c);
    free(cpu_c);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    
    return max_error;
}

float testPerformance(
    void (*gpuSgemm) (float *, float *, float *, const int, const int, const int),
    dim3 gridDim, dim3 blockDim, const int M, const int N, const int K, const int repeat
) 
{
    size_t size_a = M * K * sizeof(float);
    size_t size_b = K * N * sizeof(float);
    size_t size_c = M * N * sizeof(float);

    float *d_a, *d_b, *d_c;

    cudaMalloc((void **)&d_a, size_a);
    cudaMalloc((void **)&d_b, size_b);
    cudaMalloc((void **)&d_c, size_c);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    for (int i = 0; i < repeat; i++) {
        gpuSgemm<<<gridDim, blockDim>>>(d_a, d_b, d_c, M, N, K);
        // cudaDeviceSynchronize();
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float elapsed_time_ms, elapsed_time_s;
    cudaEventElapsedTime(&elapsed_time_ms, start, stop);
    elapsed_time_s = elapsed_time_ms / 1000.0 / repeat;
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    
    return elapsed_time_s;
}