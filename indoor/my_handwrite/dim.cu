#include <iostream>
#include <cuda_runtime.h>

// CUDA kernel, 进行矩阵加法的操作
__global__ void matrixAdd(int *A, int *B, int *C, int width, int height, int constant) {
    // 计算每个线程处理的矩阵元素的行列索引
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    // 确保线程索引在矩阵范围内
    if (row < height && col < width) {
        int idx = row * width + col;
        C[idx] = A[idx] + B[idx] + constant; // 执行矩阵加法
    }
}

int main() {
    const int width = 16;  // 矩阵的宽度
    const int height = 16; // 矩阵的高度
    const int matrixSize = width * height * sizeof(int);

    int *h_A = new int[width * height];
    int *h_B = new int[width * height];
    int *h_C = new int[width * height];

    // 初始化矩阵 A 和 B
    for (int i = 0; i < width * height; ++i) {
        h_A[i] = i;
        h_B[i] = i * 2;
    }

    int *d_A, *d_B, *d_C;
    cudaMalloc((void**)&d_A, matrixSize);
    cudaMalloc((void**)&d_B, matrixSize);
    cudaMalloc((void**)&d_C, matrixSize);

    cudaMemcpy(d_A, h_A, matrixSize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, matrixSize, cudaMemcpyHostToDevice);

    // 定义 grid 和 block 的维度
    dim3 threadsPerBlock(4, 4);   // 每个block有4x4=16个线程
    dim3 numBlocks((width + 3) / 4, (height + 3) / 4); // ceil(width/4), ceil(height/4)

    // 调用kernel
    matrixAdd<<<numBlocks, threadsPerBlock>>>(d_A, d_B, d_C, width, height, 10);

    // 从设备拷贝结果到主机
    cudaMemcpy(h_C, d_C, matrixSize, cudaMemcpyDeviceToHost);

    // 打印结果矩阵 C
    for (int i = 0; i < height; ++i) {
        for (int j = 0; j < width; ++j) {
            std::cout << h_C[i * width + j] << " ";
        }
        std::cout << std::endl;
    }

    // 清理内存
    delete[] h_A;
    delete[] h_B;
    delete[] h_C;
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;
}
