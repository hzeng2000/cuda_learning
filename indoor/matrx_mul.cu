#include <iostream>
#include <cuda_runtime.h>


struct Matrix
{
    int width;
    int height;
    float* elements;
};

// 辅助函数，获取矩阵对应位置元素值
__device__ float getElement(const Matrix* A, int row, int col)
{
    return A->elements[row * A->width + col];
}
// 辅助函数，set矩阵对应位置元素值
__device__ float setElement(const Matrix* A, int row, int col, float value)
{
    A->elements[row * A->width + col] = value;
}

// 矩阵乘法kernel
__global__ void matMulKernel(const Matrix* A, const Matrix* B, Matrix* C)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < C->height && col < C->width)
    {
        float value = 0;
        for (int i = 0; i < A->width; ++i)
        {
            value += getElement(A, row, i) * getElement(B, i, col);
        }
        setElement(C, row, col, value);
    }
}

int main()
{
    int width = 1 << 10;
    int height = 1 << 10;
    Matrix* A, * B, * C;
    // 申请托管内存
    cudaMallocManaged((void**)&A, sizeof(Matrix));
    cudaMallocManaged((void**)&B, sizeof(Matrix));
    cudaMallocManaged((void**)&C, sizeof(Matrix));
    int nBytes = width * height * sizeof(float);
    cudaMallocManaged((void**)&A->elements, nBytes);
    cudaMallocManaged((void**)&B->elements, nBytes);
    cudaMallocManaged((void**)&C->elements, nBytes);

    // 初始化数据
    A->height = height;
    A->width = width;
    B->height = height;
    B->width = width;
    C->height = height;
    C->width = width;
    for (int i = 0; i < width * height; ++i)
    {
        A->elements[i] = 1.0;
        B->elements[i] = 2.0;
    }

    // 定义kernel的执行配置
    dim3 blockSize(32, 32);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x, 
        (height + blockSize.y - 1) / blockSize.y);
    // 执行kernel
    matMulKernel << < gridSize, blockSize >> >(A, B, C);


    // 同步device 保证结果能正确访问
    cudaDeviceSynchronize();
    // 检查执行结果
    float maxError = 0.0;
    for (int i = 0; i < width * height; ++i)
        maxError = fmax(maxError, fabs(C->elements[i] - 2 * width));
    std::cout << "最大误差: " << maxError << std::endl;

    return 0;
}