#include <iostream>
#include <cuda_runtime.h>

struct Matrix 
{
    int width;
    int height;
    float *elements;
};

/**
 auxiliary func:
    getElement
    setElement
 kernel:
    matMul
 main
   1. initialize matrix and using cudaMallocManaged to allocate memory
   2. define block and grid
   3. launch kernel 
   4. synchronize and check error
 */

 __device__ float getElement(Matrix *m, int row, int col)
{
    return m->elements[row * m->width + col];
}

__device__ void setElement(Matrix *m, int row, int col, float value)
{
    m->elements[row * m->width + col] = value;
}

__global__ void matMul(Matrix *a, Matrix *b, Matrix *c)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < a->height && col < b->width)
    {
        float tmp = 0.0f;
        for (int k = 0; k < a->width; k++)
        {
            tmp += getElement(a, row, k) * getElement(b, k, col);
        }
        setElement(c, row, col, tmp);
    }
}

int main()
{
    int width = 1 << 10;
    int height = 1 << 10;
    Matrix* A, *B, *C;
    cudaMallocManaged((void**)&A, sizeof(Matrix));
    cudaMallocManaged((void**)&B, sizeof(Matrix));
    cudaMallocManaged((void**)&C, sizeof(Matrix));
    int nBytes = width * height * sizeof(float);
    cudaMallocManaged((void**)&A->elements, nBytes);
    cudaMallocManaged((void**)&B->elements, nBytes);
    cudaMallocManaged((void**)&C->elements, nBytes);

    A->width = width;
    A->height = height;
    B->width = width;
    B->height = height;
    C->width = width;
    C->height = height;
    for (int i = 0; i < width * height; i++)
    {
        A->elements[i] = 1.0f;
        B->elements[i] = 2.0f;
    }
    dim3 blockDim(32, 32);
    dim3 gridDim((width + blockDim.x - 1) / blockDim.x, 
            (height + blockDim.y - 1) / blockDim.y);
    matMul<<<gridDim, blockDim>>>(A, B, C);
    cudaDeviceSynchronize();
    float maxError = 0.0f;
    for (int i = 0; i < width * height; i++)
    {
        maxError = fmax(maxError, fabs(C->elements[i] - 2.0f*width));
    }
    std::cout << "Max error: " << maxError << std::endl;
    return 0;
}