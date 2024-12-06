#include <iostream>
#include <cuda_runtime.h>

/**
 * Vector Addition Kernel 
 * description: every thread computes 1 elem c[i] = a[i] + b[i]
 * @param a, b, c: input and output vectors
 * @param N: size of the vectors
 */
__global__ void vectorAdd(float *a, float *b, float *c, const int N) 
{
    int index = blockDim.x * blockIdx.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = index; i < N; i += stride) 
    {
        c[i] = a[i] + b[i];
    }
}

/**
 * Test the basic functionality of the vector add kernel
 with unified memory, we can simplify the code
 1. 
 2. initialize the data
 3. 
 4. 
 5. define the grid and block size
 6. Launch the kernel
 7. synchronize
 8. 
 9. check results and free the memory(explicitly or program will do it automatically)
 */
void test_add_basic() {
    // 1. Allocate memory on the host
    //  create vectors a,b,c with size 2^10
    int N = 1 << 20;
    std::cout << "Vector size: " << N << std::endl;
    int nBytes = N * sizeof(float);
    float *a, *b, *c;
    cudaMallocManaged((void**)&a, nBytes);
    cudaMallocManaged((void**)&b, nBytes);
    cudaMallocManaged((void**)&c, nBytes);

    // 2. initialize the data
    for (int i = 0; i < N; i++) {
        a[i] = 1.0f;
        b[i] = 2.0f;
    }

    // 5. define the grid and block size
    //  1 block with 128 threads
    //  1 grid with 2 block
    //  one thread computes many element
    dim3 blocksize(128);
    dim3 girdSize(2);

    // 6. Launch the kernel
    vectorAdd<<<girdSize, blocksize>>>(a, b, c, N);

    // 7. synchronize
    cudaDeviceSynchronize();

    // 9. check results and free the memory
    float maxError = 0.0f;
    for (int i = 0; i < N; i++) {
        maxError = max(maxError, fabs(c[i] - 3.0f));
    }
    std::cout << "Max error: " << maxError << std::endl;

    // free the memory on the device and host
    cudaFree(a);
    cudaFree(b);
    cudaFree(c);
    return;
}

int main() {
    test_add_basic();
    return 0;
}