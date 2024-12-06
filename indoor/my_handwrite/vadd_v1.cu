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
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < N) c[i] = a[i] + b[i];
}

/**
 * Test the basic functionality of the vector add kernel
 1. Allocate memory on the host
 2. initialize the data
 3. Allocate memory on the device
 4. Copy data from host to device
 5. define the grid and block size
 6. Launch the kernel
 7. synchronize
 8. Copy data from device to host
 9. check results and free the memory
 */
void test_add_basic() {
    // 1. Allocate memory on the host
    //  create vectors a,b,c with size 2^10
    int N = 1 << 10;
    std::cout << "Vector size: " << N << std::endl;
    int nBytes = N * sizeof(float);
    float *a, *b, *c;
    a = (float *) malloc(nBytes);
    b = (float *) malloc(nBytes);
    c = (float *) malloc(nBytes);

    // 2. initialize the data
    for (int i = 0; i < N; i++) {
        a[i] = 1.0f;
        b[i] = 2.0f;
    }

    // 3. Allocate memory on the device
    float *d_a, *d_b, *d_c;
    cudaMalloc((void **) &d_a, nBytes);
    cudaMalloc((void **) &d_b, nBytes);
    cudaMalloc((void **) &d_c, nBytes);

    // 4. Copy data from host to device
    cudaMemcpy(d_a, a, nBytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, nBytes, cudaMemcpyHostToDevice);

    // 5. define the grid and block size
    //  1 block with 128 threads
    //  ensure that all elements are processed
    //  e.g. N = 1281, block size = 128, then grid size = 11(10 + 1) to cover all elements
    dim3 blocksize(128);
    dim3 girdSize((N + blocksize.x -1) / blocksize.x);

    // 6. Launch the kernel
    vectorAdd<<<girdSize, blocksize>>>(d_a, d_b, d_c, N);

    // 7. synchronize
    cudaDeviceSynchronize();

    // 8. Copy data from device to host
    cudaMemcpy(c, d_c, nBytes, cudaMemcpyDeviceToHost);

    // 9. check results and free the memory
    float maxError = 0.0f;
    for (int i = 0; i < N; i++) {
        maxError = max(maxError, fabs(c[i] - 3.0f));
    }
    std::cout << "Max error: " << maxError << std::endl;

    // free the memory on the device and host
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    free(a);
    free(b);
    free(c);
    return;
}

int main() {
    test_add_basic();
    return 0;
}