__global__ void MatAdd(float *a, float *b, float *c, const int M, const int N)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    int idx = row * N + col;
    if (row < M && col < N)
    {
        c[idx] = a[idx] + b[idx];
    }
}

void launch_MatAdd(float *a, float *b, float *c, const int M, const int N)
{
    dim3 blockDim(32, 32);
    dim3 gridDim((N + blockDim.x - 1) / blockDim.x, (M + blockDim.y - 1) / blockDim.y);

    MatAdd<<<gridDim, blockDim>>>(a, b, c, M, N);
}