vector_c[i] = vector_a[i] + vector_b[i];
## vadd_v1
simplest vadd kernel, 1-dim grid, 1-dim block
every thread only process one element for vector_c

## vadd_v2
1-dim grid, 1-dim block
every thread can process more than element for vector_c(stride)
using cudaMallocManaged to manage memory(tested with 70 times performance loss)

## dim
understanding the dim of grid and block, threadIdx.x and blockIdx.x

