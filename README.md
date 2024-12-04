# cuda_learning

### 介绍
cuda 学习
#### 编译

``` bash
nvcc -arch=sm_80 -O2 -o example.out example.cu
# for 3090 sm_86
# for 2080 sm_75
```

如果你的 CUDA 程序依赖于其他库（如 cuDNN、cuBLAS 等），你需要在编译时链接这些库。例如：

``` bash
nvcc -o example example.cu -lcudnn -lcublas
```
#### profiler
``` bash
nsys profile naive.out
```