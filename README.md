# cuda_learning

### 介绍
cuda 学习
参考：
1. 入门：
- https://zhuanlan.zhihu.com/p/34587739 （对应indoor里面的vadd_v1, vadd_v2, mm_v1）
- https://zhuanlan.zhihu.com/p/645330027 (对应torch_kernel里面部分内容， 包括完整的定义cuda kernel并add进pytorch算子)
2. reduce优化
- https://zhuanlan.zhihu.com/p/654027980 （对应02_reduce内容， 待完整）
- https://github.com/ifromeast/cuda_learning/blob/main/02_reduce/reduce_gpu.cu （02_reduce部分源码， 待完整）
3. GEMM优化
- https://zhuanlan.zhihu.com/p/703256080 (对应gemm里面部分内容，待完整)
- https://zhuanlan.zhihu.com/p/654027980 (GPU的内存体系及其优化指南，待完整)
- https://zhuanlan.zhihu.com/p/657632577 (通用矩阵乘法：从入门到熟练，待完整)
- https://chiemon.github.io/2020/02/06/CUDA-%E7%9F%A9%E9%98%B5%E4%B9%98%E6%B3%95-%E4%BC%98%E5%8C%96%E5%8F%8A%E6%80%A7%E8%83%BD%E5%88%86%E6%9E%90-%E4%B8%8A.html

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

## profile
```bash
# profile and generate stats
nsys profile --stats=true vadd_v1.out
# check kernel summary and memop summary
nsys stats --report cuda_gpu_kern_sum --report cuda_gpu_mem_time_sum report3.nsys-rep

or
# need to unlock nvidia counter
ncu --set full my_program
```