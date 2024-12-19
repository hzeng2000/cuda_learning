# cuda_learning

### 介绍
cuda 学习
参考：
1. 入门：
- https://zhuanlan.zhihu.com/p/34587739 （对应indoor里面的vadd_v1, vadd_v2, mm_v1）
- https://zhuanlan.zhihu.com/p/645330027 (对应torch_kernel里面部分内容， 包括完整的定义cuda kernel并add进pytorch算子)
2. GEMM优化
- https://zhuanlan.zhihu.com/p/703256080 (对应gemm里面部分内容，待完整)
- https://zhuanlan.zhihu.com/p/654027980 (GPU的内存体系及其优化指南，待完整)
- https://zhuanlan.zhihu.com/p/657632577 (通用矩阵乘法：从入门到熟练，待完整)
3. reduce优化
- https://zhuanlan.zhihu.com/p/654027980 （对应02_reduce内容， 待完整）
- https://github.com/ifromeast/cuda_learning/blob/main/02_reduce/reduce_gpu.cu （02_reduce部分源码， 待完整）

## profile
```bash
# profile and generate stats
nsys profile --stats=true vadd_v1.out
# check kernel summary and memop summary
nsys stats --report cuda_gpu_kern_sum --report cuda_gpu_mem_time_sum report3.nsys-rep
```