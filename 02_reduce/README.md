# cuda_learning

## reduce相关操作
### 02_reduce/reduce_cpu.cu
cpu实现reduce
### 02_reduce/reduce_gpu.cu
利用折半规约法(具体见代码注释), 三种方法实现规约
- 使用全局内存
- 使用共享内存
- 使用动态共享内存
### 02_reduce/reduce_atomic.cu
对上述折半规约的改进, 上述方法核函数功能是把一个长度为N的数组规约到(n/blocksize)的数组,在cpu完成剩余的规约操作
但是明显会带来性能损失,把整个规约操作都放到gpu上能提高性能
简单的想法就是在kernel最后再做一次规约,得到把各个idx的结果规约到d_y[0]
即dy[0]+=s_y[0]，但是这样会遇到多线程冲突问题，reduce_atomic.cu通过原子操作解决