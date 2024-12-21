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
### 02_reduce/reduce_warp.cu
- 线程同步函数
    当所涉及的线程都在一个线程束内时，可以用线程块同步函数__syncthreads 换成一个更加廉价的线程束同步函数 __syncwarp
    因此可以当offset<=16时，使用__syncwarp
- 线程束洗牌函数
    可以用线程束洗牌函数完成规约计算，函数 __shfl_down_sync 的作用是将高线程号的数据平移到低线程号中去，这正是归约问题中需要的操作
- 线程块片洗牌函数
    协作组(cooperative groups)可以看作是线程块和线程束同步机制的推广,使用thread_group g32 = tiled_partition(this_thread_block(), 32)把一个block继续划分成tile，每个tile构成新的线程组，大小支持2,4，8,16,32,然后也有自己的洗牌函数，这里就是按32划分，并使用洗牌函数进行规约

在之前的例子中，block-size是128，当offset是64时，只使用了1/2的线程，其余线程闲置
    以此类推，当offset是1时，只使用了1/128的线程，其余线程闲置
    故规约过程中的线程利用率只有（1/2+1/4+...+1/128）/ 7 = 1/7 
    另外注意，之前的程序，都是总的线程数 >= N,
### 02_reduce/reduce_staic.cu
- 进一步优化
    1. 当N很大的时候，所有的线程都无法一次性处理完，可以先做一次stride的规约，把 x + n * (gridsize*blocksize)的树都规约到x上，这样就避免了上面说的线程利用率低的问题，也解决了N很大无法处理的问题（如果还是沿用之前的方法二分规约，将会规约更多次，而随着总线程数增加，浪费每次是总线程数折半，线程利用率更低）
    2. 把原本的最后一步规约，通过再次调用一次核函数完成，grid=1,block=1024,之前得到的y实际上长度是griddim，因此这里再次调用
        reduce_cp<<<1, 1024, sizeof(real) * 1024>>>(d_y, d_y, GRID_SIZE);