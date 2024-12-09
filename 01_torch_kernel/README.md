# cuda_learning

## kernel
cuda_op里面定义了cuda kernel madd.cu
以及对应的wrapper madd_ops.cpp，通过pytorch c++注册算子调用
## run
run_time.py里面可以选择三种编译方式

1. jit
直接运行torch_kernel/cuda_op/run_time.py，即时编译
```bash
python3 run_time.py --compiler jit
```
2. setup编译
```bash
# 编译生成动态链接库
python setup.py install
# 运行
python3 run_time.py --compiler setup
```

3. cmake编译
目前运行的时候有OSError: /data/hzeng/prj/cuda_learning/torch_kernel/cuda_op/build/lib/libadd2.so: undefined symbol: _ZN8pybind116detail11type_casterIN2at6TensorEvE4loadENS_6handleEb错误
```bash
mkdir build
cd build
cmake ..
make

python3 run_time.py --compiler cmake
```