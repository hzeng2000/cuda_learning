# 指定最低版本要求
cmake_minimum_required(VERSION 3.1)

# 项目名称和版本
project(add2 LANGUAGES C CXX CUDA)

# 设置C++标准为C++17
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 设置CUDA标准为CUDA 14（你也可以设置为其他版本）
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# export Torch_DIR=/data/hzeng/miniconda3/envs/py310_pt24/lib/python3.10/site-packages/torch
set(CMAKE_PREFIX_PATH "/data/hzeng/miniconda3/envs/py310_pt24/lib/python3.10/site-packages/torch/share/cmake")

# 指定输出的共享库文件名
set(LIBRARY_NAME add2)

find_package(Python3 REQUIRED)
include_directories(${Python3_INCLUDE_DIRS})

# 如果仍然找不到 Python.h，可以手动指定 Python 头文件路径
include_directories(/data/hzeng/miniconda3/envs/py310_pt24/include/python3.10)

# 确定PyTorch的路径
find_package(Torch REQUIRED)

# 确保CUDA可用
find_package(CUDA REQUIRED)

# 添加包含目录
include_directories(${CMAKE_SOURCE_DIR}/include)

# 添加源文件
set(SOURCES
    kernel/madd_ops.cpp
    kernel/madd.cu
)

# 设置编译选项（适配NVCC和C++编译器）
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler -fPIC")
link_directories(/home/spack/opt/spack/linux-ubuntu20.04-cascadelake/gcc-9.4.0/cuda-12.1.1-othej3jzw4zw6czpdimdseejzvr4nxws/lib64)

# 定义共享库
add_library(${LIBRARY_NAME} SHARED ${SOURCES})

# 链接Torch和CUDA的依赖库
target_link_libraries(${LIBRARY_NAME} PRIVATE ${TORCH_LIBRARIES} CUDA::cudart)

# 设置 pybind11 配置路径
set(pybind11_DIR "/data/hzeng/miniconda3/envs/py310_pt24/lib/python3.10/site-packages/pybind11/share/cmake/pybind11")

# 查找 pybind11 包
find_package(pybind11 REQUIRED)
include_directories(${pybind11_INCLUDE_DIRS})

# 链接 pybind11 库
target_link_libraries(${LIBRARY_NAME} PRIVATE pybind11::module)

# 设置生成文件输出路径
set_target_properties(${LIBRARY_NAME}
    PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib
)

# 打印提示信息
message(STATUS "Torch library directories: ${TORCH_LIBRARIES}")
message(STATUS "CUDA support enabled.")