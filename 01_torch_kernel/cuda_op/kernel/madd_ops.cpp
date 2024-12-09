#include <torch/extension.h>
#include "madd_ops.h"

void torch_launch_madd_kernel(
    torch::Tensor &a,
    torch::Tensor &b,
    torch::Tensor &c,
    int64_t m,
    int64_t n) 
{
    launch_MatAdd((float*)a.data_ptr(),
                 (float*)b.data_ptr(),
                 (float*)c.data_ptr(),
                 m, n);  
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("torch_launch_madd", &torch_launch_madd_kernel, "MatAdd kernel");
}