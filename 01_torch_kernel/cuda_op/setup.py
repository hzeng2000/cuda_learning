from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name="add2",
    include_dirs=["include"],
    ext_modules=[
        CUDAExtension(
            "add2",
            ["kernel/madd_ops.cpp", "kernel/madd.cu"],
        )
    ],
    cmdclass={
        "build_ext": BuildExtension
    }
)