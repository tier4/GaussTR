"""Setup script for CUDA voxelizer extension."""

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='voxelize_cuda_ext',
    ext_modules=[
        CUDAExtension(
            name='voxelize_cuda_ext',
            sources=['voxelize_cuda.cu'],
            extra_compile_args={
                'cxx': ['-O3'],
                'nvcc': [
                    '-O3',
                    '--use_fast_math',
                    '-gencode=arch=compute_70,code=sm_70',  # V100
                    '-gencode=arch=compute_75,code=sm_75',  # T4
                    '-gencode=arch=compute_80,code=sm_80',  # A100
                    '-gencode=arch=compute_86,code=sm_86',  # RTX 3090
                    '-gencode=arch=compute_89,code=sm_89',  # RTX 4090
                    '-gencode=arch=compute_90,code=sm_90',  # H100
                ]
            }
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)
