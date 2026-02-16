from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='pointops',
    ext_modules=[
        CUDAExtension(
            name='_C',
            sources=[
                'src/pointops_api.cpp',
                'src/sampling_cuda.cpp',
                'src/sampling_cuda_kernel.cu',
            ],
            extra_compile_args={
                'cxx': ['-O3'],
                'nvcc': ['-O3'],
            },
        ),
    ],
    cmdclass={'build_ext': BuildExtension},
)
