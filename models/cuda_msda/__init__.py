"""Multi-Scale Deformable Attention CUDA extension.

This module provides the CUDA implementation for Multi-Scale Deformable Attention
used in Deformable DETR and similar architectures.

To rebuild the CUDA extension:
    cd gausstr_lightning/models/cuda_msda
    python setup.py build_ext --inplace
"""

import os
import sys

# Add the directory containing the .so file to the path
_cuda_msda_dir = os.path.dirname(os.path.abspath(__file__))
if _cuda_msda_dir not in sys.path:
    sys.path.insert(0, _cuda_msda_dir)

try:
    import MultiScaleDeformableAttention as MSDA
    ms_deform_attn_forward = MSDA.ms_deform_attn_forward
    ms_deform_attn_backward = MSDA.ms_deform_attn_backward
    HAS_CUDA_MSDA = True
except ImportError as e:
    MSDA = None
    ms_deform_attn_forward = None
    ms_deform_attn_backward = None
    HAS_CUDA_MSDA = False
    import warnings
    warnings.warn(
        f"Failed to import MultiScaleDeformableAttention CUDA extension: {e}\n"
        "Please rebuild with: cd gausstr_lightning/models/cuda_msda && python setup.py build_ext --inplace"
    )

__all__ = ['MSDA', 'ms_deform_attn_forward', 'ms_deform_attn_backward', 'HAS_CUDA_MSDA']
