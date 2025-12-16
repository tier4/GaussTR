"""Standalone Multi-Scale Deformable Attention module.

This module provides a drop-in replacement for mmcv.ops.MultiScaleDeformableAttention
without requiring mmcv. It uses the CUDA implementation from Deformable-DETR.
"""

from __future__ import absolute_import, division, print_function

import math
import warnings
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import constant_, xavier_uniform_
from torch.autograd import Function
from torch.autograd.function import once_differentiable
from torch import Tensor

# Import the compiled CUDA module from local cuda_msda package
try:
    from models.cuda_msda import MSDA, HAS_CUDA_MSDA
except ImportError:
    HAS_CUDA_MSDA = False
    MSDA = None
    warnings.warn(
        "MultiScaleDeformableAttention CUDA module not found. "
        "Please build it from models/cuda_msda/"
    )

# Global variables to store library objects and prevent garbage collection
_msda_lib = None
_msda_impl_lib = None

# Register custom ops for torch.compile compatibility
if HAS_CUDA_MSDA:
    # Define the library for our custom ops (stored globally to prevent GC)
    _msda_lib = torch.library.Library("msda", "DEF")

    # Define op schemas
    _msda_lib.define(
        "ms_deform_attn_forward(Tensor value, Tensor spatial_shapes, Tensor level_start_index, "
        "Tensor sampling_locations, Tensor attention_weights, int im2col_step) -> Tensor"
    )
    _msda_lib.define(
        "ms_deform_attn_backward(Tensor value, Tensor spatial_shapes, Tensor level_start_index, "
        "Tensor sampling_locations, Tensor attention_weights, Tensor grad_output, int im2col_step) "
        "-> (Tensor, Tensor, Tensor)"
    )

    # Create implementation library (stored globally to prevent GC)
    _msda_impl_lib = torch.library.Library("msda", "IMPL")

    # Register CUDA implementations
    def _ms_deform_attn_forward_cuda(
        value: Tensor,
        spatial_shapes: Tensor,
        level_start_index: Tensor,
        sampling_locations: Tensor,
        attention_weights: Tensor,
        im2col_step: int
    ) -> Tensor:
        return MSDA.ms_deform_attn_forward(
            value, spatial_shapes, level_start_index,
            sampling_locations, attention_weights, im2col_step
        )

    def _ms_deform_attn_backward_cuda(
        value: Tensor,
        spatial_shapes: Tensor,
        level_start_index: Tensor,
        sampling_locations: Tensor,
        attention_weights: Tensor,
        grad_output: Tensor,
        im2col_step: int
    ) -> Tuple[Tensor, Tensor, Tensor]:
        return tuple(MSDA.ms_deform_attn_backward(
            value, spatial_shapes, level_start_index,
            sampling_locations, attention_weights, grad_output, im2col_step
        ))

    _msda_impl_lib.impl("ms_deform_attn_forward", _ms_deform_attn_forward_cuda, "CUDA")
    _msda_impl_lib.impl("ms_deform_attn_backward", _ms_deform_attn_backward_cuda, "CUDA")

    # Register abstract (fake tensor) implementations for torch.compile tracing
    @torch.library.register_fake("msda::ms_deform_attn_forward")
    def _ms_deform_attn_forward_fake(
        value: Tensor,
        spatial_shapes: Tensor,
        level_start_index: Tensor,
        sampling_locations: Tensor,
        attention_weights: Tensor,
        im2col_step: int
    ) -> Tensor:
        # value: [batch, spatial_size, num_heads, channels]
        # sampling_locations: [batch, num_query, num_heads, num_levels, num_points, 2]
        # output: [batch, num_query, num_heads * channels]
        batch = value.shape[0]
        num_heads = value.shape[2]
        channels = value.shape[3]
        num_query = sampling_locations.shape[1]
        return value.new_empty(batch, num_query, num_heads * channels)

    @torch.library.register_fake("msda::ms_deform_attn_backward")
    def _ms_deform_attn_backward_fake(
        value: Tensor,
        spatial_shapes: Tensor,
        level_start_index: Tensor,
        sampling_locations: Tensor,
        attention_weights: Tensor,
        grad_output: Tensor,
        im2col_step: int
    ) -> Tuple[Tensor, Tensor, Tensor]:
        grad_value = torch.empty_like(value)
        grad_sampling_loc = torch.empty_like(sampling_locations)
        grad_attn_weight = torch.empty_like(attention_weights)
        return grad_value, grad_sampling_loc, grad_attn_weight


def _is_power_of_2(n: int) -> bool:
    """Check if n is a power of 2."""
    if (not isinstance(n, int)) or (n < 0):
        raise ValueError(f"invalid input for _is_power_of_2: {n} (type: {type(n)})")
    return (n & (n - 1) == 0) and n != 0


class MSDeformAttnFunction(Function):
    """Autograd function for multi-scale deformable attention.

    Uses registered custom ops (msda::ms_deform_attn_forward/backward) for
    torch.compile compatibility.
    """

    @staticmethod
    def forward(ctx, value, value_spatial_shapes, value_level_start_index,
                sampling_locations, attention_weights, im2col_step):
        ctx.im2col_step = im2col_step

        # CUDA kernel only supports FP32 - cast inputs and restore dtype after
        input_dtype = value.dtype
        ctx.input_dtype = input_dtype
        if input_dtype != torch.float32:
            value = value.float()
            sampling_locations = sampling_locations.float()
            attention_weights = attention_weights.float()

        output = torch.ops.msda.ms_deform_attn_forward(
            value, value_spatial_shapes, value_level_start_index,
            sampling_locations, attention_weights, im2col_step)
        ctx.save_for_backward(value, value_spatial_shapes, value_level_start_index,
                              sampling_locations, attention_weights)

        # Cast output back to original dtype
        if input_dtype != torch.float32:
            output = output.to(input_dtype)

        return output

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        value, value_spatial_shapes, value_level_start_index, \
            sampling_locations, attention_weights = ctx.saved_tensors

        # CUDA kernel only supports FP32 - cast grad_output to match saved tensors
        input_dtype = ctx.input_dtype
        if input_dtype != torch.float32:
            grad_output = grad_output.float()

        grad_value, grad_sampling_loc, grad_attn_weight = \
            torch.ops.msda.ms_deform_attn_backward(
                value, value_spatial_shapes, value_level_start_index,
                sampling_locations, attention_weights, grad_output, ctx.im2col_step)

        # Cast gradients back to original dtype
        if input_dtype != torch.float32:
            grad_value = grad_value.to(input_dtype)
            grad_sampling_loc = grad_sampling_loc.to(input_dtype)
            grad_attn_weight = grad_attn_weight.to(input_dtype)

        return grad_value, None, None, grad_sampling_loc, grad_attn_weight, None


def ms_deform_attn_core_pytorch(value, value_spatial_shapes, sampling_locations, attention_weights):
    """Pure PyTorch implementation for debugging/testing."""
    N_, S_, M_, D_ = value.shape
    _, Lq_, M_, L_, P_, _ = sampling_locations.shape
    value_list = value.split([H_ * W_ for H_, W_ in value_spatial_shapes], dim=1)
    sampling_grids = 2 * sampling_locations - 1
    sampling_value_list = []
    for lid_, (H_, W_) in enumerate(value_spatial_shapes):
        value_l_ = value_list[lid_].flatten(2).transpose(1, 2).reshape(N_ * M_, D_, H_, W_)
        sampling_grid_l_ = sampling_grids[:, :, :, lid_].transpose(1, 2).flatten(0, 1)
        sampling_value_l_ = F.grid_sample(
            value_l_, sampling_grid_l_,
            mode='bilinear', padding_mode='zeros', align_corners=False)
        sampling_value_list.append(sampling_value_l_)
    attention_weights = attention_weights.transpose(1, 2).reshape(N_ * M_, 1, Lq_, L_ * P_)
    output = (torch.stack(sampling_value_list, dim=-2).flatten(-2) * attention_weights).sum(-1).view(N_, M_ * D_, Lq_)
    return output.transpose(1, 2).contiguous()


class MultiScaleDeformableAttention(nn.Module):
    """Multi-Scale Deformable Attention Module.

    This is a drop-in replacement for mmcv.ops.MultiScaleDeformableAttention.

    Args:
        embed_dims: Embedding dimension (default: 256).
        num_heads: Number of attention heads (default: 8).
        num_levels: Number of feature levels (default: 4).
        num_points: Number of sampling points per head per level (default: 4).
        im2col_step: Step size for im2col (default: 64).
        dropout: Dropout probability (default: 0.0).
        batch_first: Whether input is batch first (default: True).
    """

    def __init__(
        self,
        embed_dims: int = 256,
        num_heads: int = 8,
        num_levels: int = 4,
        num_points: int = 4,
        im2col_step: int = 64,
        dropout: float = 0.0,
        batch_first: bool = True,
        **kwargs  # Accept extra args for compatibility
    ):
        super().__init__()
        if embed_dims % num_heads != 0:
            raise ValueError(
                f'embed_dims must be divisible by num_heads, '
                f'but got {embed_dims} and {num_heads}')

        dim_per_head = embed_dims // num_heads
        if not _is_power_of_2(dim_per_head):
            warnings.warn(
                "You'd better set embed_dims in MultiScaleDeformableAttention "
                "to make the dimension of each attention head a power of 2 "
                "which is more efficient in our CUDA implementation.")

        self.im2col_step = im2col_step
        self.embed_dims = embed_dims
        self.num_levels = num_levels
        self.num_heads = num_heads
        self.num_points = num_points
        self.batch_first = batch_first
        self.dropout = nn.Dropout(dropout)

        # Projection layers
        self.sampling_offsets = nn.Linear(
            embed_dims, num_heads * num_levels * num_points * 2)
        self.attention_weights = nn.Linear(
            embed_dims, num_heads * num_levels * num_points)
        self.value_proj = nn.Linear(embed_dims, embed_dims)
        self.output_proj = nn.Linear(embed_dims, embed_dims)

        self._reset_parameters()

    def _reset_parameters(self):
        """Initialize parameters."""
        constant_(self.sampling_offsets.weight.data, 0.)
        thetas = torch.arange(
            self.num_heads, dtype=torch.float32) * (2.0 * math.pi / self.num_heads)
        grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)
        grid_init = (grid_init / grid_init.abs().max(-1, keepdim=True)[0]).view(
            self.num_heads, 1, 1, 2).repeat(1, self.num_levels, self.num_points, 1)
        for i in range(self.num_points):
            grid_init[:, :, i, :] *= i + 1
        with torch.no_grad():
            self.sampling_offsets.bias = nn.Parameter(grid_init.view(-1))
        constant_(self.attention_weights.weight.data, 0.)
        constant_(self.attention_weights.bias.data, 0.)
        xavier_uniform_(self.value_proj.weight.data)
        constant_(self.value_proj.bias.data, 0.)
        xavier_uniform_(self.output_proj.weight.data)
        constant_(self.output_proj.bias.data, 0.)

    def forward(
        self,
        query: torch.Tensor,
        key: Optional[torch.Tensor] = None,
        value: Optional[torch.Tensor] = None,
        identity: Optional[torch.Tensor] = None,
        query_pos: Optional[torch.Tensor] = None,
        key_pos: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
        reference_points: Optional[torch.Tensor] = None,
        spatial_shapes: Optional[torch.Tensor] = None,
        level_start_index: Optional[torch.Tensor] = None,
        **kwargs
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            query: Query tensor [B, num_queries, embed_dims].
            key: Key tensor (unused, for compatibility).
            value: Value tensor [B, num_value, embed_dims]. If None, uses key.
            identity: Identity tensor for residual connection.
            query_pos: Positional encoding for queries.
            key_pos: Positional encoding for keys (unused).
            key_padding_mask: Padding mask [B, num_value].
            reference_points: Reference points [B, num_queries, num_levels, 2].
            spatial_shapes: Spatial shapes of each level [num_levels, 2].
            level_start_index: Start index for each level [num_levels].

        Returns:
            Output tensor [B, num_queries, embed_dims].
        """
        if value is None:
            value = query

        if identity is None:
            identity = query

        if query_pos is not None:
            query = query + query_pos

        bs, num_query, _ = query.shape
        bs, num_value, _ = value.shape

        assert (spatial_shapes[:, 0] * spatial_shapes[:, 1]).sum() == num_value

        # Project value
        value = self.value_proj(value)
        if key_padding_mask is not None:
            value = value.masked_fill(key_padding_mask[..., None], 0.0)
        value = value.view(bs, num_value, self.num_heads, -1)

        # Compute sampling offsets and attention weights
        sampling_offsets = self.sampling_offsets(query).view(
            bs, num_query, self.num_heads, self.num_levels, self.num_points, 2)
        attention_weights = self.attention_weights(query).view(
            bs, num_query, self.num_heads, self.num_levels * self.num_points)
        attention_weights = attention_weights.softmax(-1).view(
            bs, num_query, self.num_heads, self.num_levels, self.num_points)

        # Compute sampling locations
        if reference_points.shape[-1] == 2:
            offset_normalizer = torch.stack(
                [spatial_shapes[..., 1], spatial_shapes[..., 0]], -1)
            sampling_locations = reference_points[:, :, None, :, None, :] \
                + sampling_offsets / offset_normalizer[None, None, None, :, None, :]
        elif reference_points.shape[-1] == 4:
            sampling_locations = reference_points[:, :, None, :, None, :2] \
                + sampling_offsets / self.num_points \
                * reference_points[:, :, None, :, None, 2:] * 0.5
        else:
            raise ValueError(
                f'Last dim of reference_points must be 2 or 4, '
                f'but got {reference_points.shape[-1]}.')

        # Apply deformable attention
        if HAS_CUDA_MSDA and value.is_cuda:
            output = MSDeformAttnFunction.apply(
                value, spatial_shapes, level_start_index,
                sampling_locations, attention_weights, self.im2col_step)
        else:
            # Fallback to PyTorch implementation
            output = ms_deform_attn_core_pytorch(
                value, spatial_shapes, sampling_locations, attention_weights)

        output = self.output_proj(output)
        return self.dropout(output) + identity
