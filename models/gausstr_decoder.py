"""GaussTR Decoder implementation.

This decoder uses deformable attention for efficient multi-scale feature aggregation.
Note: MultiScaleDeformableAttention is kept from mmcv.ops as it's a highly optimized CUDA op.
"""

import math
from typing import Optional, List, Dict, Any, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# Use standalone deformable attention (no mmcv dependency)
try:
    from .deformable_attn import MultiScaleDeformableAttention, HAS_CUDA_MSDA
    HAS_MMCV_OPS = HAS_CUDA_MSDA
except ImportError:
    HAS_MMCV_OPS = False
    print("Warning: Standalone MultiScaleDeformableAttention not available. "
          "Using slower PyTorch fallback.")


def coordinate_to_encoding(
    coord_tensor: torch.Tensor,
    num_feats: int = 128,
    temperature: int = 10000,
    scale: float = 2 * math.pi,
    dim_t: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """Convert coordinate tensor to positional encoding.

    Args:
        coord_tensor: Coordinate tensor of shape [..., 2] or [..., 4].
        num_feats: Feature dimension for each position along x/y axis.
            Final dimension is 2x or 4x this value.
        temperature: Temperature for scaling position embedding.
        scale: Scale factor for position embedding.
        dim_t: Pre-computed temperature scaling tensor. If None, computed on the fly.

    Returns:
        Positional encoding tensor.
    """
    if dim_t is None:
        dim_t = torch.arange(
            num_feats, dtype=torch.float32, device=coord_tensor.device)
        dim_t = temperature**(2 * (dim_t // 2) / num_feats)

    x_embed = coord_tensor[..., 0] * scale
    y_embed = coord_tensor[..., 1] * scale
    pos_x = x_embed[..., None] / dim_t
    pos_y = y_embed[..., None] / dim_t
    pos_x = torch.stack((pos_x[..., 0::2].sin(), pos_x[..., 1::2].cos()),
                        dim=-1).flatten(2)
    pos_y = torch.stack((pos_y[..., 0::2].sin(), pos_y[..., 1::2].cos()),
                        dim=-1).flatten(2)

    if coord_tensor.size(-1) == 2:
        pos = torch.cat((pos_y, pos_x), dim=-1)
    elif coord_tensor.size(-1) == 4:
        w_embed = coord_tensor[..., 2] * scale
        pos_w = w_embed[..., None] / dim_t
        pos_w = torch.stack((pos_w[..., 0::2].sin(), pos_w[..., 1::2].cos()),
                            dim=-1).flatten(2)

        h_embed = coord_tensor[..., 3] * scale
        pos_h = h_embed[..., None] / dim_t
        pos_h = torch.stack((pos_h[..., 0::2].sin(), pos_h[..., 1::2].cos()),
                            dim=-1).flatten(2)

        pos = torch.cat((pos_y, pos_x, pos_w, pos_h), dim=-1)
    else:
        raise ValueError(f'Unknown pos_tensor shape(-1): {coord_tensor.size(-1)}')
    return pos


def inverse_sigmoid(x: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    """Inverse function of sigmoid.

    Args:
        x: Input tensor.
        eps: Small value for numerical stability.

    Returns:
        Inverse sigmoid of x.
    """
    x = x.clamp(min=0, max=1)
    x1 = x.clamp(min=eps)
    x2 = (1 - x).clamp(min=eps)
    return torch.log(x1 / x2)


class MLP(nn.Module):
    """Simple MLP for reference point head."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int
    ):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


class FFN(nn.Module):
    """Feed-Forward Network used in transformer.

    Args:
        embed_dims: Embedding dimensions.
        feedforward_channels: Hidden layer dimensions.
        num_fcs: Number of fully connected layers.
        ffn_drop: Dropout probability.
    """

    def __init__(
        self,
        embed_dims: int = 256,
        feedforward_channels: int = 2048,
        num_fcs: int = 2,
        ffn_drop: float = 0.0
    ):
        super().__init__()
        self.embed_dims = embed_dims
        self.feedforward_channels = feedforward_channels
        self.num_fcs = num_fcs

        # Structure to match original checkpoint:
        # layers[0] = Sequential(Linear, ReLU, Dropout)  -> layers.0.0 is Linear
        # layers[1] = Linear (output projection)         -> layers.1 is Linear
        # Dropout at end has no weights
        layers = nn.ModuleList()
        in_channels = embed_dims
        for _ in range(num_fcs - 1):
            layers.append(nn.Sequential(
                nn.Linear(in_channels, feedforward_channels),
                nn.ReLU(inplace=True),
                nn.Dropout(ffn_drop)
            ))
            in_channels = feedforward_channels
        layers.append(nn.Linear(feedforward_channels, embed_dims))
        layers.append(nn.Dropout(ffn_drop))
        self.layers = layers

    def forward(self, x: torch.Tensor, identity: Optional[torch.Tensor] = None) -> torch.Tensor:
        out = x
        for layer in self.layers:
            out = layer(out)
        if identity is None:
            identity = x
        return identity + out


class MultiheadAttention(nn.Module):
    """Multi-head Attention module.

    Args:
        embed_dims: Embedding dimensions.
        num_heads: Number of attention heads.
        dropout: Dropout probability.
    """

    def __init__(
        self,
        embed_dims: int = 256,
        num_heads: int = 8,
        dropout: float = 0.0,
        **kwargs  # Accept extra args for compatibility
    ):
        super().__init__()
        self.embed_dims = embed_dims
        self.num_heads = num_heads

        self.attn = nn.MultiheadAttention(
            embed_dims,
            num_heads,
            dropout=dropout,
            batch_first=True
        )

    def forward(
        self,
        query: torch.Tensor,
        key: Optional[torch.Tensor] = None,
        value: Optional[torch.Tensor] = None,
        query_pos: Optional[torch.Tensor] = None,
        key_pos: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None,
        **kwargs
    ) -> torch.Tensor:
        if key is None:
            key = query
        if value is None:
            value = key
        if query_pos is not None:
            query = query + query_pos
        if key_pos is not None:
            key = key + key_pos

        out, _ = self.attn(query, key, value, attn_mask=attn_mask)
        return out


class GaussTRDecoderLayer(nn.Module):
    """Single decoder layer for GaussTR.

    Uses cross-attention with deformable attention followed by self-attention and FFN.

    Args:
        self_attn_cfg: Config for self-attention.
        cross_attn_cfg: Config for cross-attention (deformable attention).
        ffn_cfg: Config for feed-forward network.
        norm_cfg: Config for layer normalization.
    """

    def __init__(
        self,
        self_attn_cfg: Dict[str, Any],
        cross_attn_cfg: Dict[str, Any],
        ffn_cfg: Dict[str, Any],
        norm_cfg: Optional[Dict[str, Any]] = None
    ):
        super().__init__()

        # Self attention
        self.self_attn = MultiheadAttention(**self_attn_cfg)
        self.embed_dims = self_attn_cfg.get('embed_dims', 256)

        # Cross attention (deformable)
        if HAS_MMCV_OPS:
            self.cross_attn = MultiScaleDeformableAttention(**cross_attn_cfg)
            self.use_deformable_attn = True
        else:
            # Fallback: simple attention without deformable
            self.cross_attn = MultiheadAttention(**cross_attn_cfg)
            self.use_deformable_attn = False

        # FFN
        self.ffn = FFN(**ffn_cfg)

        # Layer norms
        self.norms = nn.ModuleList([
            nn.LayerNorm(self.embed_dims) for _ in range(3)
        ])

    def forward(
        self,
        query: torch.Tensor,
        key: Optional[torch.Tensor] = None,
        value: Optional[torch.Tensor] = None,
        query_pos: Optional[torch.Tensor] = None,
        key_pos: Optional[torch.Tensor] = None,
        self_attn_mask: Optional[torch.Tensor] = None,
        cross_attn_mask: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
        **kwargs
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            query: Query tensor [B, num_queries, embed_dims].
            key: Key tensor. Optional.
            value: Value tensor. Optional.
            query_pos: Query positional encoding. Optional.
            key_pos: Key positional encoding. Optional.
            self_attn_mask: Mask for self-attention. Optional.
            cross_attn_mask: Mask for cross-attention. Optional.
            key_padding_mask: Padding mask for keys. Optional.
            **kwargs: Additional arguments for deformable attention.

        Returns:
            Output tensor [B, num_queries, embed_dims].
        """
        # Cross attention
        if self.use_deformable_attn:
            # Use deformable attention with all kwargs
            query = self.cross_attn(
                query=query,
                key=key,
                value=value,
                query_pos=query_pos,
                key_pos=key_pos,
                attn_mask=cross_attn_mask,
                key_padding_mask=key_padding_mask,
                **kwargs)
        else:
            # Fallback: use standard attention
            # For standard attention, key=value=memory (the encoder output)
            # query attends to memory features
            q = query + query_pos if query_pos is not None else query
            k = value  # memory features
            v = value
            query = query + self.cross_attn(
                query=q,
                key=k,
                value=v,
                attn_mask=cross_attn_mask)
        query = self.norms[0](query)

        # Self attention
        query = query + self.self_attn(
            query=query,
            key=query,
            value=query,
            query_pos=query_pos,
            key_pos=query_pos,
            attn_mask=self_attn_mask)
        query = self.norms[1](query)

        # FFN
        query = self.ffn(query)
        query = self.norms[2](query)

        return query


class GaussTRDecoder(nn.Module):
    """GaussTR Transformer Decoder.

    Uses deformable attention for efficient multi-scale feature aggregation.

    Args:
        num_layers: Number of decoder layers.
        layer_cfg: Configuration for each decoder layer.
        return_intermediate: Whether to return intermediate outputs.
        post_norm_cfg: Config for post normalization. Optional.
    """

    def __init__(
        self,
        num_layers: int = 3,
        layer_cfg: Optional[Dict[str, Any]] = None,
        return_intermediate: bool = True,
        post_norm_cfg: Optional[Dict[str, Any]] = None
    ):
        super().__init__()
        self.num_layers = num_layers
        self.return_intermediate = return_intermediate

        # Default layer config
        if layer_cfg is None:
            layer_cfg = {
                'self_attn_cfg': {'embed_dims': 256, 'num_heads': 8, 'dropout': 0.0},
                'cross_attn_cfg': {'embed_dims': 256, 'num_levels': 4},
                'ffn_cfg': {'embed_dims': 256, 'feedforward_channels': 2048}
            }

        # Build layers
        self.layers = nn.ModuleList([
            GaussTRDecoderLayer(**layer_cfg)
            for _ in range(num_layers)
        ])

        embed_dims = layer_cfg['self_attn_cfg']['embed_dims']
        self.embed_dims = embed_dims

        # Reference point head for positional encoding
        self.ref_point_head = MLP(embed_dims, embed_dims, embed_dims, 2)

        # Cache positional encoding temperature scaling tensor (avoids recomputation)
        num_feats = embed_dims // 2
        temperature = 10000
        dim_t = torch.arange(num_feats, dtype=torch.float32)
        dim_t = temperature**(2 * (dim_t // 2) / num_feats)
        self.register_buffer('_pos_enc_dim_t', dim_t)

        # Final layer norm
        self.norm = nn.LayerNorm(embed_dims)

    def forward(
        self,
        query: torch.Tensor,
        value: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor],
        reference_points: torch.Tensor,
        spatial_shapes: torch.Tensor,
        level_start_index: torch.Tensor,
        valid_ratios: torch.Tensor,
        reg_branches: Optional[nn.ModuleList] = None,
        **kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Args:
            query: Query embeddings [B, num_queries, embed_dims].
            value: Memory/value from encoder [B, num_feat_points, embed_dims].
            key_padding_mask: Padding mask [B, num_feat_points]. Optional.
            reference_points: Initial reference points [B, num_queries, 2].
            spatial_shapes: Spatial shapes of each level [num_levels, 2].
            level_start_index: Start index for each level [num_levels].
            valid_ratios: Valid ratios for each level [B, num_levels, 2].
            reg_branches: Regression heads for iterative refinement. Optional.

        Returns:
            Tuple of:
                - Output embeddings [num_layers, B, num_queries, embed_dims]
                - Reference points [num_layers, B, num_queries, 2]
        """
        intermediate = []
        intermediate_reference_points = []

        for lid, layer in enumerate(self.layers):
            # Scale reference points by valid ratios
            if reference_points.shape[-1] == 4:
                reference_points_input = \
                    reference_points[:, :, None] * torch.cat([
                        valid_ratios, valid_ratios], -1)[:, None]
            else:
                assert reference_points.shape[-1] == 2
                reference_points_input = \
                    reference_points[:, :, None] * valid_ratios[:, None]

            # Compute positional encoding from reference points (using cached dim_t)
            query_sine_embed = coordinate_to_encoding(
                reference_points_input[:, :, 0, :],
                query.size(-1) // 2,
                dim_t=self._pos_enc_dim_t)
            query_pos = self.ref_point_head(query_sine_embed)

            # Apply decoder layer
            query = layer(
                query,
                query_pos=query_pos,
                value=value,
                key_padding_mask=key_padding_mask,
                spatial_shapes=spatial_shapes,
                level_start_index=level_start_index,
                valid_ratios=valid_ratios,
                reference_points=reference_points_input,
                **kwargs)

            # Iterative reference point refinement
            if reg_branches is not None:
                tmp_reg_preds = reg_branches[lid](query)[..., :2]
                new_reference_points = tmp_reg_preds + inverse_sigmoid(
                    reference_points)
                new_reference_points = new_reference_points.sigmoid()
                reference_points = new_reference_points.detach()

            if self.return_intermediate:
                intermediate.append(self.norm(query))
                intermediate_reference_points.append(reference_points)

        if self.return_intermediate:
            return torch.stack(intermediate), torch.stack(
                intermediate_reference_points)

        return query, reference_points
