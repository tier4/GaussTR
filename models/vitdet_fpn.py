"""ViTDet FPN implementation for GaussTR Lightning.

Pure PyTorch implementation without MMEngine/MMCV dependencies.
"""

from typing import List, Optional, Dict, Any, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class LN2d(nn.Module):
    """A LayerNorm variant for 2D inputs (images).

    Performs pointwise mean and variance normalization over the channel
    dimension for inputs that have shape (batch_size, channels, height, width).
    """

    def __init__(self, normalized_shape: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.normalized_shape = (normalized_shape,)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape [B, C, H, W].

        Returns:
            Normalized tensor of shape [B, C, H, W].
        """
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x


class ConvNormAct(nn.Module):
    """Convolution + Normalization + Activation module."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 1,
        stride: int = 1,
        padding: int = 0,
        norm_type: str = "LN2d",
        act_type: Optional[str] = None
    ):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size,
            stride=stride, padding=padding, bias=False
        )

        # Normalization (named norm_layer to match original checkpoint)
        if norm_type == "LN2d":
            self.norm_layer = LN2d(out_channels)
        elif norm_type == "BN":
            self.norm_layer = nn.BatchNorm2d(out_channels)
        elif norm_type is None:
            self.norm_layer = nn.Identity()
        else:
            raise ValueError(f"Unknown norm type: {norm_type}")

        # Activation
        if act_type == "gelu":
            self.act = nn.GELU()
        elif act_type == "relu":
            self.act = nn.ReLU(inplace=True)
        elif act_type is None:
            self.act = nn.Identity()
        else:
            raise ValueError(f"Unknown activation type: {act_type}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.norm_layer(x)
        x = self.act(x)
        return x


def get_scaling_modules(
    scale: int,
    channels: int,
    norm_type: str = "LN2d"
) -> nn.Module:
    """Get scaling modules for FPN.

    Args:
        scale: Scaling factor (-2, -1, 0, 1).
            -2: 4x upsample
            -1: 2x upsample
            0: no change
            1: 2x downsample
        channels: Number of input channels.
        norm_type: Type of normalization to use.

    Returns:
        Scaling module.
    """
    assert -2 <= scale <= 1

    if scale == -2:
        # 4x upsample
        if norm_type == "LN2d":
            norm = LN2d(channels // 2)
        else:
            norm = nn.BatchNorm2d(channels // 2)
        return nn.Sequential(
            nn.ConvTranspose2d(channels, channels // 2, 2, 2),
            norm,
            nn.GELU(),
            nn.ConvTranspose2d(channels // 2, channels // 4, 2, 2)
        )
    elif scale == -1:
        # 2x upsample
        return nn.ConvTranspose2d(channels, channels // 2, 2, 2)
    elif scale == 0:
        # No change
        return nn.Identity()
    elif scale == 1:
        # 2x downsample
        return nn.MaxPool2d(kernel_size=2, stride=2)


class ViTDetFPN(nn.Module):
    """Simple Feature Pyramid Network for ViTDet.

    Creates multi-scale features from a single-scale input by applying
    different scaling operations (upsampling/downsampling).

    Args:
        in_channels: Number of input channels.
        out_channels: Number of output channels for each level.
        scales: Tuple of scaling factors for each level.
            Default: (-2, -1, 0, 1) for 4x, 2x, 1x, 0.5x scales.
        norm_type: Type of normalization. Default: "LN2d".
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        scales: Tuple[int, ...] = (-2, -1, 0, 1),
        norm_type: str = "LN2d"
    ):
        super().__init__()

        self.scale_convs = nn.ModuleList([
            get_scaling_modules(scale, in_channels, norm_type)
            for scale in scales
        ])

        # Calculate channels after scaling
        channels = [int(in_channels * 2**min(scale, 0)) for scale in scales]

        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()

        for i in range(len(channels)):
            # 1x1 conv to reduce channels
            l_conv = ConvNormAct(
                channels[i],
                out_channels,
                kernel_size=1,
                norm_type=norm_type
            )

            # 3x3 conv for feature refinement
            fpn_conv = ConvNormAct(
                out_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                norm_type=norm_type
            )

            self.lateral_convs.append(l_conv)
            self.fpn_convs.append(fpn_conv)

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """Forward pass.

        Args:
            x: Input feature tensor of shape [B, C, H, W].

        Returns:
            List of multi-scale feature tensors.
        """
        # Apply scaling to get different resolution features
        inputs = [scale_conv(x) for scale_conv in self.scale_convs]

        # Apply lateral convs
        laterals = [
            lateral_conv(inputs[i])
            for i, lateral_conv in enumerate(self.lateral_convs)
        ]

        # Apply FPN convs
        outs = [
            fpn_conv(laterals[i])
            for i, fpn_conv in enumerate(self.fpn_convs)
        ]

        return outs
