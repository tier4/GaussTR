"""Standalone Feature Pyramid Network (no mmdet/mmcv dependency)."""

import torch.nn as nn
import torch.nn.functional as F


class FPN(nn.Module):
    """Feature Pyramid Network.

    Takes multi-scale backbone features and produces multi-scale outputs
    with the same channel dimension via lateral connections + top-down pathway.

    Args:
        in_channels: Input channels per level (e.g. [256, 512, 1024, 2048] for ResNet50).
        out_channels: Output channels for all levels (e.g. 256).
        num_outs: Number of output levels. Extra levels use stride-2 conv on last input.
    """

    def __init__(self, in_channels, out_channels, num_outs=None):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_outs = num_outs or len(in_channels)

        # Lateral 1x1 convolutions (reduce channels)
        self.lateral_convs = nn.ModuleList()
        for ch in in_channels:
            self.lateral_convs.append(nn.Conv2d(ch, out_channels, 1))

        # Smoothing 3x3 convolutions (after top-down merge)
        self.fpn_convs = nn.ModuleList()
        for _ in in_channels:
            self.fpn_convs.append(nn.Conv2d(out_channels, out_channels, 3, padding=1))

        # Extra output levels via stride-2 conv on last feature
        if self.num_outs > len(in_channels):
            self.extra_convs = nn.ModuleList()
            for _ in range(self.num_outs - len(in_channels)):
                self.extra_convs.append(nn.Conv2d(out_channels, out_channels, 3, stride=2, padding=1))
        else:
            self.extra_convs = None

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, inputs):
        """
        Args:
            inputs: List of feature tensors from backbone, one per level.

        Returns:
            List of FPN feature tensors.
        """
        assert len(inputs) == len(self.in_channels)

        # Lateral connections
        laterals = [conv(x) for conv, x in zip(self.lateral_convs, inputs)]

        # Top-down pathway (from highest to lowest resolution)
        for i in range(len(laterals) - 1, 0, -1):
            laterals[i - 1] = laterals[i - 1] + F.interpolate(
                laterals[i], size=laterals[i - 1].shape[2:], mode='nearest')

        # Smoothing
        outs = [conv(lat) for conv, lat in zip(self.fpn_convs, laterals)]

        # Extra levels
        if self.extra_convs is not None:
            x = outs[-1]
            for conv in self.extra_convs:
                x = conv(F.relu(x))
                outs.append(x)

        return outs
