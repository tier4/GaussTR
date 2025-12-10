"""Utility functions for GaussTR Lightning.

Pure PyTorch implementations without MMEngine dependencies.
"""

from functools import reduce
from typing import List, Tuple, Dict, Any, Optional, Callable, Union

import numpy as np
import torch
import torch.nn as nn
from pyquaternion import Quaternion


def cumprod(xs):
    """Cumulative product of a list."""
    return reduce(lambda x, y: x * y, xs)


def nlc_to_nchw(x: torch.Tensor, shape: Tuple[int, int]) -> torch.Tensor:
    """Convert [N, L, C] shape tensor to [N, C, H, W] shape tensor.

    Args:
        x: Input tensor of shape [N, L, C].
        shape: The height and width of output feature map (H, W).

    Returns:
        Output tensor of shape [N, C, H, W].
    """
    B, L, C = x.shape
    return x.transpose(1, 2).reshape(B, C, *shape).contiguous()


def nchw_to_nlc(x: torch.Tensor) -> torch.Tensor:
    """Flatten [N, C, H, W] shape tensor to [N, L, C] shape tensor.

    Args:
        x: Input tensor of shape [N, C, H, W].

    Returns:
        Output tensor of shape [N, L, C].
    """
    return x.flatten(2).transpose(1, 2).contiguous()


def flatten_multi_scale_feats(
    feats: List[torch.Tensor]
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Flatten multi-scale features into a single tensor.

    Args:
        feats: List of feature tensors with shapes [B, C, H_i, W_i].

    Returns:
        Tuple of:
            - Flattened features of shape [B, sum(H_i*W_i), C]
            - Spatial shapes tensor of shape [num_levels, 2]
    """
    feat_flatten = torch.cat([nchw_to_nlc(feat) for feat in feats], dim=1)
    shapes = torch.stack([
        torch.tensor(feat.shape[2:], device=feat_flatten.device)
        for feat in feats
    ])
    return feat_flatten, shapes


def get_level_start_index(shapes: torch.Tensor) -> torch.Tensor:
    """Get start indices for each level in flattened features.

    Args:
        shapes: Spatial shapes tensor of shape [num_levels, 2].

    Returns:
        Start indices tensor of shape [num_levels].
    """
    return torch.cat((shapes.new_zeros((1,)), shapes.prod(1).cumsum(0)[:-1]))


def generate_grid(
    grid_shape: Tuple[int, ...],
    value: Optional[Tuple[int, ...]] = None,
    offset: float = 0,
    normalize: bool = False
) -> torch.Tensor:
    """Generate coordinate grid.

    Args:
        grid_shape: The shape of grid to generate.
        value: The unscaled value the grid represents. Defaults to grid_shape.
        offset: Offset to add to coordinates. Defaults to 0.
        normalize: Whether to normalize coordinates to [0, 1]. Defaults to False.

    Returns:
        Grid coordinates of shape [*grid_shape, len(grid_shape)].
    """
    if value is None:
        value = grid_shape
    grid = []
    for i, (s, val) in enumerate(zip(grid_shape, value)):
        g = torch.linspace(offset, val - 1 + offset, s, dtype=torch.float)
        if normalize:
            g /= val
        shape_ = [1 for _ in grid_shape]
        shape_[i] = s
        g = g.reshape(*shape_).expand(*grid_shape)
        grid.append(g)
    return torch.stack(grid, dim=-1)


def cam2world(
    points: torch.Tensor,
    cam2img: torch.Tensor,
    cam2ego: torch.Tensor,
    img_aug_mat: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """Transform points from camera image space to world (ego) space.

    Args:
        points: Points in image space [B, N, Q, 3] (u, v, depth).
        cam2img: Camera intrinsics [B, N, 4, 4] or [B, N, 3, 3].
        cam2ego: Camera to ego transformation [B, N, 4, 4].
        img_aug_mat: Image augmentation matrix [B, N, 4, 4]. Optional.

    Returns:
        Points in world (ego) space [B, N, Q, 3].
    """
    if img_aug_mat is not None:
        post_rots = img_aug_mat[..., :3, :3]
        post_trans = img_aug_mat[..., :3, 3]
        points = points - post_trans.unsqueeze(-2)
        points = (torch.inverse(post_rots).unsqueeze(2)
                  @ points.unsqueeze(-1)).squeeze(-1)

    cam2img = cam2img[..., :3, :3]
    with torch.amp.autocast('cuda', enabled=False):
        combine = cam2ego[..., :3, :3] @ torch.inverse(cam2img)
        points = points.float()
        points = torch.cat(
            [points[..., :2] * points[..., 2:3], points[..., 2:3]], dim=-1)
        points = combine.unsqueeze(2) @ points.unsqueeze(-1)
    points = points.squeeze(-1) + cam2ego[..., None, :3, 3]
    return points


def world2cam(
    points: torch.Tensor,
    cam2img: torch.Tensor,
    cam2ego: torch.Tensor,
    img_aug_mat: Optional[torch.Tensor] = None,
    eps: float = 1e-6
) -> torch.Tensor:
    """Transform points from world (ego) space to camera image space.

    Args:
        points: Points in world space [B, N, Q, 3].
        cam2img: Camera intrinsics [B, N, 4, 4] or [B, N, 3, 3].
        cam2ego: Camera to ego transformation [B, N, 4, 4].
        img_aug_mat: Image augmentation matrix [B, N, 4, 4]. Optional.
        eps: Small value for numerical stability.

    Returns:
        Points in image space [B, N, Q, 2] (u, v).
    """
    points = points - cam2ego[..., None, :3, 3]
    points = torch.inverse(cam2ego[..., None, :3, :3]) @ points.unsqueeze(-1)
    points = (cam2img[..., None, :3, :3] @ points).squeeze(-1)
    points = points / points[..., 2:3].clamp(eps)
    if img_aug_mat is not None:
        points = img_aug_mat[..., None, :3, :3] @ points.unsqueeze(-1)
        points = points.squeeze(-1) + img_aug_mat[..., None, :3, 3]
    return points[..., :2]


def rotmat_to_quat(rot_matrices: torch.Tensor) -> torch.Tensor:
    """Convert rotation matrices to quaternions (pure PyTorch GPU implementation).

    Uses Shepperd's method for numerical stability.

    Args:
        rot_matrices: Rotation matrices of shape [..., 3, 3].

    Returns:
        Quaternions of shape [..., 4] in (w, x, y, z) format.
    """
    batch_shape = rot_matrices.shape[:-2]
    R = rot_matrices.reshape(-1, 3, 3)

    # Shepperd's method - numerically stable
    # Compute all four possible quaternion representations
    trace = R[:, 0, 0] + R[:, 1, 1] + R[:, 2, 2]

    # Pre-allocate output
    quats = torch.zeros(R.shape[0], 4, device=R.device, dtype=R.dtype)

    # Case 1: trace > 0
    mask1 = trace > 0
    if mask1.any():
        s = torch.sqrt(trace[mask1] + 1.0) * 2  # s = 4 * w
        quats[mask1, 0] = 0.25 * s
        quats[mask1, 1] = (R[mask1, 2, 1] - R[mask1, 1, 2]) / s
        quats[mask1, 2] = (R[mask1, 0, 2] - R[mask1, 2, 0]) / s
        quats[mask1, 3] = (R[mask1, 1, 0] - R[mask1, 0, 1]) / s

    # Case 2: R[0,0] > R[1,1] and R[0,0] > R[2,2]
    mask2 = (~mask1) & (R[:, 0, 0] > R[:, 1, 1]) & (R[:, 0, 0] > R[:, 2, 2])
    if mask2.any():
        s = torch.sqrt(1.0 + R[mask2, 0, 0] - R[mask2, 1, 1] - R[mask2, 2, 2]) * 2  # s = 4 * x
        quats[mask2, 0] = (R[mask2, 2, 1] - R[mask2, 1, 2]) / s
        quats[mask2, 1] = 0.25 * s
        quats[mask2, 2] = (R[mask2, 0, 1] + R[mask2, 1, 0]) / s
        quats[mask2, 3] = (R[mask2, 0, 2] + R[mask2, 2, 0]) / s

    # Case 3: R[1,1] > R[2,2]
    mask3 = (~mask1) & (~mask2) & (R[:, 1, 1] > R[:, 2, 2])
    if mask3.any():
        s = torch.sqrt(1.0 + R[mask3, 1, 1] - R[mask3, 0, 0] - R[mask3, 2, 2]) * 2  # s = 4 * y
        quats[mask3, 0] = (R[mask3, 0, 2] - R[mask3, 2, 0]) / s
        quats[mask3, 1] = (R[mask3, 0, 1] + R[mask3, 1, 0]) / s
        quats[mask3, 2] = 0.25 * s
        quats[mask3, 3] = (R[mask3, 1, 2] + R[mask3, 2, 1]) / s

    # Case 4: else
    mask4 = (~mask1) & (~mask2) & (~mask3)
    if mask4.any():
        s = torch.sqrt(1.0 + R[mask4, 2, 2] - R[mask4, 0, 0] - R[mask4, 1, 1]) * 2  # s = 4 * z
        quats[mask4, 0] = (R[mask4, 1, 0] - R[mask4, 0, 1]) / s
        quats[mask4, 1] = (R[mask4, 0, 2] + R[mask4, 2, 0]) / s
        quats[mask4, 2] = (R[mask4, 1, 2] + R[mask4, 2, 1]) / s
        quats[mask4, 3] = 0.25 * s

    # Normalize to unit quaternion
    quats = quats / quats.norm(dim=-1, keepdim=True).clamp(min=1e-8)

    return quats.reshape(*batch_shape, 4)


def quat_to_rotmat(quats: torch.Tensor) -> torch.Tensor:
    """Convert quaternions to rotation matrices.

    Args:
        quats: Quaternions of shape [..., 4] in (w, x, y, z) format.

    Returns:
        Rotation matrices of shape [..., 3, 3].
    """
    q = quats / torch.sqrt((quats**2).sum(dim=-1, keepdim=True))
    r, x, y, z = [i.squeeze(-1) for i in q.split(1, dim=-1)]

    R = torch.zeros((*r.shape, 3, 3)).to(r)
    R[..., 0, 0] = 1 - 2 * (y * y + z * z)
    R[..., 0, 1] = 2 * (x * y - r * z)
    R[..., 0, 2] = 2 * (x * z + r * y)
    R[..., 1, 0] = 2 * (x * y + r * z)
    R[..., 1, 1] = 1 - 2 * (x * x + z * z)
    R[..., 1, 2] = 2 * (y * z - r * x)
    R[..., 2, 0] = 2 * (x * z - r * y)
    R[..., 2, 1] = 2 * (y * z + r * x)
    R[..., 2, 2] = 1 - 2 * (x * x + y * y)
    return R


def get_covariance(s: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
    """Compute covariance matrices from scales and rotations.

    Args:
        s: Scales of shape [B, N, 3].
        r: Rotation matrices of shape [B, N, 3, 3].

    Returns:
        Covariance matrices of shape [B, N, 3, 3].
    """
    L = torch.zeros((*s.shape[:2], 3, 3)).to(s)
    for i in range(s.size(-1)):
        L[..., i, i] = s[..., i]

    L = r @ L
    covariance = L @ L.mT
    return covariance


def unbatched_forward(func: Callable) -> Callable:
    """Decorator to apply function to each batch element separately.

    This is useful for functions that don't support batched inputs.
    """
    def wrapper(*args, **kwargs):
        bs = None
        for arg in list(args) + list(kwargs.values()):
            if isinstance(arg, torch.Tensor):
                if bs is None:
                    bs = arg.size(0)
                else:
                    assert bs == arg.size(0)

        outputs = []
        for i in range(bs):
            output = func(
                *[
                    arg[i] if isinstance(arg, torch.Tensor) else arg
                    for arg in args
                ], **{
                    k: v[i] if isinstance(v, torch.Tensor) else v
                    for k, v in kwargs.items()
                })
            outputs.append(output)

        if isinstance(outputs[0], tuple):
            return tuple([
                torch.stack([out[i] for out in outputs])
                for i in range(len(outputs[0]))
            ])
        else:
            return torch.stack(outputs)

    return wrapper


def apply_to_items(
    func: Callable,
    iterable: Union[List, Dict]
) -> Union[List, Dict]:
    """Apply function to each item in list or dict."""
    if isinstance(iterable, list):
        return [func(i) for i in iterable]
    elif isinstance(iterable, dict):
        return {k: func(v) for k, v in iterable.items()}


def flatten_bsn_forward(func: Callable, *args, **kwargs):
    """Flatten batch and sequence dimensions, apply function, then reshape back."""
    args = list(args)
    bsn = None
    for i, arg in enumerate(args):
        if isinstance(arg, torch.Tensor):
            if bsn is None:
                bsn = arg.shape[:2]
            args[i] = arg.flatten(0, 1)
    for k, v in kwargs.items():
        if isinstance(v, torch.Tensor):
            if bsn is None:
                bsn = v.shape[:2]
            kwargs[k] = v.flatten(0, 1)
    outs = func(*args, **kwargs)
    if isinstance(outs, tuple):
        outs = list(outs)
        for i, out in enumerate(outs):
            outs[i] = out.reshape(bsn + out.shape[1:])
        return tuple(outs)
    else:
        outs = outs.reshape(bsn + outs.shape[1:])
    return outs


# OCC3D category definitions
OCC3D_CATEGORIES = (
    ['barrier'],
    ['bicycle'],
    ['bus'],
    ['car'],
    ['construction vehicle'],
    ['motorcycle'],
    ['person'],
    ['cone'],
    ['trailer'],
    ['truck'],
    ['road'],
    ['sidewalk'],
    ['terrain', 'grass'],
    ['building', 'wall', 'fence', 'pole', 'sign'],
    ['vegetation'],
    ['sky'],
)
