"""Pure PyTorch Gaussian voxelizer.

This matches the original MMCV GaussianVoxelizer implementation.
No Taichi dependency - works correctly with multi-GPU DDP.
"""

import torch
import torch.nn as nn
from typing import Tuple, Optional

from .utils import get_covariance, quat_to_rotmat, unbatched_forward


def generate_grid(shape, offset=0.5):
    """Generate 3D grid coordinates."""
    coords = [torch.arange(s, dtype=torch.float32) + offset for s in shape]
    grid = torch.stack(torch.meshgrid(*coords, indexing='ij'), dim=-1)
    return grid


def splat_into_3d(
    grid_coords: torch.Tensor,
    means3d: torch.Tensor,
    opacities: torch.Tensor,
    covariances: torch.Tensor,
    vol_range: torch.Tensor,
    voxel_size: float,
    features: Optional[torch.Tensor] = None,
    eps: float = 1e-6
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Splat Gaussians into 3D voxel grid.

    This is the pure PyTorch implementation matching the original MMCV version.

    Args:
        grid_coords: Voxel center coordinates [X, Y, Z, 3].
        means3d: Gaussian centers [N, 3].
        opacities: Gaussian opacities [N, 1].
        covariances: Gaussian covariance matrices [N, 3, 3].
        vol_range: Volume range tensor [6] (xmin, ymin, zmin, xmax, ymax, zmax).
        voxel_size: Size of each voxel in meters.
        features: Gaussian features [N, C]. Optional.
        eps: Small value for numerical stability.

    Returns:
        Tuple of (density [X, Y, Z, 1], features [X, Y, Z, C]).
    """
    grid_density = torch.zeros(
        (*grid_coords.shape[:-1], 1),
        device=grid_coords.device,
        dtype=grid_coords.dtype
    )

    if features is not None:
        grid_feats = torch.zeros(
            (*grid_coords.shape[:-1], features.size(-1)),
            device=grid_coords.device,
            dtype=grid_coords.dtype
        )

    vol_min = vol_range[:3]
    vol_max = vol_range[3:]

    for g in range(means3d.size(0)):
        # Get Gaussian parameters
        mean = means3d[g]
        cov = covariances[g]
        opac = opacities[g]

        # Compute 3-sigma bounds
        sigma = torch.sqrt(torch.diag(cov))
        factor = 3 * torch.tensor([-1, 1], device=mean.device, dtype=mean.dtype)[:, None]
        bounds = mean[None] + factor * sigma[None]

        # Check if Gaussian is within volume range
        if not (((bounds > vol_min[None]).max(0).values.min()) and
                ((bounds < vol_max[None]).max(0).values.min())):
            continue

        # Clamp bounds to volume range
        bounds = bounds.clamp(vol_min, vol_max)
        bounds = ((bounds - vol_min) / voxel_size).int().tolist()
        slices = tuple([slice(lo, hi + 1) for lo, hi in zip(*bounds)])

        # Compute Mahalanobis distance for voxels in bounds
        diff = grid_coords[slices] - mean
        cov_inv = cov.inverse()
        maha_dist = (diff.unsqueeze(-2) @ cov_inv @ diff.unsqueeze(-1)).squeeze(-1)

        # Compute density contribution
        density = opac * torch.exp(-0.5 * maha_dist)
        grid_density[slices] += density

        if features is not None:
            grid_feats[slices] += density * features[g]

    if features is None:
        return grid_density, None

    # Normalize features by density
    grid_feats = grid_feats / grid_density.clamp(min=eps)
    return grid_density, grid_feats


class PyTorchVoxelizer(nn.Module):
    """Pure PyTorch Gaussian voxelizer.

    This is a direct port of the original MMCV GaussianVoxelizer.
    Works correctly with multi-GPU DDP (no Taichi global state issues).

    Args:
        vol_range: Volume range [xmin, ymin, zmin, xmax, ymax, zmax].
        voxel_size: Size of each voxel in meters.
        filter_gaussians: Whether to filter Gaussians outside volume.
        opacity_thresh: Minimum opacity threshold for filtering.
        covariance_thresh: Minimum covariance threshold for filtering.
    """

    def __init__(
        self,
        vol_range,
        voxel_size,
        filter_gaussians=False,
        opacity_thresh=0,
        covariance_thresh=0
    ):
        super().__init__()
        self.voxel_size = voxel_size
        vol_range = torch.tensor(vol_range, dtype=torch.float32)
        self.register_buffer('vol_range', vol_range)

        # Compute grid shape
        self.grid_shape = ((vol_range[3:] - vol_range[:3]) / voxel_size).int().tolist()

        # Pre-compute grid coordinates
        grid_coords = generate_grid(self.grid_shape, offset=0.5)
        grid_coords = grid_coords * voxel_size + vol_range[:3]
        self.register_buffer('grid_coords', grid_coords)

        self.filter_gaussians = filter_gaussians
        self.opacity_thresh = opacity_thresh
        self.covariance_thresh = covariance_thresh

    @unbatched_forward
    def __call__(
        self,
        means3d: torch.Tensor,
        opacities: torch.Tensor,
        covariances: Optional[torch.Tensor] = None,
        scales: Optional[torch.Tensor] = None,
        rotations: Optional[torch.Tensor] = None,
        features: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Voxelize Gaussians.

        Args:
            means3d: Gaussian centers [N, 3].
            opacities: Gaussian opacities [N, 1].
            covariances: Gaussian covariance matrices [N, 3, 3]. Optional.
            scales: Gaussian scales [N, 3]. Used if covariances not provided.
            rotations: Gaussian rotations [N, 4]. Used if covariances not provided.
            features: Gaussian features [N, C].

        Returns:
            Tuple of (density [X, Y, Z, 1], features [X, Y, Z, C]).
        """
        # Compute covariances if not provided
        if covariances is None:
            covariances = get_covariance(scales, quat_to_rotmat(rotations))

        # Prepare Gaussian dict
        gaussians = dict(
            means3d=means3d.float(),
            opacities=opacities.float(),
            covariances=covariances.float(),
        )
        if features is not None:
            gaussians['features'] = features.float()

        # Filter Gaussians outside volume
        if self.filter_gaussians:
            mask = opacities.squeeze(1) > self.opacity_thresh
            for i in range(3):
                mask &= (means3d[:, i] >= self.vol_range[i]) & \
                        (means3d[:, i] <= self.vol_range[i + 3])

            if self.covariance_thresh > 0:
                cov_diag = torch.diagonal(covariances, dim1=1, dim2=2)
                mask &= ((cov_diag.min(1)[0] * 6) > self.covariance_thresh)

            # Apply mask
            gaussians = {k: v[mask] for k, v in gaussians.items()}

        # Splat into 3D grid
        density, grid_feats = splat_into_3d(
            self.grid_coords,
            gaussians['means3d'],
            gaussians['opacities'],
            gaussians['covariances'],
            self.vol_range,
            self.voxel_size,
            features=gaussians.get('features'),
        )

        return density, grid_feats
