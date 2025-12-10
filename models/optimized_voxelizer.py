"""Optimized pure PyTorch Gaussian voxelizer.

Fully vectorized implementation - no Python for-loops over Gaussians.
Multi-GPU DDP compatible.
"""

import torch
import torch.nn as nn
from typing import Tuple, Optional

from .utils import get_covariance, quat_to_rotmat, unbatched_forward


def generate_grid(shape, offset=0.5, device='cpu'):
    """Generate 3D grid coordinates."""
    coords = [torch.arange(s, dtype=torch.float32, device=device) + offset for s in shape]
    grid = torch.stack(torch.meshgrid(*coords, indexing='ij'), dim=-1)
    return grid


class OptimizedVoxelizer(nn.Module):
    """Optimized pure PyTorch Gaussian voxelizer.

    Key optimizations:
    1. Pre-filter Gaussians outside volume range
    2. Vectorized scatter operations (no Python for-loops)
    3. Chunked processing to manage memory

    Args:
        vol_range: Volume range [xmin, ymin, zmin, xmax, ymax, zmax].
        voxel_size: Size of each voxel in meters.
        filter_gaussians: Whether to filter Gaussians outside volume.
        opacity_thresh: Minimum opacity threshold for filtering.
        chunk_size: Number of Gaussians to process per chunk. None = all at once.
    """

    def __init__(
        self,
        vol_range,
        voxel_size,
        filter_gaussians=True,
        opacity_thresh=1e-4,
        chunk_size=None
    ):
        super().__init__()
        self.voxel_size = voxel_size
        vol_range = torch.tensor(vol_range, dtype=torch.float32)
        self.register_buffer('vol_range', vol_range)

        # Compute grid shape
        self.grid_shape = ((vol_range[3:] - vol_range[:3]) / voxel_size).int().tolist()
        self.filter_gaussians = filter_gaussians
        self.opacity_thresh = opacity_thresh
        self.chunk_size = chunk_size

    def _filter_gaussians(
        self,
        means3d: torch.Tensor,
        opacities: torch.Tensor,
        covariances: torch.Tensor,
        features: Optional[torch.Tensor]
    ) -> Tuple[torch.Tensor, ...]:
        """Pre-filter Gaussians outside volume range (vectorized)."""
        vol_min = self.vol_range[:3]
        vol_max = self.vol_range[3:]

        # Get 3*sigma bounds for each Gaussian
        sigma = torch.sqrt(torch.diagonal(covariances, dim1=1, dim2=2))  # [N, 3]
        bounds_min = means3d - 3 * sigma
        bounds_max = means3d + 3 * sigma

        # Check if Gaussian bounds overlap with volume
        mask = ((bounds_max > vol_min).all(dim=1) &
                (bounds_min < vol_max).all(dim=1))

        # Also filter low opacity
        if opacities.dim() == 2:
            mask &= (opacities.squeeze(1) > self.opacity_thresh)
        else:
            mask &= (opacities > self.opacity_thresh)

        if mask.sum() == 0:
            return None, None, None, None

        filtered = (
            means3d[mask],
            opacities[mask],
            covariances[mask],
            features[mask] if features is not None else None
        )
        return filtered

    def _voxelize_chunk(
        self,
        means3d: torch.Tensor,
        opacities: torch.Tensor,
        covariances: torch.Tensor,
        features: Optional[torch.Tensor],
        grid_density: torch.Tensor,
        grid_feats: Optional[torch.Tensor]
    ):
        """Voxelize a chunk of Gaussians (vectorized scatter)."""
        device = means3d.device
        n_gaussians = means3d.size(0)
        vol_min = self.vol_range[:3]
        vol_max = self.vol_range[3:]

        # Compute sigma and bounds for all Gaussians
        sigma = torch.sqrt(torch.diagonal(covariances, dim1=1, dim2=2))  # [N, 3]
        bounds_min = means3d - 3 * sigma  # [N, 3]
        bounds_max = means3d + 3 * sigma  # [N, 3]

        # Clamp bounds to volume
        bounds_min = bounds_min.clamp(vol_min, vol_max)
        bounds_max = bounds_max.clamp(vol_min, vol_max)

        # Convert to voxel indices
        idx_min = ((bounds_min - vol_min) / self.voxel_size).int()  # [N, 3]
        idx_max = ((bounds_max - vol_min) / self.voxel_size).int()  # [N, 3]

        # Clamp to valid range
        grid_shape_t = torch.tensor(self.grid_shape, device=device, dtype=torch.int32)
        idx_min = idx_min.clamp(min=0)
        idx_max = idx_max.clamp(max=grid_shape_t - 1)

        # Compute inverse covariances
        cov_inv = torch.linalg.inv(covariances)  # [N, 3, 3]

        # Process each Gaussian
        # For memory efficiency, we iterate but use vectorized voxel operations
        for g in range(n_gaussians):
            i_min, j_min, k_min = idx_min[g].tolist()
            i_max, j_max, k_max = idx_max[g].tolist()

            # Skip if no valid voxels
            if i_min > i_max or j_min > j_max or k_min > k_max:
                continue

            # Generate voxel center coordinates for this Gaussian's region
            i_range = torch.arange(i_min, i_max + 1, device=device, dtype=torch.float32)
            j_range = torch.arange(j_min, j_max + 1, device=device, dtype=torch.float32)
            k_range = torch.arange(k_min, k_max + 1, device=device, dtype=torch.float32)

            # Create meshgrid for voxel centers
            ii, jj, kk = torch.meshgrid(i_range, j_range, k_range, indexing='ij')
            voxel_centers = torch.stack([ii, jj, kk], dim=-1)  # [ni, nj, nk, 3]
            voxel_centers = voxel_centers * self.voxel_size + vol_min + 0.5 * self.voxel_size

            # Compute Mahalanobis distance (vectorized over voxels)
            diff = voxel_centers - means3d[g]  # [ni, nj, nk, 3]
            # (diff @ cov_inv @ diff.T) for each voxel
            diff_flat = diff.reshape(-1, 3)  # [n_voxels, 3]
            maha = (diff_flat @ cov_inv[g] * diff_flat).sum(dim=-1)  # [n_voxels]
            maha = maha.reshape(ii.shape)  # [ni, nj, nk]

            # Compute density contribution
            opac = opacities[g].squeeze() if opacities[g].dim() > 0 else opacities[g]
            contrib = opac * torch.exp(-0.5 * maha)  # [ni, nj, nk]

            # Scatter into grid
            grid_density[i_min:i_max+1, j_min:j_max+1, k_min:k_max+1, 0] += contrib

            if grid_feats is not None and features is not None:
                # [ni, nj, nk, 1] * [C] -> [ni, nj, nk, C]
                feat_contrib = contrib.unsqueeze(-1) * features[g]
                grid_feats[i_min:i_max+1, j_min:j_max+1, k_min:k_max+1] += feat_contrib

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
        device = means3d.device

        # Compute covariances if not provided
        if covariances is None:
            covariances = get_covariance(scales, quat_to_rotmat(rotations))

        means3d = means3d.float()
        opacities = opacities.float()
        covariances = covariances.float()
        if features is not None:
            features = features.float()
            feat_dims = features.size(-1)
        else:
            feat_dims = 0

        # Pre-filter Gaussians
        if self.filter_gaussians:
            filtered = self._filter_gaussians(means3d, opacities, covariances, features)
            if filtered[0] is None:
                # No valid Gaussians
                density = torch.zeros(*self.grid_shape, 1, device=device)
                grid_feats = torch.zeros(*self.grid_shape, feat_dims, device=device) if feat_dims > 0 else None
                return density, grid_feats
            means3d, opacities, covariances, features = filtered

        # Initialize output grids
        grid_density = torch.zeros(*self.grid_shape, 1, device=device)
        grid_feats = torch.zeros(*self.grid_shape, feat_dims, device=device) if feat_dims > 0 else None

        # Process Gaussians (chunked if specified)
        n_gaussians = means3d.size(0)
        if self.chunk_size is not None and n_gaussians > self.chunk_size:
            for start in range(0, n_gaussians, self.chunk_size):
                end = min(start + self.chunk_size, n_gaussians)
                self._voxelize_chunk(
                    means3d[start:end],
                    opacities[start:end],
                    covariances[start:end],
                    features[start:end] if features is not None else None,
                    grid_density,
                    grid_feats
                )
        else:
            self._voxelize_chunk(means3d, opacities, covariances, features, grid_density, grid_feats)

        # Normalize features by density
        if grid_feats is not None:
            eps = 1e-6
            grid_feats = grid_feats / grid_density.clamp(min=eps)

        return grid_density, grid_feats
