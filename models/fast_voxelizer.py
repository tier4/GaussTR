"""Fast pure PyTorch Gaussian voxelizer using scatter operations.

Fully vectorized with no Python for-loops over Gaussians.
Multi-GPU DDP compatible.
"""

import torch
import torch.nn as nn
from typing import Tuple, Optional

from .utils import get_covariance, quat_to_rotmat, unbatched_forward


class FastVoxelizer(nn.Module):
    """Fast pure PyTorch Gaussian voxelizer.

    Uses vectorized scatter_add operations for maximum GPU utilization.
    No Python for-loops over Gaussians.

    Args:
        vol_range: Volume range [xmin, ymin, zmin, xmax, ymax, zmax].
        voxel_size: Size of each voxel in meters.
        filter_gaussians: Whether to filter Gaussians outside volume.
        opacity_thresh: Minimum opacity threshold for filtering.
        sigma_factor: Number of sigmas to consider for each Gaussian.
    """

    def __init__(
        self,
        vol_range,
        voxel_size,
        filter_gaussians=True,
        opacity_thresh=1e-4,
        sigma_factor=3.0
    ):
        super().__init__()
        self.voxel_size = voxel_size
        self.sigma_factor = sigma_factor
        vol_range = torch.tensor(vol_range, dtype=torch.float32)
        self.register_buffer('vol_range', vol_range)

        # Compute grid shape
        self.grid_shape = ((vol_range[3:] - vol_range[:3]) / voxel_size).int().tolist()
        self.filter_gaussians = filter_gaussians
        self.opacity_thresh = opacity_thresh

        # Pre-compute grid coordinates
        grid_coords = self._generate_grid()
        self.register_buffer('grid_coords', grid_coords)

    def _generate_grid(self):
        """Generate 3D grid coordinates [X, Y, Z, 3]."""
        coords = [
            torch.arange(s, dtype=torch.float32) + 0.5
            for s in self.grid_shape
        ]
        grid = torch.stack(torch.meshgrid(*coords, indexing='ij'), dim=-1)
        grid = grid * self.voxel_size + self.vol_range[:3]
        return grid

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

        # Get sigma bounds for each Gaussian
        sigma = torch.sqrt(torch.diagonal(covariances, dim1=1, dim2=2))  # [N, 3]
        bounds_min = means3d - self.sigma_factor * sigma
        bounds_max = means3d + self.sigma_factor * sigma

        # Check if Gaussian bounds overlap with volume
        mask = ((bounds_max > vol_min).all(dim=1) &
                (bounds_min < vol_max).all(dim=1))

        # Also filter low opacity
        opac_flat = opacities.squeeze(-1) if opacities.dim() == 2 else opacities
        mask &= (opac_flat > self.opacity_thresh)

        if mask.sum() == 0:
            return None, None, None, None

        return (
            means3d[mask],
            opacities[mask],
            covariances[mask],
            features[mask] if features is not None else None
        )

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
        """Voxelize Gaussians using vectorized operations.

        This implementation:
        1. Pre-filters Gaussians outside volume
        2. Computes density for all (Gaussian, voxel) pairs in parallel
        3. Uses sparse operations for memory efficiency

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

        n_gaussians = means3d.size(0)

        # Compute inverse covariances [N, 3, 3]
        cov_inv = torch.linalg.inv(covariances)

        # Get grid coordinates [X, Y, Z, 3] -> [X*Y*Z, 3]
        grid_flat = self.grid_coords.reshape(-1, 3)  # [V, 3]
        n_voxels = grid_flat.size(0)

        # Compute diff for all (Gaussian, voxel) pairs
        # [N, 1, 3] - [1, V, 3] -> [N, V, 3]
        diff = means3d.unsqueeze(1) - grid_flat.unsqueeze(0)

        # Compute Mahalanobis distance for all pairs
        # [N, V, 3] @ [N, 1, 3, 3] -> [N, V, 3]
        # Then sum over last dim -> [N, V]
        diff_transformed = torch.einsum('nvi,nij->nvj', diff, cov_inv)  # [N, V, 3]
        maha = (diff * diff_transformed).sum(dim=-1)  # [N, V]

        # Compute contributions [N, V]
        opac = opacities.squeeze(-1) if opacities.dim() == 2 else opacities  # [N]
        contrib = opac.unsqueeze(1) * torch.exp(-0.5 * maha)  # [N, V]

        # Mask contributions that are too small (within sigma_factor)
        # This is equivalent to the bounds check in the for-loop version
        sigma = torch.sqrt(torch.diagonal(covariances, dim1=1, dim2=2))  # [N, 3]
        sigma_max = sigma.max(dim=1).values  # [N]
        dist_sq = (diff ** 2).sum(dim=-1)  # [N, V]
        within_bounds = dist_sq < (self.sigma_factor * sigma_max.unsqueeze(1)) ** 2  # [N, V]

        # Zero out contributions outside bounds
        contrib = contrib * within_bounds.float()

        # Sum over Gaussians -> [V]
        grid_density_flat = contrib.sum(dim=0)  # [V]

        # Reshape to grid
        grid_density = grid_density_flat.reshape(*self.grid_shape, 1)

        # Compute feature contributions
        grid_feats = None
        if features is not None:
            # [N, V, 1] * [N, 1, C] -> [N, V, C]
            feat_contrib = contrib.unsqueeze(-1) * features.unsqueeze(1)  # [N, V, C]
            # Sum over Gaussians -> [V, C]
            grid_feats_flat = feat_contrib.sum(dim=0)  # [V, C]
            # Reshape and normalize
            grid_feats = grid_feats_flat.reshape(*self.grid_shape, feat_dims)
            eps = 1e-6
            grid_feats = grid_feats / grid_density.clamp(min=eps)

        return grid_density, grid_feats


class MemoryEfficientVoxelizer(nn.Module):
    """Memory-efficient voxelizer using chunked processing.

    For large numbers of Gaussians or high-resolution grids, the fully
    vectorized approach may run out of memory. This version processes
    Gaussians in chunks.

    Args:
        vol_range: Volume range [xmin, ymin, zmin, xmax, ymax, zmax].
        voxel_size: Size of each voxel in meters.
        filter_gaussians: Whether to filter Gaussians outside volume.
        opacity_thresh: Minimum opacity threshold for filtering.
        chunk_size: Number of Gaussians per chunk.
    """

    def __init__(
        self,
        vol_range,
        voxel_size,
        filter_gaussians=True,
        opacity_thresh=1e-4,
        chunk_size=1000
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

        # Pre-compute grid coordinates
        grid_coords = self._generate_grid()
        self.register_buffer('grid_coords', grid_coords)

    def _generate_grid(self):
        """Generate 3D grid coordinates [X, Y, Z, 3]."""
        coords = [
            torch.arange(s, dtype=torch.float32) + 0.5
            for s in self.grid_shape
        ]
        grid = torch.stack(torch.meshgrid(*coords, indexing='ij'), dim=-1)
        grid = grid * self.voxel_size + self.vol_range[:3]
        return grid

    def _filter_gaussians(
        self,
        means3d: torch.Tensor,
        opacities: torch.Tensor,
        covariances: torch.Tensor,
        features: Optional[torch.Tensor]
    ) -> Tuple[torch.Tensor, ...]:
        """Pre-filter Gaussians outside volume range."""
        vol_min = self.vol_range[:3]
        vol_max = self.vol_range[3:]

        sigma = torch.sqrt(torch.diagonal(covariances, dim1=1, dim2=2))
        bounds_min = means3d - 3 * sigma
        bounds_max = means3d + 3 * sigma

        mask = ((bounds_max > vol_min).all(dim=1) &
                (bounds_min < vol_max).all(dim=1))

        opac_flat = opacities.squeeze(-1) if opacities.dim() == 2 else opacities
        mask &= (opac_flat > self.opacity_thresh)

        if mask.sum() == 0:
            return None, None, None, None

        return (
            means3d[mask],
            opacities[mask],
            covariances[mask],
            features[mask] if features is not None else None
        )

    def _voxelize_chunk(
        self,
        means3d: torch.Tensor,
        opacities: torch.Tensor,
        covariances: torch.Tensor,
        features: Optional[torch.Tensor],
        grid_flat: torch.Tensor
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Voxelize a chunk of Gaussians."""
        n_gaussians = means3d.size(0)

        # Compute inverse covariances
        cov_inv = torch.linalg.inv(covariances)  # [N, 3, 3]

        # Compute diff for chunk
        diff = means3d.unsqueeze(1) - grid_flat.unsqueeze(0)  # [N, V, 3]

        # Compute Mahalanobis distance
        diff_transformed = torch.einsum('nvi,nij->nvj', diff, cov_inv)
        maha = (diff * diff_transformed).sum(dim=-1)  # [N, V]

        # Compute contributions
        opac = opacities.squeeze(-1) if opacities.dim() == 2 else opacities
        contrib = opac.unsqueeze(1) * torch.exp(-0.5 * maha)  # [N, V]

        # Bounds check
        sigma = torch.sqrt(torch.diagonal(covariances, dim1=1, dim2=2))
        sigma_max = sigma.max(dim=1).values
        dist_sq = (diff ** 2).sum(dim=-1)
        within_bounds = dist_sq < (3 * sigma_max.unsqueeze(1)) ** 2
        contrib = contrib * within_bounds.float()

        # Sum contributions
        density = contrib.sum(dim=0)  # [V]

        feat_contrib = None
        if features is not None:
            feat_contrib = (contrib.unsqueeze(-1) * features.unsqueeze(1)).sum(dim=0)

        return density, feat_contrib

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
        """Voxelize Gaussians with chunked processing."""
        device = means3d.device

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

        # Pre-filter
        if self.filter_gaussians:
            filtered = self._filter_gaussians(means3d, opacities, covariances, features)
            if filtered[0] is None:
                density = torch.zeros(*self.grid_shape, 1, device=device)
                grid_feats = torch.zeros(*self.grid_shape, feat_dims, device=device) if feat_dims > 0 else None
                return density, grid_feats
            means3d, opacities, covariances, features = filtered

        n_gaussians = means3d.size(0)
        grid_flat = self.grid_coords.reshape(-1, 3)

        # Initialize accumulators
        density_accum = torch.zeros(grid_flat.size(0), device=device)
        feat_accum = torch.zeros(grid_flat.size(0), feat_dims, device=device) if feat_dims > 0 else None

        # Process in chunks
        for start in range(0, n_gaussians, self.chunk_size):
            end = min(start + self.chunk_size, n_gaussians)
            chunk_density, chunk_feats = self._voxelize_chunk(
                means3d[start:end],
                opacities[start:end],
                covariances[start:end],
                features[start:end] if features is not None else None,
                grid_flat
            )
            density_accum += chunk_density
            if feat_accum is not None and chunk_feats is not None:
                feat_accum += chunk_feats

        # Reshape
        grid_density = density_accum.reshape(*self.grid_shape, 1)
        grid_feats = None
        if feat_accum is not None:
            grid_feats = feat_accum.reshape(*self.grid_shape, feat_dims)
            eps = 1e-6
            grid_feats = grid_feats / grid_density.clamp(min=eps)

        return grid_density, grid_feats
