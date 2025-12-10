"""CUDA-accelerated Gaussian voxelizer.

GPU-accelerated voxelization using custom CUDA kernels.
Multi-GPU DDP compatible with no global state issues.
"""

import os
import torch
import torch.nn as nn
from typing import Tuple, Optional
from pathlib import Path

# Try to import compiled CUDA extension
_cuda_ext = None
_cuda_available = False


def _get_cuda_ext():
    """Lazily load CUDA extension with JIT compilation fallback."""
    global _cuda_ext, _cuda_available

    if _cuda_ext is not None:
        return _cuda_ext

    # Try loading pre-compiled extension
    try:
        from . import voxelize_cuda_ext
        _cuda_ext = voxelize_cuda_ext
        _cuda_available = True
        return _cuda_ext
    except ImportError:
        pass

    # Try JIT compilation
    try:
        from torch.utils.cpp_extension import load

        cuda_src = Path(__file__).parent / "voxelize_cuda.cu"
        if cuda_src.exists():
            print("[CUDAVoxelizer] JIT compiling CUDA extension...")
            _cuda_ext = load(
                name="voxelize_cuda_ext",
                sources=[str(cuda_src)],
                extra_cuda_cflags=["-O3", "--use_fast_math"],
                verbose=False,
            )
            _cuda_available = True
            print("[CUDAVoxelizer] CUDA extension compiled successfully")
            return _cuda_ext
    except Exception as e:
        print(f"[CUDAVoxelizer] Failed to compile CUDA extension: {e}")

    _cuda_available = False
    return None


def _filter_gaussians_torch(
    means3d: torch.Tensor,
    opacities: torch.Tensor,
    covariances: torch.Tensor,
    features: Optional[torch.Tensor],
    vol_range: torch.Tensor,
    opacity_thresh: float = 1e-4,
    sigma_factor: float = 3.0
) -> Tuple[torch.Tensor, ...]:
    """Pre-filter Gaussians outside volume range (vectorized PyTorch)."""
    vol_min = vol_range[:3]
    vol_max = vol_range[3:]

    # Get sigma bounds for each Gaussian
    sigma = torch.sqrt(torch.diagonal(covariances, dim1=1, dim2=2))  # [N, 3]
    bounds_min = means3d - sigma_factor * sigma
    bounds_max = means3d + sigma_factor * sigma

    # Check if Gaussian bounds overlap with volume
    mask = ((bounds_max > vol_min).all(dim=1) &
            (bounds_min < vol_max).all(dim=1))

    # Also filter low opacity
    opac_flat = opacities.squeeze(-1) if opacities.dim() == 2 else opacities
    mask &= (opac_flat > opacity_thresh)

    if mask.sum() == 0:
        return None, None, None, None

    return (
        means3d[mask],
        opacities[mask],
        covariances[mask],
        features[mask] if features is not None else None
    )


class CUDAVoxelizer(nn.Module):
    """CUDA-accelerated Gaussian voxelizer.

    Uses custom CUDA kernels for maximum GPU utilization.
    No Python for-loops over Gaussians.
    Multi-GPU DDP compatible (no global state).

    Args:
        vol_range: Volume range [xmin, ymin, zmin, xmax, ymax, zmax].
        voxel_size: Size of each voxel in meters.
        filter_gaussians: Whether to filter Gaussians outside volume.
        opacity_thresh: Minimum opacity threshold for filtering.
        sigma_factor: Number of sigmas to consider for each Gaussian.
        eps: Small value for numerical stability.
    """

    def __init__(
        self,
        vol_range,
        voxel_size,
        filter_gaussians=True,
        opacity_thresh=1e-4,
        sigma_factor=3.0,
        eps=1e-6
    ):
        super().__init__()
        self.voxel_size = voxel_size
        self.sigma_factor = sigma_factor
        self.eps = eps
        vol_range = torch.tensor(vol_range, dtype=torch.float32)
        self.register_buffer('vol_range', vol_range)

        # Compute grid shape
        self.grid_shape = ((vol_range[3:] - vol_range[:3]) / voxel_size).int().tolist()
        self.filter_gaussians = filter_gaussians
        self.opacity_thresh = opacity_thresh

        # Check CUDA availability
        self._cuda_ext = None

    def _get_covariance(
        self,
        scales: torch.Tensor,
        rotations: torch.Tensor
    ) -> torch.Tensor:
        """Compute covariance matrices from scales and rotations."""
        # Convert quaternions to rotation matrices
        q = rotations / torch.sqrt((rotations**2).sum(dim=-1, keepdim=True) + 1e-8)
        r, x, y, z = q.unbind(-1)

        R = torch.zeros((*r.shape, 3, 3), device=rotations.device, dtype=rotations.dtype)
        R[..., 0, 0] = 1 - 2 * (y * y + z * z)
        R[..., 0, 1] = 2 * (x * y - r * z)
        R[..., 0, 2] = 2 * (x * z + r * y)
        R[..., 1, 0] = 2 * (x * y + r * z)
        R[..., 1, 1] = 1 - 2 * (x * x + z * z)
        R[..., 1, 2] = 2 * (y * z - r * x)
        R[..., 2, 0] = 2 * (x * z - r * y)
        R[..., 2, 1] = 2 * (y * z + r * x)
        R[..., 2, 2] = 1 - 2 * (x * x + y * y)

        # Create diagonal scale matrix
        L = torch.diag_embed(scales)  # [N, 3, 3]

        # Covariance = R @ L @ L^T @ R^T = R @ S @ R^T where S = L @ L^T
        L = R @ L
        covariance = L @ L.transpose(-1, -2)
        return covariance

    def _unbatched_forward(
        self,
        means3d: torch.Tensor,
        opacities: torch.Tensor,
        covariances: torch.Tensor,
        features: Optional[torch.Tensor]
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Forward pass for a single sample (no batch dimension)."""
        device = means3d.device

        # Pre-filter Gaussians
        if self.filter_gaussians:
            filtered = _filter_gaussians_torch(
                means3d, opacities, covariances, features,
                self.vol_range, self.opacity_thresh, self.sigma_factor
            )
            if filtered[0] is None:
                # No valid Gaussians
                feat_dim = features.size(-1) if features is not None else 0
                density = torch.zeros(*self.grid_shape, 1, device=device)
                grid_feats = torch.zeros(*self.grid_shape, feat_dim, device=device) if feat_dim > 0 else None
                return density, grid_feats
            means3d, opacities, covariances, features = filtered

        # Ensure tensors are contiguous
        means3d = means3d.float().contiguous()
        opacities = opacities.float().contiguous()
        covariances = covariances.float().contiguous()
        if features is not None:
            features = features.float().contiguous()
        else:
            features = torch.empty((0, 0), device=device, dtype=torch.float32)

        # Get CUDA extension
        if self._cuda_ext is None:
            self._cuda_ext = _get_cuda_ext()

        if self._cuda_ext is not None and means3d.is_cuda:
            # Use CUDA kernel
            vol_range_list = self.vol_range.tolist()
            grid_density, grid_feats = self._cuda_ext.voxelize_gaussians(
                means3d,
                opacities,
                covariances,
                features,
                self.grid_shape,
                vol_range_list,
                self.voxel_size,
                self.sigma_factor,
                self.eps
            )
            if grid_feats.numel() == 0:
                grid_feats = None
        else:
            # Fallback to PyTorch implementation
            grid_density, grid_feats = self._pytorch_fallback(
                means3d, opacities, covariances, features
            )

        return grid_density, grid_feats

    def _pytorch_fallback(
        self,
        means3d: torch.Tensor,
        opacities: torch.Tensor,
        covariances: torch.Tensor,
        features: Optional[torch.Tensor]
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """PyTorch fallback when CUDA is not available."""
        device = means3d.device
        n_gaussians = means3d.size(0)
        vol_min = self.vol_range[:3]

        # Initialize grids
        grid_density = torch.zeros(*self.grid_shape, 1, device=device)
        feat_dim = features.size(-1) if features is not None and features.numel() > 0 else 0
        grid_feats = torch.zeros(*self.grid_shape, feat_dim, device=device) if feat_dim > 0 else None

        # Compute inverse covariances
        cov_inv = torch.linalg.inv(covariances)  # [N, 3, 3]

        # Get sigma for bounds
        sigma = torch.sqrt(torch.diagonal(covariances, dim1=1, dim2=2))  # [N, 3]

        for g in range(n_gaussians):
            mean = means3d[g]
            sig = sigma[g]

            # Compute voxel index bounds
            idx_min = ((mean - self.sigma_factor * sig - vol_min) / self.voxel_size).int()
            idx_max = ((mean + self.sigma_factor * sig - vol_min) / self.voxel_size).int()

            idx_min = idx_min.clamp(min=0)
            idx_max = torch.tensor(self.grid_shape, device=device) - 1
            idx_max = idx_max.clamp(max=torch.tensor(self.grid_shape, device=device) - 1)

            i_min, j_min, k_min = idx_min.tolist()
            i_max, j_max, k_max = ((mean + self.sigma_factor * sig - vol_min) / self.voxel_size).int().clamp(
                max=torch.tensor(self.grid_shape, device=device) - 1
            ).tolist()

            if i_min > i_max or j_min > j_max or k_min > k_max:
                continue

            # Generate voxel centers for this region
            i_range = torch.arange(i_min, i_max + 1, device=device, dtype=torch.float32)
            j_range = torch.arange(j_min, j_max + 1, device=device, dtype=torch.float32)
            k_range = torch.arange(k_min, k_max + 1, device=device, dtype=torch.float32)

            ii, jj, kk = torch.meshgrid(i_range, j_range, k_range, indexing='ij')
            voxel_centers = torch.stack([ii, jj, kk], dim=-1)
            voxel_centers = voxel_centers * self.voxel_size + vol_min + 0.5 * self.voxel_size

            # Compute Mahalanobis distance
            diff = voxel_centers - mean  # [ni, nj, nk, 3]
            diff_flat = diff.reshape(-1, 3)  # [n_voxels, 3]
            maha = (diff_flat @ cov_inv[g] * diff_flat).sum(dim=-1)  # [n_voxels]
            maha = maha.reshape(ii.shape)  # [ni, nj, nk]

            # Compute contribution
            opac = opacities[g].squeeze() if opacities[g].dim() > 0 else opacities[g]
            contrib = opac * torch.exp(-0.5 * maha)

            # Update grid
            grid_density[i_min:i_max+1, j_min:j_max+1, k_min:k_max+1, 0] += contrib

            if grid_feats is not None and features is not None and features.numel() > 0:
                feat_contrib = contrib.unsqueeze(-1) * features[g]
                grid_feats[i_min:i_max+1, j_min:j_max+1, k_min:k_max+1] += feat_contrib

        # Normalize features
        if grid_feats is not None:
            grid_feats = grid_feats / grid_density.clamp(min=self.eps)

        return grid_density, grid_feats

    def forward(
        self,
        means3d: torch.Tensor,
        opacities: torch.Tensor,
        covariances: Optional[torch.Tensor] = None,
        scales: Optional[torch.Tensor] = None,
        rotations: Optional[torch.Tensor] = None,
        features: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Voxelize Gaussians.

        Args:
            means3d: Gaussian centers [B, N, 3].
            opacities: Gaussian opacities [B, N, 1].
            covariances: Gaussian covariance matrices [B, N, 3, 3]. Optional.
            scales: Gaussian scales [B, N, 3]. Used if covariances not provided.
            rotations: Gaussian rotations [B, N, 4]. Used if covariances not provided.
            features: Gaussian features [B, N, C].

        Returns:
            Tuple of:
                - density: [B, X, Y, Z, 1]
                - features: [B, X, Y, Z, C] or None
        """
        batch_size = means3d.size(0)

        # Compute covariances if not provided
        if covariances is None:
            # Flatten batch for covariance computation
            scales_flat = scales.reshape(-1, 3)
            rotations_flat = rotations.reshape(-1, 4)
            covariances = self._get_covariance(scales_flat, rotations_flat)
            covariances = covariances.reshape(batch_size, -1, 3, 3)

        # Process each batch element
        densities = []
        feats_list = []

        for b in range(batch_size):
            density, feats = self._unbatched_forward(
                means3d[b],
                opacities[b],
                covariances[b],
                features[b] if features is not None else None
            )
            densities.append(density)
            if feats is not None:
                feats_list.append(feats)

        # Stack results
        grid_density = torch.stack(densities, dim=0)
        grid_features = torch.stack(feats_list, dim=0) if feats_list else None

        return grid_density, grid_features
