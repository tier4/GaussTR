"""CUDA-accelerated Gaussian voxelizer.

GPU-accelerated voxelization using custom CUDA kernels.
Multi-GPU DDP compatible with no global state issues.

Uses GaussianFormer-style precision matrix representation (6 elements)
and optimized Gaussian evaluation formula.
Reference: https://github.com/huang-yh/GaussianFormer
"""

import torch
import torch.nn as nn
from typing import Tuple, Optional, List
from pathlib import Path
from torch import Tensor

# Try to import compiled CUDA extension
_cuda_ext = None
_cuda_available = False
_ops_registered = False


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
        _register_custom_ops()
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
            _register_custom_ops()
            print("[CUDAVoxelizer] CUDA extension compiled successfully")
            return _cuda_ext
    except Exception as e:
        print(f"[CUDAVoxelizer] Failed to compile CUDA extension: {e}")

    _cuda_available = False
    return None


def _register_custom_ops():
    """Register voxelizer ops for torch.compile compatibility."""
    global _ops_registered, _voxelizer_lib, _voxelizer_impl_lib
    if _ops_registered:
        return
    _ops_registered = True

    # Define the library for our custom ops (stored globally to prevent GC)
    _voxelizer_lib = torch.library.Library("voxelizer", "DEF")

    # Define op schema - note: List[int] for grid_shape, List[float] for vol_range and voxel_size
    _voxelizer_lib.define(
        "voxelize_gaussians(Tensor means3d, Tensor opacities, Tensor cov3D, Tensor radii, "
        "Tensor features, int[] grid_shape, float[] vol_range, float[] voxel_size, float eps) -> (Tensor, Tensor)"
    )

    # Create implementation library (stored globally to prevent GC)
    _voxelizer_impl_lib = torch.library.Library("voxelizer", "IMPL")

    # Register CUDA implementation
    def _voxelize_gaussians_cuda(
        means3d: Tensor,
        opacities: Tensor,
        cov3D: Tensor,
        radii: Tensor,
        features: Tensor,
        grid_shape: List[int],
        vol_range: List[float],
        voxel_size: List[float],
        eps: float
    ) -> Tuple[Tensor, Tensor]:
        return _cuda_ext.voxelize_gaussians(
            means3d, opacities, cov3D, radii, features,
            grid_shape, vol_range, voxel_size, eps
        )

    _voxelizer_impl_lib.impl("voxelize_gaussians", _voxelize_gaussians_cuda, "CUDA")

    # Register abstract (fake tensor) implementation for torch.compile tracing
    @torch.library.register_fake("voxelizer::voxelize_gaussians")
    def _voxelize_gaussians_fake(
        means3d: Tensor,
        opacities: Tensor,
        cov3D: Tensor,
        radii: Tensor,
        features: Tensor,
        grid_shape: List[int],
        vol_range: List[float],
        voxel_size: List[float],
        eps: float
    ) -> Tuple[Tensor, Tensor]:
        # Output shapes: [X, Y, Z, 1] for density, [X, Y, Z, feat_dim] for features
        X, Y, Z = grid_shape
        feat_dim = features.shape[-1] if features.numel() > 0 else 0

        grid_density = means3d.new_empty(X, Y, Z, 1)
        if feat_dim > 0:
            grid_feats = means3d.new_empty(X, Y, Z, feat_dim)
        else:
            grid_feats = means3d.new_empty(0, 0)

        return grid_density, grid_feats


# Global variables to store library objects and prevent garbage collection
_voxelizer_lib = None
_voxelizer_impl_lib = None


def _filter_gaussians_torch(
    means3d: torch.Tensor,
    opacities: torch.Tensor,
    cov3D: torch.Tensor,
    radii: torch.Tensor,
    features: Optional[torch.Tensor],
    vol_range: torch.Tensor,
    opacity_thresh: float = 1e-4
) -> Tuple[torch.Tensor, ...]:
    """Pre-filter Gaussians outside volume range (vectorized PyTorch)."""
    vol_min = vol_range[:3]
    vol_max = vol_range[3:]

    # Check if Gaussian is inside or overlapping volume (using simple mean check)
    mask = ((means3d >= vol_min).all(dim=1) & (means3d <= vol_max).all(dim=1))

    # Also filter low opacity
    opac_flat = opacities.squeeze(-1) if opacities.dim() == 2 else opacities
    mask &= (opac_flat > opacity_thresh)

    # Filter out zero radii
    mask &= (radii > 0)

    if mask.sum() == 0:
        return None, None, None, None, None

    return (
        means3d[mask],
        opacities[mask],
        cov3D[mask],
        radii[mask],
        features[mask] if features is not None else None
    )


class CUDAVoxelizer(nn.Module):
    """CUDA-accelerated Gaussian voxelizer.

    Uses custom CUDA kernels for maximum GPU utilization.
    No Python for-loops over Gaussians.
    Multi-GPU DDP compatible (no global state).

    Uses GaussianFormer-style precision matrix format (6 elements):
    [Λ_xx, Λ_yy, Λ_zz, Λ_xy, Λ_yz, Λ_xz]

    Args:
        vol_range: Volume range [xmin, ymin, zmin, xmax, ymax, zmax].
        voxel_size: Size of each voxel in meters.
        filter_gaussians: Whether to filter Gaussians outside volume.
        opacity_thresh: Minimum opacity threshold for filtering.
        sigma_factor: Number of sigmas to consider for each Gaussian (scale multiplier).
        eps: Small value for numerical stability.
    """

    def __init__(
        self,
        vol_range,
        voxel_size,
        filter_gaussians=False,  # Default False to match original GaussTR
        opacity_thresh=0.0,  # Default 0 to match original GaussTR
        sigma_factor=3.0,
        eps=1e-6
    ):
        super().__init__()
        self.sigma_factor = sigma_factor
        self.eps = eps
        vol_range = torch.tensor(vol_range, dtype=torch.float32)
        self.register_buffer('vol_range', vol_range)

        # Handle voxel_size: scalar -> [v, v, v], list -> tensor
        if isinstance(voxel_size, (int, float)):
            voxel_size = torch.tensor([voxel_size, voxel_size, voxel_size], dtype=torch.float32)
        elif isinstance(voxel_size, (list, tuple)):
            voxel_size = torch.tensor(voxel_size, dtype=torch.float32)
        elif isinstance(voxel_size, torch.Tensor) and voxel_size.dim() == 0:
            # Scalar tensor -> expand to 3D
            voxel_size = voxel_size.float().expand(3).clone()
        self.register_buffer('voxel_size', voxel_size)

        # Compute grid shape
        self.grid_shape = ((vol_range[3:] - vol_range[:3]) / voxel_size).int().tolist()
        self.filter_gaussians = filter_gaussians
        self.opacity_thresh = opacity_thresh

        # Cache indices for precision matrix packing (avoids tensor creation every forward)
        self.register_buffer('_cov_indices', torch.tensor([0, 4, 8, 1, 5, 2], dtype=torch.long))

        # Check CUDA availability
        self._cuda_ext = None

    def _prepare_gaussian_args(
        self,
        scales: torch.Tensor,
        rotations: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute precision matrix and radii from scales and rotations.

        Steps:
        1. Compute covariance: Cov = R @ S @ S^T @ R^T = (R @ S) @ (R @ S)^T
        2. Invert to get precision matrix: CovInv = Cov^(-1)
        3. Pack as 6 elements: [Λ_xx, Λ_yy, Λ_zz, Λ_xy, Λ_yz, Λ_xz]
        4. Compute radii from scales

        Args:
            scales: [N, 3] Gaussian scales
            rotations: [N, 4] Gaussian rotations (quaternions)

        Returns:
            cov3D: [N, 6] Packed precision matrix
            radii: [N] Voxel radii for each Gaussian
        """
        n_gaussians = scales.size(0)
        device = scales.device
        dtype = scales.dtype

        # Normalize quaternions
        q = rotations / torch.sqrt((rotations**2).sum(dim=-1, keepdim=True) + 1e-8)
        r, x, y, z = q.unbind(-1)

        # Build rotation matrix R
        R = torch.zeros(n_gaussians, 3, 3, device=device, dtype=dtype)
        R[:, 0, 0] = 1 - 2 * (y * y + z * z)
        R[:, 0, 1] = 2 * (x * y - r * z)
        R[:, 0, 2] = 2 * (x * z + r * y)
        R[:, 1, 0] = 2 * (x * y + r * z)
        R[:, 1, 1] = 1 - 2 * (x * x + z * z)
        R[:, 1, 2] = 2 * (y * z - r * x)
        R[:, 2, 0] = 2 * (x * z - r * y)
        R[:, 2, 1] = 2 * (y * z + r * x)
        R[:, 2, 2] = 1 - 2 * (x * x + y * y)

        # Build scale matrix S (diagonal)
        S = torch.zeros(n_gaussians, 3, 3, device=device, dtype=dtype)
        S[:, 0, 0] = scales[:, 0]
        S[:, 1, 1] = scales[:, 1]
        S[:, 2, 2] = scales[:, 2]

        # Compute M = R @ S
        M = torch.bmm(R, S)

        # Compute covariance: Cov = M @ M^T = R @ S² @ R^T
        Cov = torch.bmm(M, M.transpose(-1, -2))

        # Invert covariance to get precision matrix (GaussianFormer style)
        # Use FP32 on GPU for numerical stability (disable autocast to avoid FP16)
        with torch.amp.autocast('cuda', enabled=False):
            CovInv = torch.linalg.inv(Cov.float()).to(dtype=dtype)

        # Pack as 6 elements: [Λ_xx, Λ_yy, Λ_zz, Λ_xy, Λ_yz, Λ_xz]
        cov3D = CovInv.flatten(1).index_select(1, self._cov_indices)

        # Compute radii: ceil(max_scale * sigma_factor / min_voxel_size)
        # Use minimum voxel size to be conservative (covers all dimensions)
        min_voxel_size = self.voxel_size.min() if isinstance(self.voxel_size, torch.Tensor) else min(self.voxel_size)
        radii = torch.ceil(
            scales.max(dim=-1)[0] * self.sigma_factor / min_voxel_size
        ).to(torch.int32)
        # Ensure radii >= 1
        radii = torch.clamp(radii, min=1)

        return cov3D, radii

    def _unbatched_forward(
        self,
        means3d: torch.Tensor,
        opacities: torch.Tensor,
        cov3D: torch.Tensor,
        radii: torch.Tensor,
        features: Optional[torch.Tensor]
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Forward pass for a single sample (no batch dimension)."""
        device = means3d.device

        # Pre-filter Gaussians
        if self.filter_gaussians:
            filtered = _filter_gaussians_torch(
                means3d, opacities, cov3D, radii, features,
                self.vol_range, self.opacity_thresh
            )
            if filtered[0] is None:
                # No valid Gaussians
                feat_dim = features.size(-1) if features is not None else 0
                density = torch.zeros(*self.grid_shape, 1, device=device)
                grid_feats = torch.zeros(*self.grid_shape, feat_dim, device=device) if feat_dim > 0 else None
                return density, grid_feats
            means3d, opacities, cov3D, radii, features = filtered

        # Ensure tensors are contiguous
        means3d = means3d.float().contiguous()
        opacities = opacities.float().contiguous()
        cov3D = cov3D.float().contiguous()
        radii = radii.int().contiguous()
        if features is not None:
            features = features.float().contiguous()
        else:
            features = torch.empty((0, 0), device=device, dtype=torch.float32)

        # Ensure CUDA extension is loaded and ops registered
        if self._cuda_ext is None:
            self._cuda_ext = _get_cuda_ext()

        if self._cuda_ext is not None and means3d.is_cuda:
            vol_range_list = self.vol_range.tolist()
            voxel_size_list = self.voxel_size.tolist()
            grid_density, grid_feats = torch.ops.voxelizer.voxelize_gaussians(
                means3d,
                opacities,
                cov3D,
                radii,
                features,
                self.grid_shape,
                vol_range_list,
                voxel_size_list,
                self.eps
            )
            if grid_feats.numel() == 0:
                grid_feats = None
        else:
            # Fallback to PyTorch implementation
            grid_density, grid_feats = self._pytorch_fallback(
                means3d, opacities, cov3D, radii, features
            )

        return grid_density, grid_feats

    def _pytorch_fallback(
        self,
        means3d: torch.Tensor,
        opacities: torch.Tensor,
        cov3D: torch.Tensor,
        radii: torch.Tensor,
        features: Optional[torch.Tensor]
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """PyTorch fallback when CUDA is not available.

        Uses GaussianFormer-style Gaussian evaluation formula.
        """
        device = means3d.device
        n_gaussians = means3d.size(0)
        vol_min = self.vol_range[:3]

        # Initialize grids
        grid_density = torch.zeros(*self.grid_shape, 1, device=device)
        feat_dim = features.size(-1) if features is not None and features.numel() > 0 else 0
        grid_feats = torch.zeros(*self.grid_shape, feat_dim, device=device) if feat_dim > 0 else None

        # Pre-convert radii to CPU list to avoid GPU-CPU sync per iteration
        radii_list = radii.cpu().tolist()

        for g in range(n_gaussians):
            mean = means3d[g]
            radius = radii_list[g]

            # Get precision matrix elements from pre-loaded CPU tensor
            cov_xx = cov3D[g, 0]
            cov_yy = cov3D[g, 1]
            cov_zz = cov3D[g, 2]
            cov_xy = cov3D[g, 3]
            cov_yz = cov3D[g, 4]
            cov_xz = cov3D[g, 5]

            # Compute voxel index of mean
            mean_idx = ((mean - vol_min) / self.voxel_size).int()

            # Compute bounds (use integer radius to avoid sync)
            idx_min = (mean_idx - radius).clamp(min=0)
            idx_max = (mean_idx + radius + 1).clamp(
                max=torch.tensor(self.grid_shape, device=device)
            ) - 1

            i_min, j_min, k_min = idx_min.cpu().tolist()
            i_max, j_max, k_max = idx_max.cpu().tolist()

            if i_min > i_max or j_min > j_max or k_min > k_max:
                continue

            # Generate voxel centers for this region
            i_range = torch.arange(i_min, i_max + 1, device=device, dtype=torch.float32)
            j_range = torch.arange(j_min, j_max + 1, device=device, dtype=torch.float32)
            k_range = torch.arange(k_min, k_max + 1, device=device, dtype=torch.float32)

            ii, jj, kk = torch.meshgrid(i_range, j_range, k_range, indexing='ij')
            voxel_centers = torch.stack([ii, jj, kk], dim=-1)
            voxel_centers = voxel_centers * self.voxel_size + vol_min + 0.5 * self.voxel_size

            # Compute diff: d = mean - voxel (GaussianFormer convention)
            diff = mean - voxel_centers  # [ni, nj, nk, 3]
            dx, dy, dz = diff[..., 0], diff[..., 1], diff[..., 2]

            # GaussianFormer-style Gaussian evaluation:
            # power = -0.5 * (Λ_xx*dx² + Λ_yy*dy² + Λ_zz*dz²) - (Λ_xy*dx*dy + Λ_yz*dy*dz + Λ_xz*dx*dz)
            power = cov_xx * dx * dx + cov_yy * dy * dy + cov_zz * dz * dz
            power = -0.5 * power - (cov_xy * dx * dy + cov_yz * dy * dz + cov_xz * dx * dz)

            # Compute contribution
            opac = opacities[g].squeeze() if opacities[g].dim() > 0 else opacities[g]
            contrib = opac * torch.exp(power)

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
            covariances: DEPRECATED. Use scales and rotations instead.
            scales: Gaussian scales [B, N, 3].
            rotations: Gaussian rotations [B, N, 4] (quaternions).
            features: Gaussian features [B, N, C].

        Returns:
            Tuple of:
                - density: [B, X, Y, Z, 1]
                - features: [B, X, Y, Z, C] or None
        """
        batch_size = means3d.size(0)
        n_gaussians = means3d.size(1)

        if scales is None or rotations is None:
            raise ValueError(
                "scales and rotations are required. "
                "The covariances argument is deprecated."
            )

        # Batch compute precision matrix and radii for ALL gaussians at once
        # This avoids kernel launch overhead from per-batch processing
        cov3D_all, radii_all = self._prepare_gaussian_args(
            scales.reshape(-1, 3),  # [B*N, 3]
            rotations.reshape(-1, 4)  # [B*N, 4]
        )
        # Reshape back to per-batch
        cov3D_all = cov3D_all.reshape(batch_size, n_gaussians, 6)
        radii_all = radii_all.reshape(batch_size, n_gaussians)

        # Process each batch element (CUDA kernel requires unbatched input)
        densities = []
        feats_list = []

        for b in range(batch_size):
            density, feats = self._unbatched_forward(
                means3d[b],
                opacities[b],
                cov3D_all[b],
                radii_all[b],
                features[b] if features is not None else None
            )
            densities.append(density)
            if feats is not None:
                feats_list.append(feats)

        # Stack results
        grid_density = torch.stack(densities, dim=0)
        grid_features = torch.stack(feats_list, dim=0) if feats_list else None

        return grid_density, grid_features
