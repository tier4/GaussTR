/*
 * CUDA kernels for Gaussian voxelization.
 *
 * This implements efficient GPU-accelerated splatting of 3D Gaussians
 * into a voxel grid for occupancy prediction.
 */

#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <cmath>

#define THREADS_PER_BLOCK 256

// Helper: 3x3 matrix inverse (for covariance)
__device__ void mat3_inverse(const float* __restrict__ mat, float* __restrict__ inv) {
    // mat is row-major [3x3]
    float a = mat[0], b = mat[1], c = mat[2];
    float d = mat[3], e = mat[4], f = mat[5];
    float g = mat[6], h = mat[7], i = mat[8];

    float det = a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g);
    float inv_det = 1.0f / (det + 1e-10f);

    inv[0] = (e * i - f * h) * inv_det;
    inv[1] = (c * h - b * i) * inv_det;
    inv[2] = (b * f - c * e) * inv_det;
    inv[3] = (f * g - d * i) * inv_det;
    inv[4] = (a * i - c * g) * inv_det;
    inv[5] = (c * d - a * f) * inv_det;
    inv[6] = (d * h - e * g) * inv_det;
    inv[7] = (b * g - a * h) * inv_det;
    inv[8] = (a * e - b * d) * inv_det;
}

// Kernel: Voxelize Gaussians (one thread per Gaussian)
// Each thread processes one Gaussian and atomically updates affected voxels
template <int FEAT_DIM>
__global__ void voxelize_gaussians_kernel(
    const float* __restrict__ means3d,      // [N, 3]
    const float* __restrict__ opacities,    // [N, 1]
    const float* __restrict__ covariances,  // [N, 3, 3]
    const float* __restrict__ features,     // [N, FEAT_DIM] or nullptr
    float* __restrict__ grid_density,       // [X, Y, Z]
    float* __restrict__ grid_features,      // [X, Y, Z, FEAT_DIM] or nullptr
    int n_gaussians,
    int grid_x, int grid_y, int grid_z,
    float vol_min_x, float vol_min_y, float vol_min_z,
    float voxel_size,
    float sigma_factor
) {
    int g = blockIdx.x * blockDim.x + threadIdx.x;
    if (g >= n_gaussians) return;

    // Load Gaussian parameters
    float mean_x = means3d[g * 3 + 0];
    float mean_y = means3d[g * 3 + 1];
    float mean_z = means3d[g * 3 + 2];
    float opacity = opacities[g];

    // Load and invert covariance
    float cov[9], cov_inv[9];
    #pragma unroll
    for (int i = 0; i < 9; i++) {
        cov[i] = covariances[g * 9 + i];
    }
    mat3_inverse(cov, cov_inv);

    // Compute sigma (sqrt of diagonal)
    float sigma_x = sqrtf(cov[0]);
    float sigma_y = sqrtf(cov[4]);
    float sigma_z = sqrtf(cov[8]);

    // Compute bounds (N-sigma)
    float bound_min_x = mean_x - sigma_factor * sigma_x;
    float bound_max_x = mean_x + sigma_factor * sigma_x;
    float bound_min_y = mean_y - sigma_factor * sigma_y;
    float bound_max_y = mean_y + sigma_factor * sigma_y;
    float bound_min_z = mean_z - sigma_factor * sigma_z;
    float bound_max_z = mean_z + sigma_factor * sigma_z;

    // Convert to voxel indices
    int i_min = max(0, (int)floorf((bound_min_x - vol_min_x) / voxel_size));
    int i_max = min(grid_x - 1, (int)floorf((bound_max_x - vol_min_x) / voxel_size));
    int j_min = max(0, (int)floorf((bound_min_y - vol_min_y) / voxel_size));
    int j_max = min(grid_y - 1, (int)floorf((bound_max_y - vol_min_y) / voxel_size));
    int k_min = max(0, (int)floorf((bound_min_z - vol_min_z) / voxel_size));
    int k_max = min(grid_z - 1, (int)floorf((bound_max_z - vol_min_z) / voxel_size));

    // Load features if provided
    float feat[FEAT_DIM];
    if (features != nullptr && FEAT_DIM > 0) {
        #pragma unroll
        for (int f = 0; f < FEAT_DIM; f++) {
            feat[f] = features[g * FEAT_DIM + f];
        }
    }

    // Iterate over affected voxels
    for (int i = i_min; i <= i_max; i++) {
        for (int j = j_min; j <= j_max; j++) {
            for (int k = k_min; k <= k_max; k++) {
                // Compute voxel center
                float vx = vol_min_x + (i + 0.5f) * voxel_size;
                float vy = vol_min_y + (j + 0.5f) * voxel_size;
                float vz = vol_min_z + (k + 0.5f) * voxel_size;

                // Compute diff
                float dx = vx - mean_x;
                float dy = vy - mean_y;
                float dz = vz - mean_z;

                // Compute Mahalanobis distance: d^T * cov_inv * d
                float maha = dx * (cov_inv[0] * dx + cov_inv[1] * dy + cov_inv[2] * dz) +
                             dy * (cov_inv[3] * dx + cov_inv[4] * dy + cov_inv[5] * dz) +
                             dz * (cov_inv[6] * dx + cov_inv[7] * dy + cov_inv[8] * dz);

                // Compute contribution
                float contrib = opacity * expf(-0.5f * maha);

                // Atomic add to density
                int voxel_idx = i * grid_y * grid_z + j * grid_z + k;
                atomicAdd(&grid_density[voxel_idx], contrib);

                // Atomic add to features (weighted)
                if (grid_features != nullptr && FEAT_DIM > 0) {
                    #pragma unroll
                    for (int f = 0; f < FEAT_DIM; f++) {
                        atomicAdd(&grid_features[voxel_idx * FEAT_DIM + f], contrib * feat[f]);
                    }
                }
            }
        }
    }
}

// Kernel: Normalize features by density
template <int FEAT_DIM>
__global__ void normalize_features_kernel(
    const float* __restrict__ grid_density,  // [X, Y, Z]
    float* __restrict__ grid_features,       // [X, Y, Z, FEAT_DIM]
    int n_voxels,
    float eps
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_voxels) return;

    float density = grid_density[idx];
    if (density > eps) {
        float inv_density = 1.0f / density;
        #pragma unroll
        for (int f = 0; f < FEAT_DIM; f++) {
            grid_features[idx * FEAT_DIM + f] *= inv_density;
        }
    }
}

// Generic kernel for arbitrary feature dimensions
__global__ void voxelize_gaussians_generic_kernel(
    const float* __restrict__ means3d,
    const float* __restrict__ opacities,
    const float* __restrict__ covariances,
    const float* __restrict__ features,
    float* __restrict__ grid_density,
    float* __restrict__ grid_features,
    int n_gaussians,
    int feat_dim,
    int grid_x, int grid_y, int grid_z,
    float vol_min_x, float vol_min_y, float vol_min_z,
    float voxel_size,
    float sigma_factor
) {
    int g = blockIdx.x * blockDim.x + threadIdx.x;
    if (g >= n_gaussians) return;

    float mean_x = means3d[g * 3 + 0];
    float mean_y = means3d[g * 3 + 1];
    float mean_z = means3d[g * 3 + 2];
    float opacity = opacities[g];

    float cov[9], cov_inv[9];
    for (int i = 0; i < 9; i++) {
        cov[i] = covariances[g * 9 + i];
    }
    mat3_inverse(cov, cov_inv);

    float sigma_x = sqrtf(cov[0]);
    float sigma_y = sqrtf(cov[4]);
    float sigma_z = sqrtf(cov[8]);

    int i_min = max(0, (int)floorf((mean_x - sigma_factor * sigma_x - vol_min_x) / voxel_size));
    int i_max = min(grid_x - 1, (int)floorf((mean_x + sigma_factor * sigma_x - vol_min_x) / voxel_size));
    int j_min = max(0, (int)floorf((mean_y - sigma_factor * sigma_y - vol_min_y) / voxel_size));
    int j_max = min(grid_y - 1, (int)floorf((mean_y + sigma_factor * sigma_y - vol_min_y) / voxel_size));
    int k_min = max(0, (int)floorf((mean_z - sigma_factor * sigma_z - vol_min_z) / voxel_size));
    int k_max = min(grid_z - 1, (int)floorf((mean_z + sigma_factor * sigma_z - vol_min_z) / voxel_size));

    for (int i = i_min; i <= i_max; i++) {
        for (int j = j_min; j <= j_max; j++) {
            for (int k = k_min; k <= k_max; k++) {
                float vx = vol_min_x + (i + 0.5f) * voxel_size;
                float vy = vol_min_y + (j + 0.5f) * voxel_size;
                float vz = vol_min_z + (k + 0.5f) * voxel_size;

                float dx = vx - mean_x;
                float dy = vy - mean_y;
                float dz = vz - mean_z;

                float maha = dx * (cov_inv[0] * dx + cov_inv[1] * dy + cov_inv[2] * dz) +
                             dy * (cov_inv[3] * dx + cov_inv[4] * dy + cov_inv[5] * dz) +
                             dz * (cov_inv[6] * dx + cov_inv[7] * dy + cov_inv[8] * dz);

                float contrib = opacity * expf(-0.5f * maha);

                int voxel_idx = i * grid_y * grid_z + j * grid_z + k;
                atomicAdd(&grid_density[voxel_idx], contrib);

                if (grid_features != nullptr && feat_dim > 0) {
                    for (int f = 0; f < feat_dim; f++) {
                        atomicAdd(&grid_features[voxel_idx * feat_dim + f],
                                  contrib * features[g * feat_dim + f]);
                    }
                }
            }
        }
    }
}

__global__ void normalize_features_generic_kernel(
    const float* __restrict__ grid_density,
    float* __restrict__ grid_features,
    int n_voxels,
    int feat_dim,
    float eps
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_voxels) return;

    float density = grid_density[idx];
    if (density > eps) {
        float inv_density = 1.0f / density;
        for (int f = 0; f < feat_dim; f++) {
            grid_features[idx * feat_dim + f] *= inv_density;
        }
    }
}

// C++ interface
std::vector<torch::Tensor> voxelize_gaussians_cuda(
    torch::Tensor means3d,      // [N, 3]
    torch::Tensor opacities,    // [N] or [N, 1]
    torch::Tensor covariances,  // [N, 3, 3]
    torch::Tensor features,     // [N, C] or empty
    std::vector<int64_t> grid_shape,  // [X, Y, Z]
    std::vector<float> vol_range,     // [xmin, ymin, zmin, xmax, ymax, zmax]
    float voxel_size,
    float sigma_factor,
    float eps
) {
    // Ensure inputs are contiguous and on CUDA
    means3d = means3d.contiguous().to(torch::kFloat32);
    opacities = opacities.contiguous().to(torch::kFloat32).view({-1});
    covariances = covariances.contiguous().to(torch::kFloat32).view({-1, 9});

    const int n_gaussians = means3d.size(0);
    const int grid_x = grid_shape[0];
    const int grid_y = grid_shape[1];
    const int grid_z = grid_shape[2];
    const int n_voxels = grid_x * grid_y * grid_z;

    // Check if features are provided
    bool has_features = features.numel() > 0;
    int feat_dim = has_features ? features.size(1) : 0;
    if (has_features) {
        features = features.contiguous().to(torch::kFloat32);
    }

    // Allocate output tensors
    auto options = torch::TensorOptions()
        .dtype(torch::kFloat32)
        .device(means3d.device());

    torch::Tensor grid_density = torch::zeros({grid_x, grid_y, grid_z}, options);
    torch::Tensor grid_features;
    if (has_features) {
        grid_features = torch::zeros({grid_x, grid_y, grid_z, feat_dim}, options);
    } else {
        grid_features = torch::empty({0}, options);
    }

    // Launch voxelization kernel
    const int threads = THREADS_PER_BLOCK;
    const int blocks = (n_gaussians + threads - 1) / threads;

    float vol_min_x = vol_range[0];
    float vol_min_y = vol_range[1];
    float vol_min_z = vol_range[2];

    // Use specialized kernels for common feature dimensions
    if (feat_dim == 128) {
        voxelize_gaussians_kernel<128><<<blocks, threads>>>(
            means3d.data_ptr<float>(),
            opacities.data_ptr<float>(),
            covariances.data_ptr<float>(),
            has_features ? features.data_ptr<float>() : nullptr,
            grid_density.data_ptr<float>(),
            has_features ? grid_features.data_ptr<float>() : nullptr,
            n_gaussians, grid_x, grid_y, grid_z,
            vol_min_x, vol_min_y, vol_min_z,
            voxel_size, sigma_factor
        );
    } else if (feat_dim == 256) {
        voxelize_gaussians_kernel<256><<<blocks, threads>>>(
            means3d.data_ptr<float>(),
            opacities.data_ptr<float>(),
            covariances.data_ptr<float>(),
            has_features ? features.data_ptr<float>() : nullptr,
            grid_density.data_ptr<float>(),
            has_features ? grid_features.data_ptr<float>() : nullptr,
            n_gaussians, grid_x, grid_y, grid_z,
            vol_min_x, vol_min_y, vol_min_z,
            voxel_size, sigma_factor
        );
    } else if (feat_dim == 512) {
        voxelize_gaussians_kernel<512><<<blocks, threads>>>(
            means3d.data_ptr<float>(),
            opacities.data_ptr<float>(),
            covariances.data_ptr<float>(),
            has_features ? features.data_ptr<float>() : nullptr,
            grid_density.data_ptr<float>(),
            has_features ? grid_features.data_ptr<float>() : nullptr,
            n_gaussians, grid_x, grid_y, grid_z,
            vol_min_x, vol_min_y, vol_min_z,
            voxel_size, sigma_factor
        );
    } else {
        // Generic kernel for other dimensions
        voxelize_gaussians_generic_kernel<<<blocks, threads>>>(
            means3d.data_ptr<float>(),
            opacities.data_ptr<float>(),
            covariances.data_ptr<float>(),
            has_features ? features.data_ptr<float>() : nullptr,
            grid_density.data_ptr<float>(),
            has_features ? grid_features.data_ptr<float>() : nullptr,
            n_gaussians, feat_dim, grid_x, grid_y, grid_z,
            vol_min_x, vol_min_y, vol_min_z,
            voxel_size, sigma_factor
        );
    }

    // Normalize features by density
    if (has_features) {
        const int norm_blocks = (n_voxels + threads - 1) / threads;

        if (feat_dim == 128) {
            normalize_features_kernel<128><<<norm_blocks, threads>>>(
                grid_density.data_ptr<float>(),
                grid_features.data_ptr<float>(),
                n_voxels, eps
            );
        } else if (feat_dim == 256) {
            normalize_features_kernel<256><<<norm_blocks, threads>>>(
                grid_density.data_ptr<float>(),
                grid_features.data_ptr<float>(),
                n_voxels, eps
            );
        } else if (feat_dim == 512) {
            normalize_features_kernel<512><<<norm_blocks, threads>>>(
                grid_density.data_ptr<float>(),
                grid_features.data_ptr<float>(),
                n_voxels, eps
            );
        } else {
            normalize_features_generic_kernel<<<norm_blocks, threads>>>(
                grid_density.data_ptr<float>(),
                grid_features.data_ptr<float>(),
                n_voxels, feat_dim, eps
            );
        }
    }

    // Add channel dimension to density
    grid_density = grid_density.unsqueeze(-1);

    return {grid_density, grid_features};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("voxelize_gaussians", &voxelize_gaussians_cuda, "Voxelize Gaussians (CUDA)");
}
