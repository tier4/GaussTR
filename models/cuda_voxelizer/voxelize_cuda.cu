/*
 * CUDA kernels for Gaussian voxelization - Voxel-centric version.
 *
 * This version uses a voxel-centric approach where each thread processes
 * one voxel and iterates over all Gaussians. This eliminates atomic
 * operations and improves memory coalescing.
 */

#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <cmath>

#define THREADS_PER_BLOCK 256
#define GAUSSIANS_PER_BATCH 64  // Load Gaussians in batches to shared memory

// Helper: 3x3 matrix inverse (for covariance)
__device__ void mat3_inverse_v2(const float* __restrict__ mat, float* __restrict__ inv) {
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

// Struct to hold precomputed Gaussian data
struct GaussianData {
    float mean_x, mean_y, mean_z;
    float opacity;
    float cov_inv[9];
    float bound_min_x, bound_max_x;
    float bound_min_y, bound_max_y;
    float bound_min_z, bound_max_z;
};

// Kernel: Precompute Gaussian bounds and inverse covariances
__global__ void precompute_gaussians_kernel(
    const float* __restrict__ means3d,      // [N, 3]
    const float* __restrict__ opacities,    // [N]
    const float* __restrict__ covariances,  // [N, 9]
    float* __restrict__ gaussian_data,      // [N, 19] - precomputed data
    int n_gaussians,
    float sigma_factor
) {
    int g = blockIdx.x * blockDim.x + threadIdx.x;
    if (g >= n_gaussians) return;

    float mean_x = means3d[g * 3 + 0];
    float mean_y = means3d[g * 3 + 1];
    float mean_z = means3d[g * 3 + 2];
    float opacity = opacities[g];

    float cov[9], cov_inv[9];
    #pragma unroll
    for (int i = 0; i < 9; i++) {
        cov[i] = covariances[g * 9 + i];
    }
    mat3_inverse_v2(cov, cov_inv);

    float sigma_x = sqrtf(cov[0]);
    float sigma_y = sqrtf(cov[4]);
    float sigma_z = sqrtf(cov[8]);

    // Store precomputed data
    float* out = gaussian_data + g * 19;
    out[0] = mean_x;
    out[1] = mean_y;
    out[2] = mean_z;
    out[3] = opacity;
    #pragma unroll
    for (int i = 0; i < 9; i++) {
        out[4 + i] = cov_inv[i];
    }
    out[13] = mean_x - sigma_factor * sigma_x;  // bound_min_x
    out[14] = mean_x + sigma_factor * sigma_x;  // bound_max_x
    out[15] = mean_y - sigma_factor * sigma_y;  // bound_min_y
    out[16] = mean_y + sigma_factor * sigma_y;  // bound_max_y
    out[17] = mean_z - sigma_factor * sigma_z;  // bound_min_z
    out[18] = mean_z + sigma_factor * sigma_z;  // bound_max_z
}

// Voxel-centric kernel: each thread handles one voxel
__global__ void voxelize_voxel_centric_kernel(
    const float* __restrict__ gaussian_data,  // [N, 19] precomputed
    const float* __restrict__ features,       // [N, C]
    float* __restrict__ grid_density,         // [X, Y, Z]
    float* __restrict__ grid_features,        // [X, Y, Z, C]
    int n_gaussians,
    int feat_dim,
    int grid_x, int grid_y, int grid_z,
    float vol_min_x, float vol_min_y, float vol_min_z,
    float voxel_size,
    float eps
) {
    int voxel_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int n_voxels = grid_x * grid_y * grid_z;
    if (voxel_idx >= n_voxels) return;

    // Convert linear index to 3D
    int i = voxel_idx / (grid_y * grid_z);
    int j = (voxel_idx / grid_z) % grid_y;
    int k = voxel_idx % grid_z;

    // Compute voxel center
    float vx = vol_min_x + (i + 0.5f) * voxel_size;
    float vy = vol_min_y + (j + 0.5f) * voxel_size;
    float vz = vol_min_z + (k + 0.5f) * voxel_size;

    // Accumulate density and features
    float density = 0.0f;

    // Dynamic feature accumulation (up to 32 features in registers)
    float feat_accum[32];
    int n_feat_regs = min(feat_dim, 32);
    #pragma unroll
    for (int f = 0; f < 32; f++) {
        feat_accum[f] = 0.0f;
    }

    // Iterate over all Gaussians
    for (int g = 0; g < n_gaussians; g++) {
        const float* gdata = gaussian_data + g * 19;

        // Load bounds for early rejection
        float bound_min_x = gdata[13];
        float bound_max_x = gdata[14];
        float bound_min_y = gdata[15];
        float bound_max_y = gdata[16];
        float bound_min_z = gdata[17];
        float bound_max_z = gdata[18];

        // AABB check - skip if voxel outside Gaussian bounds
        if (vx < bound_min_x || vx > bound_max_x ||
            vy < bound_min_y || vy > bound_max_y ||
            vz < bound_min_z || vz > bound_max_z) {
            continue;
        }

        // Load Gaussian parameters
        float mean_x = gdata[0];
        float mean_y = gdata[1];
        float mean_z = gdata[2];
        float opacity = gdata[3];

        // Compute diff
        float dx = vx - mean_x;
        float dy = vy - mean_y;
        float dz = vz - mean_z;

        // Load inverse covariance and compute Mahalanobis distance
        float maha = dx * (gdata[4] * dx + gdata[5] * dy + gdata[6] * dz) +
                     dy * (gdata[7] * dx + gdata[8] * dy + gdata[9] * dz) +
                     dz * (gdata[10] * dx + gdata[11] * dy + gdata[12] * dz);

        // Compute contribution
        float contrib = opacity * expf(-0.5f * maha);
        density += contrib;

        // Accumulate features
        if (features != nullptr && feat_dim > 0) {
            const float* gfeat = features + g * feat_dim;
            for (int f = 0; f < n_feat_regs; f++) {
                feat_accum[f] += contrib * gfeat[f];
            }
            // Handle remaining features (if feat_dim > 32)
            for (int f = 32; f < feat_dim; f++) {
                atomicAdd(&grid_features[voxel_idx * feat_dim + f], contrib * gfeat[f]);
            }
        }
    }

    // Write density (no atomic needed!)
    grid_density[voxel_idx] = density;

    // Write and normalize features
    if (features != nullptr && feat_dim > 0 && density > eps) {
        float inv_density = 1.0f / density;
        for (int f = 0; f < n_feat_regs; f++) {
            grid_features[voxel_idx * feat_dim + f] = feat_accum[f] * inv_density;
        }
        // Normalize remaining features (if feat_dim > 32)
        for (int f = 32; f < feat_dim; f++) {
            grid_features[voxel_idx * feat_dim + f] *= inv_density;
        }
    }
}

// Optimized voxel-centric kernel with shared memory for Gaussian data
__global__ void voxelize_voxel_centric_smem_kernel(
    const float* __restrict__ gaussian_data,  // [N, 19] precomputed
    const float* __restrict__ features,       // [N, C]
    float* __restrict__ grid_density,         // [X, Y, Z]
    float* __restrict__ grid_features,        // [X, Y, Z, C]
    int n_gaussians,
    int feat_dim,
    int grid_x, int grid_y, int grid_z,
    float vol_min_x, float vol_min_y, float vol_min_z,
    float voxel_size,
    float eps
) {
    // Shared memory for Gaussian bounds (for fast AABB rejection)
    __shared__ float smem_bounds[GAUSSIANS_PER_BATCH * 6];  // min/max xyz
    __shared__ float smem_means[GAUSSIANS_PER_BATCH * 4];   // mean xyz + opacity

    int voxel_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int n_voxels = grid_x * grid_y * grid_z;

    // Convert linear index to 3D
    int i = voxel_idx / (grid_y * grid_z);
    int j = (voxel_idx / grid_z) % grid_y;
    int k = voxel_idx % grid_z;

    // Compute voxel center (even if out of bounds, for shared mem loading)
    float vx = vol_min_x + (i + 0.5f) * voxel_size;
    float vy = vol_min_y + (j + 0.5f) * voxel_size;
    float vz = vol_min_z + (k + 0.5f) * voxel_size;

    // Accumulate density and features
    float density = 0.0f;
    float feat_accum[17];  // Optimized for feat_dim=17
    #pragma unroll
    for (int f = 0; f < 17; f++) {
        feat_accum[f] = 0.0f;
    }

    // Process Gaussians in batches
    for (int batch_start = 0; batch_start < n_gaussians; batch_start += GAUSSIANS_PER_BATCH) {
        int batch_size = min(GAUSSIANS_PER_BATCH, n_gaussians - batch_start);

        // Cooperatively load bounds to shared memory
        int tid = threadIdx.x;
        for (int offset = tid; offset < batch_size * 6; offset += blockDim.x) {
            int g_local = offset / 6;
            int g_global = batch_start + g_local;
            int field = offset % 6;
            smem_bounds[offset] = gaussian_data[g_global * 19 + 13 + field];
        }
        for (int offset = tid; offset < batch_size * 4; offset += blockDim.x) {
            int g_local = offset / 4;
            int g_global = batch_start + g_local;
            int field = offset % 4;
            smem_means[offset] = gaussian_data[g_global * 19 + field];
        }
        __syncthreads();

        // Skip if this thread's voxel is out of bounds
        if (voxel_idx < n_voxels) {
            // Process this batch of Gaussians
            for (int g_local = 0; g_local < batch_size; g_local++) {
                // Fast AABB check using shared memory
                float* bounds = smem_bounds + g_local * 6;
                if (vx < bounds[0] || vx > bounds[1] ||
                    vy < bounds[2] || vy > bounds[3] ||
                    vz < bounds[4] || vz > bounds[5]) {
                    continue;
                }

                int g_global = batch_start + g_local;
                const float* gdata = gaussian_data + g_global * 19;

                // Load from shared memory
                float* means = smem_means + g_local * 4;
                float mean_x = means[0];
                float mean_y = means[1];
                float mean_z = means[2];
                float opacity = means[3];

                // Compute diff
                float dx = vx - mean_x;
                float dy = vy - mean_y;
                float dz = vz - mean_z;

                // Load inverse covariance from global memory (less frequent access)
                float maha = dx * (gdata[4] * dx + gdata[5] * dy + gdata[6] * dz) +
                             dy * (gdata[7] * dx + gdata[8] * dy + gdata[9] * dz) +
                             dz * (gdata[10] * dx + gdata[11] * dy + gdata[12] * dz);

                float contrib = opacity * expf(-0.5f * maha);
                density += contrib;

                // Accumulate features
                if (features != nullptr && feat_dim > 0) {
                    const float* gfeat = features + g_global * feat_dim;
                    #pragma unroll
                    for (int f = 0; f < 17; f++) {
                        if (f < feat_dim) {
                            feat_accum[f] += contrib * gfeat[f];
                        }
                    }
                }
            }
        }
        __syncthreads();
    }

    // Write results
    if (voxel_idx < n_voxels) {
        grid_density[voxel_idx] = density;

        if (features != nullptr && feat_dim > 0 && density > eps) {
            float inv_density = 1.0f / density;
            for (int f = 0; f < feat_dim; f++) {
                grid_features[voxel_idx * feat_dim + f] = feat_accum[f] * inv_density;
            }
        }
    }
}

// C++ interface
std::vector<torch::Tensor> voxelize_gaussians_v2_cuda(
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

    bool has_features = features.numel() > 0;
    int feat_dim = has_features ? features.size(1) : 0;
    if (has_features) {
        features = features.contiguous().to(torch::kFloat32);
    }

    auto options = torch::TensorOptions()
        .dtype(torch::kFloat32)
        .device(means3d.device());

    // Allocate precomputed Gaussian data
    torch::Tensor gaussian_data = torch::empty({n_gaussians, 19}, options);

    // Precompute Gaussian bounds and inverse covariances
    const int threads = THREADS_PER_BLOCK;
    int blocks = (n_gaussians + threads - 1) / threads;

    precompute_gaussians_kernel<<<blocks, threads>>>(
        means3d.data_ptr<float>(),
        opacities.data_ptr<float>(),
        covariances.data_ptr<float>(),
        gaussian_data.data_ptr<float>(),
        n_gaussians,
        sigma_factor
    );

    // Allocate output tensors
    torch::Tensor grid_density = torch::zeros({grid_x, grid_y, grid_z}, options);
    torch::Tensor grid_features;
    if (has_features) {
        grid_features = torch::zeros({grid_x, grid_y, grid_z, feat_dim}, options);
    } else {
        grid_features = torch::empty({0}, options);
    }

    // Launch voxel-centric kernel
    blocks = (n_voxels + threads - 1) / threads;

    float vol_min_x = vol_range[0];
    float vol_min_y = vol_range[1];
    float vol_min_z = vol_range[2];

    // Use shared memory version for better performance
    voxelize_voxel_centric_smem_kernel<<<blocks, threads>>>(
        gaussian_data.data_ptr<float>(),
        has_features ? features.data_ptr<float>() : nullptr,
        grid_density.data_ptr<float>(),
        has_features ? grid_features.data_ptr<float>() : nullptr,
        n_gaussians, feat_dim,
        grid_x, grid_y, grid_z,
        vol_min_x, vol_min_y, vol_min_z,
        voxel_size, eps
    );

    // Add channel dimension to density
    grid_density = grid_density.unsqueeze(-1);

    return {grid_density, grid_features};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("voxelize_gaussians", &voxelize_gaussians_v2_cuda, "Voxelize Gaussians - Voxel-centric (CUDA)");
}
