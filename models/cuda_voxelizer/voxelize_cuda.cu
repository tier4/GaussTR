/*
 * CUDA kernels for Gaussian voxelization.
 *
 * This implements GaussianFormer-style voxel-centric rendering:
 * 1. Preprocess: count voxels touched per Gaussian
 * 2. Prefix sum: compute offsets
 * 3. Duplicate: emit (voxel_key, gaussian_idx) pairs
 * 4. Radix sort: sort by voxel key
 * 5. Identify ranges: find start/end per voxel
 * 6. Render: each voxel iterates over its Gaussians (NO atomicAdd!)
 *
 * Reference: https://github.com/huang-yh/GaussianFormer
 */

#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cub/cub.cuh>
#include <cub/device/device_radix_sort.cuh>
#include <cub/device/device_scan.cuh>
#include <cooperative_groups.h>
#include <vector>
#include <cmath>

namespace cg = cooperative_groups;

#define THREADS_PER_BLOCK 256

// Helper: get bounding box of voxels touched by a Gaussian
__device__ void getRect(
    int mean_x, int mean_y, int mean_z,
    int radius,
    int grid_x, int grid_y, int grid_z,
    int3& rect_min, int3& rect_max
) {
    rect_min.x = max(0, mean_x - radius);
    rect_min.y = max(0, mean_y - radius);
    rect_min.z = max(0, mean_z - radius);
    rect_max.x = min(grid_x, mean_x + radius + 1);
    rect_max.y = min(grid_y, mean_y + radius + 1);
    rect_max.z = min(grid_z, mean_z + radius + 1);
}

// Kernel 1: Count voxels touched by each Gaussian
__global__ void preprocessKernel(
    const int P,
    const int* __restrict__ means3d_int,  // [P, 3] - integer voxel coords of Gaussian centers
    const int* __restrict__ radii,         // [P] - voxel radius for each Gaussian
    int grid_x, int grid_y, int grid_z,
    uint32_t* __restrict__ tiles_touched   // [P] - output: count of voxels touched
) {
    auto idx = cg::this_grid().thread_rank();
    if (idx >= P) return;

    tiles_touched[idx] = 0;

    int mean_x = means3d_int[idx * 3 + 0];
    int mean_y = means3d_int[idx * 3 + 1];
    int mean_z = means3d_int[idx * 3 + 2];
    int radius = radii[idx];

    int3 rect_min, rect_max;
    getRect(mean_x, mean_y, mean_z, radius, grid_x, grid_y, grid_z, rect_min, rect_max);

    int count = (rect_max.x - rect_min.x) * (rect_max.y - rect_min.y) * (rect_max.z - rect_min.z);
    if (count > 0) {
        tiles_touched[idx] = count;
    }
}

// Kernel 2: Duplicate Gaussians with voxel keys
__global__ void duplicateWithKeysKernel(
    const int P,
    const int* __restrict__ means3d_int,
    const int* __restrict__ radii,
    const uint32_t* __restrict__ offsets,  // [P] - prefix sum of tiles_touched
    int grid_x, int grid_y, int grid_z,
    uint32_t* __restrict__ keys_unsorted,   // Output: voxel keys
    uint32_t* __restrict__ values_unsorted  // Output: Gaussian indices
) {
    auto idx = cg::this_grid().thread_rank();
    if (idx >= P) return;

    uint32_t off = (idx == 0) ? 0 : offsets[idx - 1];

    int mean_x = means3d_int[idx * 3 + 0];
    int mean_y = means3d_int[idx * 3 + 1];
    int mean_z = means3d_int[idx * 3 + 2];
    int radius = radii[idx];

    int3 rect_min, rect_max;
    getRect(mean_x, mean_y, mean_z, radius, grid_x, grid_y, grid_z, rect_min, rect_max);

    for (int x = rect_min.x; x < rect_max.x; x++) {
        for (int y = rect_min.y; y < rect_max.y; y++) {
            for (int z = rect_min.z; z < rect_max.z; z++) {
                uint32_t key = x * grid_y * grid_z + y * grid_z + z;
                keys_unsorted[off] = key;
                values_unsorted[off] = idx;
                off++;
            }
        }
    }
}

// Kernel 3: Identify start/end ranges for each voxel in sorted list
__global__ void identifyRangesKernel(
    int L,  // Total number of (voxel, gaussian) pairs
    const uint32_t* __restrict__ sorted_keys,
    uint2* __restrict__ ranges  // [num_voxels] - output: (start, end) for each voxel
) {
    auto idx = cg::this_grid().thread_rank();
    if (idx >= L) return;

    uint32_t curr_voxel = sorted_keys[idx];

    if (idx == 0) {
        ranges[curr_voxel].x = 0;
    } else {
        uint32_t prev_voxel = sorted_keys[idx - 1];
        if (curr_voxel != prev_voxel) {
            ranges[prev_voxel].y = idx;
            ranges[curr_voxel].x = idx;
        }
    }
    if (idx == L - 1) {
        ranges[curr_voxel].y = L;
    }
}

// Kernel 4: Voxel-centric rendering (GaussianFormer style - NO atomicAdd!)
template <int FEAT_DIM>
__global__ void renderKernel(
    const int num_voxels,
    const float* __restrict__ voxel_centers,  // [num_voxels, 3] - voxel center coordinates
    const uint2* __restrict__ ranges,          // [num_voxels] - (start, end) into sorted_gaussians
    const uint32_t* __restrict__ sorted_gaussians,  // Sorted Gaussian indices
    const float* __restrict__ means3d,         // [P, 3] - Gaussian centers
    const float* __restrict__ cov3D,           // [P, 6] - precision matrix
    const float* __restrict__ opacities,       // [P]
    const float* __restrict__ features,        // [P, FEAT_DIM] or nullptr
    float* __restrict__ out_density,           // [num_voxels]
    float* __restrict__ out_features           // [num_voxels, FEAT_DIM] or nullptr
) {
    auto idx = cg::this_grid().thread_rank();
    if (idx >= num_voxels) return;

    // Load voxel center
    float vx = voxel_centers[idx * 3 + 0];
    float vy = voxel_centers[idx * 3 + 1];
    float vz = voxel_centers[idx * 3 + 2];

    // Load range of Gaussians for this voxel
    uint2 range = ranges[idx];

    // Accumulate contributions
    float density = 0.0f;
    float feat[FEAT_DIM > 0 ? FEAT_DIM : 1];  // Avoid zero-sized array
    #pragma unroll
    for (int f = 0; f < FEAT_DIM; f++) {
        feat[f] = 0.0f;
    }

    // Iterate over Gaussians affecting this voxel
    for (uint32_t i = range.x; i < range.y; i++) {
        uint32_t gs_idx = sorted_gaussians[i];

        // Load Gaussian parameters
        float mean_x = means3d[gs_idx * 3 + 0];
        float mean_y = means3d[gs_idx * 3 + 1];
        float mean_z = means3d[gs_idx * 3 + 2];

        // Compute diff: d = mean - voxel (GaussianFormer convention)
        float dx = mean_x - vx;
        float dy = mean_y - vy;
        float dz = mean_z - vz;

        // Load precision matrix elements
        float cov_xx = cov3D[gs_idx * 6 + 0];
        float cov_yy = cov3D[gs_idx * 6 + 1];
        float cov_zz = cov3D[gs_idx * 6 + 2];
        float cov_xy = cov3D[gs_idx * 6 + 3];
        float cov_yz = cov3D[gs_idx * 6 + 4];
        float cov_xz = cov3D[gs_idx * 6 + 5];

        // GaussianFormer-style Gaussian evaluation
        float power = cov_xx * dx * dx + cov_yy * dy * dy + cov_zz * dz * dz;
        power = -0.5f * power - (cov_xy * dx * dy + cov_yz * dy * dz + cov_xz * dx * dz);

        float contrib = opacities[gs_idx] * expf(power);
        density += contrib;

        // Accumulate weighted features
        if (features != nullptr && FEAT_DIM > 0) {
            #pragma unroll
            for (int f = 0; f < FEAT_DIM; f++) {
                feat[f] += features[gs_idx * FEAT_DIM + f] * contrib;
            }
        }
    }

    // Write output (direct write, no atomicAdd!)
    out_density[idx] = density;

    if (out_features != nullptr && FEAT_DIM > 0) {
        #pragma unroll
        for (int f = 0; f < FEAT_DIM; f++) {
            out_features[idx * FEAT_DIM + f] = feat[f];
        }
    }
}

// Generic render kernel for arbitrary feature dimensions
__global__ void renderKernelGeneric(
    const int num_voxels,
    const int feat_dim,
    const float* __restrict__ voxel_centers,
    const uint2* __restrict__ ranges,
    const uint32_t* __restrict__ sorted_gaussians,
    const float* __restrict__ means3d,
    const float* __restrict__ cov3D,
    const float* __restrict__ opacities,
    const float* __restrict__ features,
    float* __restrict__ out_density,
    float* __restrict__ out_features
) {
    auto idx = cg::this_grid().thread_rank();
    if (idx >= num_voxels) return;

    float vx = voxel_centers[idx * 3 + 0];
    float vy = voxel_centers[idx * 3 + 1];
    float vz = voxel_centers[idx * 3 + 2];

    uint2 range = ranges[idx];

    float density = 0.0f;

    // Use dynamic allocation for features
    extern __shared__ float shared_feat[];
    float* feat = &shared_feat[threadIdx.x * feat_dim];
    for (int f = 0; f < feat_dim; f++) {
        feat[f] = 0.0f;
    }

    for (uint32_t i = range.x; i < range.y; i++) {
        uint32_t gs_idx = sorted_gaussians[i];

        float mean_x = means3d[gs_idx * 3 + 0];
        float mean_y = means3d[gs_idx * 3 + 1];
        float mean_z = means3d[gs_idx * 3 + 2];

        float dx = mean_x - vx;
        float dy = mean_y - vy;
        float dz = mean_z - vz;

        float cov_xx = cov3D[gs_idx * 6 + 0];
        float cov_yy = cov3D[gs_idx * 6 + 1];
        float cov_zz = cov3D[gs_idx * 6 + 2];
        float cov_xy = cov3D[gs_idx * 6 + 3];
        float cov_yz = cov3D[gs_idx * 6 + 4];
        float cov_xz = cov3D[gs_idx * 6 + 5];

        float power = cov_xx * dx * dx + cov_yy * dy * dy + cov_zz * dz * dz;
        power = -0.5f * power - (cov_xy * dx * dy + cov_yz * dy * dz + cov_xz * dx * dz);

        float contrib = opacities[gs_idx] * expf(power);
        density += contrib;

        if (features != nullptr && feat_dim > 0) {
            for (int f = 0; f < feat_dim; f++) {
                feat[f] += features[gs_idx * feat_dim + f] * contrib;
            }
        }
    }

    out_density[idx] = density;

    if (out_features != nullptr && feat_dim > 0) {
        for (int f = 0; f < feat_dim; f++) {
            out_features[idx * feat_dim + f] = feat[f];
        }
    }
}

// Kernel: Normalize features by density
template <int FEAT_DIM>
__global__ void normalizeKernel(
    const int num_voxels,
    const float* __restrict__ density,
    float* __restrict__ features,
    float eps
) {
    auto idx = cg::this_grid().thread_rank();
    if (idx >= num_voxels) return;

    float d = density[idx];
    if (d > eps) {
        float inv_d = 1.0f / d;
        #pragma unroll
        for (int f = 0; f < FEAT_DIM; f++) {
            features[idx * FEAT_DIM + f] *= inv_d;
        }
    }
}

__global__ void normalizeKernelGeneric(
    const int num_voxels,
    const int feat_dim,
    const float* __restrict__ density,
    float* __restrict__ features,
    float eps
) {
    auto idx = cg::this_grid().thread_rank();
    if (idx >= num_voxels) return;

    float d = density[idx];
    if (d > eps) {
        float inv_d = 1.0f / d;
        for (int f = 0; f < feat_dim; f++) {
            features[idx * feat_dim + f] *= inv_d;
        }
    }
}

// C++ interface
std::vector<torch::Tensor> voxelize_gaussians_cuda(
    torch::Tensor means3d,      // [P, 3]
    torch::Tensor opacities,    // [P]
    torch::Tensor cov3D,        // [P, 6] - precision matrix
    torch::Tensor radii,        // [P] - voxel radii
    torch::Tensor features,     // [P, C] or empty
    std::vector<int64_t> grid_shape,  // [X, Y, Z]
    std::vector<float> vol_range,     // [xmin, ymin, zmin, xmax, ymax, zmax]
    float voxel_size,
    float eps
) {
    // Ensure inputs are contiguous
    means3d = means3d.contiguous().to(torch::kFloat32);
    opacities = opacities.contiguous().to(torch::kFloat32).view({-1});
    cov3D = cov3D.contiguous().to(torch::kFloat32).view({-1, 6});
    radii = radii.contiguous().to(torch::kInt32).view({-1});

    const int P = means3d.size(0);
    const int grid_x = grid_shape[0];
    const int grid_y = grid_shape[1];
    const int grid_z = grid_shape[2];
    const int num_voxels = grid_x * grid_y * grid_z;

    bool has_features = features.numel() > 0;
    int feat_dim = has_features ? features.size(1) : 0;
    if (has_features) {
        features = features.contiguous().to(torch::kFloat32);
    }

    auto device = means3d.device();
    auto options_float = torch::TensorOptions().dtype(torch::kFloat32).device(device);
    auto options_int = torch::TensorOptions().dtype(torch::kInt32).device(device);
    auto options_uint = torch::TensorOptions().dtype(torch::kInt32).device(device);  // uint32 as int32

    float vol_min_x = vol_range[0];
    float vol_min_y = vol_range[1];
    float vol_min_z = vol_range[2];

    // Step 0: Convert means3d to integer voxel coordinates
    torch::Tensor means3d_int = ((means3d - torch::tensor({vol_min_x, vol_min_y, vol_min_z}, options_float)) / voxel_size).to(torch::kInt32);

    // Step 1: Preprocess - count voxels touched per Gaussian
    torch::Tensor tiles_touched = torch::zeros({P}, options_uint);

    int blocks = (P + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    preprocessKernel<<<blocks, THREADS_PER_BLOCK>>>(
        P,
        means3d_int.data_ptr<int>(),
        radii.data_ptr<int>(),
        grid_x, grid_y, grid_z,
        reinterpret_cast<uint32_t*>(tiles_touched.data_ptr<int>())
    );

    // Step 2: Prefix sum to get offsets
    torch::Tensor offsets = torch::zeros({P}, options_uint);

    // Use CUB for prefix sum
    void* d_temp_storage = nullptr;
    size_t temp_storage_bytes = 0;
    cub::DeviceScan::InclusiveSum(
        d_temp_storage, temp_storage_bytes,
        reinterpret_cast<uint32_t*>(tiles_touched.data_ptr<int>()),
        reinterpret_cast<uint32_t*>(offsets.data_ptr<int>()),
        P
    );
    torch::Tensor temp_storage = torch::empty({(int64_t)temp_storage_bytes}, torch::TensorOptions().dtype(torch::kUInt8).device(device));
    cub::DeviceScan::InclusiveSum(
        temp_storage.data_ptr(),
        temp_storage_bytes,
        reinterpret_cast<uint32_t*>(tiles_touched.data_ptr<int>()),
        reinterpret_cast<uint32_t*>(offsets.data_ptr<int>()),
        P
    );

    // Get total number of pairs
    int num_pairs = 0;
    if (P > 0) {
        cudaMemcpy(&num_pairs, offsets.data_ptr<int>() + P - 1, sizeof(int), cudaMemcpyDeviceToHost);
    }

    if (num_pairs == 0) {
        // No Gaussians touch any voxels
        torch::Tensor grid_density = torch::zeros({grid_x, grid_y, grid_z, 1}, options_float);
        torch::Tensor grid_features = has_features ?
            torch::zeros({grid_x, grid_y, grid_z, feat_dim}, options_float) :
            torch::empty({0}, options_float);
        return {grid_density, grid_features};
    }

    // Step 3: Duplicate with keys
    torch::Tensor keys_unsorted = torch::empty({num_pairs}, options_uint);
    torch::Tensor values_unsorted = torch::empty({num_pairs}, options_uint);

    duplicateWithKeysKernel<<<blocks, THREADS_PER_BLOCK>>>(
        P,
        means3d_int.data_ptr<int>(),
        radii.data_ptr<int>(),
        reinterpret_cast<uint32_t*>(offsets.data_ptr<int>()),
        grid_x, grid_y, grid_z,
        reinterpret_cast<uint32_t*>(keys_unsorted.data_ptr<int>()),
        reinterpret_cast<uint32_t*>(values_unsorted.data_ptr<int>())
    );

    // Step 4: Radix sort by voxel key
    torch::Tensor keys_sorted = torch::empty({num_pairs}, options_uint);
    torch::Tensor values_sorted = torch::empty({num_pairs}, options_uint);

    temp_storage_bytes = 0;
    cub::DeviceRadixSort::SortPairs(
        nullptr, temp_storage_bytes,
        reinterpret_cast<uint32_t*>(keys_unsorted.data_ptr<int>()),
        reinterpret_cast<uint32_t*>(keys_sorted.data_ptr<int>()),
        reinterpret_cast<uint32_t*>(values_unsorted.data_ptr<int>()),
        reinterpret_cast<uint32_t*>(values_sorted.data_ptr<int>()),
        num_pairs
    );
    temp_storage = torch::empty({(int64_t)temp_storage_bytes}, torch::TensorOptions().dtype(torch::kUInt8).device(device));
    cub::DeviceRadixSort::SortPairs(
        temp_storage.data_ptr(),
        temp_storage_bytes,
        reinterpret_cast<uint32_t*>(keys_unsorted.data_ptr<int>()),
        reinterpret_cast<uint32_t*>(keys_sorted.data_ptr<int>()),
        reinterpret_cast<uint32_t*>(values_unsorted.data_ptr<int>()),
        reinterpret_cast<uint32_t*>(values_sorted.data_ptr<int>()),
        num_pairs
    );

    // Step 5: Identify ranges for each voxel
    // uint2 is 8 bytes = 2 int32
    torch::Tensor ranges = torch::zeros({num_voxels, 2}, options_uint);

    int range_blocks = (num_pairs + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    identifyRangesKernel<<<range_blocks, THREADS_PER_BLOCK>>>(
        num_pairs,
        reinterpret_cast<uint32_t*>(keys_sorted.data_ptr<int>()),
        reinterpret_cast<uint2*>(ranges.data_ptr<int>())
    );

    // Step 6: Generate voxel centers
    torch::Tensor voxel_centers = torch::empty({num_voxels, 3}, options_float);
    {
        // Generate grid of voxel centers
        auto x_coords = torch::arange(0, grid_x, options_float) * voxel_size + vol_min_x + 0.5f * voxel_size;
        auto y_coords = torch::arange(0, grid_y, options_float) * voxel_size + vol_min_y + 0.5f * voxel_size;
        auto z_coords = torch::arange(0, grid_z, options_float) * voxel_size + vol_min_z + 0.5f * voxel_size;

        auto grid = torch::meshgrid({x_coords, y_coords, z_coords}, "ij");
        voxel_centers = torch::stack({grid[0].flatten(), grid[1].flatten(), grid[2].flatten()}, 1).contiguous();
    }

    // Step 7: Render - voxel-centric, NO atomicAdd!
    torch::Tensor out_density = torch::zeros({num_voxels}, options_float);
    torch::Tensor out_features = has_features ?
        torch::zeros({num_voxels, feat_dim}, options_float) :
        torch::empty({0}, options_float);

    int render_blocks = (num_voxels + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

    if (feat_dim == 128) {
        renderKernel<128><<<render_blocks, THREADS_PER_BLOCK>>>(
            num_voxels,
            voxel_centers.data_ptr<float>(),
            reinterpret_cast<uint2*>(ranges.data_ptr<int>()),
            reinterpret_cast<uint32_t*>(values_sorted.data_ptr<int>()),
            means3d.data_ptr<float>(),
            cov3D.data_ptr<float>(),
            opacities.data_ptr<float>(),
            has_features ? features.data_ptr<float>() : nullptr,
            out_density.data_ptr<float>(),
            has_features ? out_features.data_ptr<float>() : nullptr
        );
    } else if (feat_dim == 256) {
        renderKernel<256><<<render_blocks, THREADS_PER_BLOCK>>>(
            num_voxels,
            voxel_centers.data_ptr<float>(),
            reinterpret_cast<uint2*>(ranges.data_ptr<int>()),
            reinterpret_cast<uint32_t*>(values_sorted.data_ptr<int>()),
            means3d.data_ptr<float>(),
            cov3D.data_ptr<float>(),
            opacities.data_ptr<float>(),
            has_features ? features.data_ptr<float>() : nullptr,
            out_density.data_ptr<float>(),
            has_features ? out_features.data_ptr<float>() : nullptr
        );
    } else if (feat_dim == 512) {
        renderKernel<512><<<render_blocks, THREADS_PER_BLOCK>>>(
            num_voxels,
            voxel_centers.data_ptr<float>(),
            reinterpret_cast<uint2*>(ranges.data_ptr<int>()),
            reinterpret_cast<uint32_t*>(values_sorted.data_ptr<int>()),
            means3d.data_ptr<float>(),
            cov3D.data_ptr<float>(),
            opacities.data_ptr<float>(),
            has_features ? features.data_ptr<float>() : nullptr,
            out_density.data_ptr<float>(),
            has_features ? out_features.data_ptr<float>() : nullptr
        );
    } else if (feat_dim > 0) {
        // Generic kernel with shared memory for features
        size_t shared_mem = THREADS_PER_BLOCK * feat_dim * sizeof(float);
        renderKernelGeneric<<<render_blocks, THREADS_PER_BLOCK, shared_mem>>>(
            num_voxels,
            feat_dim,
            voxel_centers.data_ptr<float>(),
            reinterpret_cast<uint2*>(ranges.data_ptr<int>()),
            reinterpret_cast<uint32_t*>(values_sorted.data_ptr<int>()),
            means3d.data_ptr<float>(),
            cov3D.data_ptr<float>(),
            opacities.data_ptr<float>(),
            features.data_ptr<float>(),
            out_density.data_ptr<float>(),
            out_features.data_ptr<float>()
        );
    } else {
        // No features, just density
        renderKernel<0><<<render_blocks, THREADS_PER_BLOCK>>>(
            num_voxels,
            voxel_centers.data_ptr<float>(),
            reinterpret_cast<uint2*>(ranges.data_ptr<int>()),
            reinterpret_cast<uint32_t*>(values_sorted.data_ptr<int>()),
            means3d.data_ptr<float>(),
            cov3D.data_ptr<float>(),
            opacities.data_ptr<float>(),
            nullptr,
            out_density.data_ptr<float>(),
            nullptr
        );
    }

    // Step 8: Normalize features by density
    if (has_features) {
        if (feat_dim == 128) {
            normalizeKernel<128><<<render_blocks, THREADS_PER_BLOCK>>>(
                num_voxels, out_density.data_ptr<float>(),
                out_features.data_ptr<float>(), eps
            );
        } else if (feat_dim == 256) {
            normalizeKernel<256><<<render_blocks, THREADS_PER_BLOCK>>>(
                num_voxels, out_density.data_ptr<float>(),
                out_features.data_ptr<float>(), eps
            );
        } else if (feat_dim == 512) {
            normalizeKernel<512><<<render_blocks, THREADS_PER_BLOCK>>>(
                num_voxels, out_density.data_ptr<float>(),
                out_features.data_ptr<float>(), eps
            );
        } else {
            normalizeKernelGeneric<<<render_blocks, THREADS_PER_BLOCK>>>(
                num_voxels, feat_dim, out_density.data_ptr<float>(),
                out_features.data_ptr<float>(), eps
            );
        }
    }

    // Reshape outputs to grid format
    torch::Tensor grid_density = out_density.view({grid_x, grid_y, grid_z, 1});
    torch::Tensor grid_features = has_features ?
        out_features.view({grid_x, grid_y, grid_z, feat_dim}) :
        torch::empty({0}, options_float);

    return {grid_density, grid_features};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("voxelize_gaussians", &voxelize_gaussians_cuda, "Voxelize Gaussians (CUDA)");
}
