#!/usr/bin/env python
"""Distributed PCA using multiple GPUs with cuML/RAPIDS.

Streams data to GPUs in parallel for fast PCA computation on large feature sets.

Requirements:
    RAPIDS cuML environment (cupy, cuml)

Usage:
    source /path/to/cuml_venv/bin/activate
    python -m scripts.t4.precompute_pca \
        --feats_root /path/to/T4_features \
        --reduce_dims 128 \
        --n_gpus 8
"""

import argparse
import os
import random
import time
from pathlib import Path
from typing import List

import cupy as cp
import numpy as np
import torch
from tqdm import tqdm


def load_feature(path: str, spatial_subsample: int = 4) -> np.ndarray:
    """Load a single feature file with spatial subsampling."""
    feat = np.load(path)
    if feat.dtype == np.int8:
        feat = feat.astype(np.float32) / 127.0
    else:
        feat = feat.astype(np.float32)
    feat = feat[:, ::spatial_subsample, ::spatial_subsample]
    feat = feat.transpose(1, 2, 0).reshape(-1, feat.shape[0])
    return feat


def load_batch_to_gpu(paths: List[str], spatial_subsample: int, gpu_id: int) -> cp.ndarray:
    """Load a batch of files directly to a specific GPU."""
    with cp.cuda.Device(gpu_id):
        feats = []
        for p in paths:
            feat = load_feature(p, spatial_subsample)
            feats.append(feat)
        data = np.concatenate(feats, axis=0)
        return cp.asarray(data, dtype=cp.float32)


def distributed_pca_8gpu(
    paths: List[str],
    n_components: int,
    spatial_subsample: int,
    n_gpus: int = 8,
) -> dict:
    """Distributed PCA across 8 GPUs.

    1. Split files across GPUs
    2. Each GPU loads its portion
    3. Compute distributed covariance
    4. Eigendecomposition on GPU 0
    """
    from concurrent.futures import ThreadPoolExecutor

    n_files = len(paths)
    files_per_gpu = (n_files + n_gpus - 1) // n_gpus

    print(f"\n[Loading] Distributing {n_files:,} files across {n_gpus} GPUs...")
    print(f"  ~{files_per_gpu:,} files per GPU")

    # Split paths for each GPU
    gpu_paths = []
    for i in range(n_gpus):
        start = i * files_per_gpu
        end = min((i + 1) * files_per_gpu, n_files)
        gpu_paths.append(paths[start:end])

    # Load data to each GPU in parallel
    gpu_data = [None] * n_gpus
    gpu_counts = [0] * n_gpus

    def load_gpu_data(gpu_id: int):
        """Load all files for one GPU with progress."""
        my_paths = gpu_paths[gpu_id]
        with cp.cuda.Device(gpu_id):
            feats = []
            for p in tqdm(my_paths, desc=f"GPU {gpu_id}", position=gpu_id, leave=False):
                feat = load_feature(p, spatial_subsample)
                feats.append(feat)
            if feats:
                data = np.concatenate(feats, axis=0)
                gpu_data[gpu_id] = cp.asarray(data, dtype=cp.float32)
                gpu_counts[gpu_id] = gpu_data[gpu_id].shape[0]
                del data, feats

    # Load all GPUs in parallel
    print()
    with ThreadPoolExecutor(max_workers=n_gpus) as executor:
        list(executor.map(load_gpu_data, range(n_gpus)))

    print(f"\n[Loaded] Data distributed across {n_gpus} GPUs")
    total_count = sum(gpu_counts)
    feat_dim = gpu_data[0].shape[1]
    print(f"  Total vectors: {total_count:,}")
    print(f"  Feature dim: {feat_dim}")

    # Compute global mean (reduce across GPUs)
    print("\n[Mean] Computing global mean...")
    with cp.cuda.Device(0):
        global_sum = cp.zeros(feat_dim, dtype=cp.float64)
        for i in range(n_gpus):
            with cp.cuda.Device(i):
                local_sum = gpu_data[i].sum(axis=0).astype(cp.float64)
            # Transfer to GPU 0
            global_sum += cp.asarray(local_sum.get())
        global_mean = (global_sum / total_count).astype(cp.float32)

    # Distribute mean to all GPUs and center data
    print("[Center] Centering data on each GPU...")
    for i in range(n_gpus):
        with cp.cuda.Device(i):
            local_mean = cp.asarray(global_mean.get())
            gpu_data[i] -= local_mean

    # Compute covariance (distributed matrix multiply)
    print("\n[Covariance] Computing distributed covariance matrix...")
    with cp.cuda.Device(0):
        cov_matrix = cp.zeros((feat_dim, feat_dim), dtype=cp.float32)

    for i in tqdm(range(n_gpus), desc="Accumulating"):
        with cp.cuda.Device(i):
            # X.T @ X on each GPU
            local_cov = gpu_data[i].T @ gpu_data[i]
        # Transfer to GPU 0 and accumulate
        with cp.cuda.Device(0):
            cov_matrix += cp.asarray(local_cov.get())

    with cp.cuda.Device(0):
        cov_matrix /= total_count

    # Eigendecomposition on GPU 0
    print("\n[Eigen] Computing eigenvectors on GPU 0...")
    start = time.time()
    with cp.cuda.Device(0):
        eigenvalues, eigenvectors = cp.linalg.eigh(cov_matrix)

        # Sort descending
        idx = cp.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]

        # Top k components
        V = eigenvectors[:, :n_components]
        S = cp.sqrt(cp.maximum(eigenvalues[:n_components], 0))

        total_var = float(eigenvalues.sum())
        explained_var = float(eigenvalues[:n_components].sum())
        var_explained = explained_var / total_var

        V_cpu = cp.asnumpy(V)
        S_cpu = cp.asnumpy(S)
        mean_cpu = cp.asnumpy(global_mean)

    print(f"  Completed in {time.time() - start:.2f}s")

    # Free GPU memory
    for i in range(n_gpus):
        gpu_data[i] = None
    cp.get_default_memory_pool().free_all_blocks()

    return {
        'v': V_cpu,
        's': S_cpu,
        'mean': mean_cpu,
        'var_explained': var_explained,
        'total_vectors': total_count,
    }


def main():
    parser = argparse.ArgumentParser(description="8-GPU distributed PCA")
    parser.add_argument("--feats_root", type=str,
                        default="/mnt/nvme3/T4_datasets_dinov3clip")
    parser.add_argument("--camera", type=str, default="CAM_FRONT_WIDE")
    parser.add_argument("--num_samples", type=int, default=-1)
    parser.add_argument("--reduce_dims", type=int, default=128)
    parser.add_argument("--output", type=str, default="ckpts/t4_pca_front_wide.pth")
    parser.add_argument("--spatial_subsample", type=int, default=4)
    parser.add_argument("--n_gpus", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    print("=" * 70)
    print(f"8-GPU Distributed PCA")
    print("=" * 70)
    print(f"Features: {args.feats_root}")
    print(f"Camera: {args.camera}")
    print(f"GPUs: {args.n_gpus}")
    print(f"Reduce dims: {args.reduce_dims}")
    print()

    # Find files
    feats_root = Path(args.feats_root)
    feature_files = []

    print("Scanning...")
    for chunk_dir in sorted(feats_root.iterdir()):
        if chunk_dir.is_dir():
            camera_dir = chunk_dir / args.camera
            if camera_dir.exists():
                feature_files.extend(str(f) for f in camera_dir.glob("*.npy"))

    print(f"Found {len(feature_files):,} files")

    if args.num_samples > 0 and len(feature_files) > args.num_samples:
        feature_files = random.sample(feature_files, args.num_samples)
        print(f"Sampled {len(feature_files):,} files")

    # Run distributed PCA
    start = time.time()
    result = distributed_pca_8gpu(
        feature_files,
        args.reduce_dims,
        args.spatial_subsample,
        args.n_gpus,
    )
    total_time = time.time() - start

    print(f"\nVariance explained: {result['var_explained'] * 100:.2f}%")

    # Save
    V = torch.from_numpy(result['v']).float()
    S = torch.from_numpy(result['s']).float()
    mean = torch.from_numpy(result['mean']).float()

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    torch.save({
        'v': V,
        'mean': mean,
        's': S,
        'variance_explained': result['var_explained'],
        'num_samples': len(feature_files),
        'total_vectors': result['total_vectors'],
        'reduce_dims': args.reduce_dims,
        'camera': args.camera,
    }, args.output)

    print(f"\nSaved to {args.output}")
    print(f"  V shape: {V.shape}")

    # Verify
    print("\nVerifying...")
    for path in random.sample(feature_files, min(3, len(feature_files))):
        feat = torch.from_numpy(load_feature(path, args.spatial_subsample)).float()
        recon = (feat @ V) @ V.T
        err = ((feat - recon) ** 2).mean().item()
        print(f"  {os.path.basename(path)}: err={err:.6f}")

    print(f"\n{'='*70}")
    print(f"Total: {total_time:.1f}s ({total_time/60:.1f} min)")
    print("Done!")


if __name__ == "__main__":
    main()
