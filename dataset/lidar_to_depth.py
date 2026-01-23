"""LiDAR-to-depth projection utilities for T4."""

from typing import Tuple

import numpy as np


def load_lidar_pcd_bin(path: str) -> np.ndarray:
    """Load T4 LiDAR .pcd.bin file as float32 array.

    Returns:
        Array of shape [N, 5] with (x, y, z, intensity, ring).
    """
    data = np.fromfile(path, dtype=np.float32)
    if data.size % 5 != 0:
        raise ValueError(f"Unexpected LiDAR data size in {path}")
    return data.reshape(-1, 5)


def project_lidar_to_camera(
    points: np.ndarray,
    cam2img: np.ndarray,
    cam2ego: np.ndarray,
    image_size: Tuple[int, int],
    min_depth: float = 0.1,
    max_depth: float = 80.0,
) -> np.ndarray:
    """Project ego-frame LiDAR points to a depth map for one camera.

    Args:
        points: [N, 5] LiDAR points in ego frame.
        cam2img: [3, 3] or [4, 4] camera intrinsic matrix.
        cam2ego: [4, 4] camera-to-ego extrinsic.
        image_size: (H, W) of the target depth map.
        min_depth: Minimum valid depth.
        max_depth: Maximum valid depth.

    Returns:
        Depth map [H, W] with 0 indicating missing depth.
    """
    H, W = image_size

    cam2img = np.asarray(cam2img, dtype=np.float32)
    if cam2img.shape == (4, 4):
        cam2img = cam2img[:3, :3]

    cam2ego = np.asarray(cam2ego, dtype=np.float32)
    if cam2ego.shape != (4, 4):
        raise ValueError(f"cam2ego must be 4x4, got {cam2ego.shape}")

    ego2cam = np.linalg.inv(cam2ego)

    xyz = points[:, :3].astype(np.float32)
    xyz_h = np.concatenate([xyz, np.ones((xyz.shape[0], 1), dtype=np.float32)], axis=1)
    xyz_cam = (ego2cam @ xyz_h.T).T[:, :3]

    z = xyz_cam[:, 2]
    valid = z > min_depth
    xyz_cam = xyz_cam[valid]
    z = z[valid]

    if xyz_cam.shape[0] == 0:
        return np.zeros((H, W), dtype=np.float32)

    uv = (cam2img @ xyz_cam.T).T
    depth = uv[:, 2]
    u = (uv[:, 0] / depth).astype(np.int32)
    v = (uv[:, 1] / depth).astype(np.int32)

    valid = (u >= 0) & (u < W) & (v >= 0) & (v < H) & (depth < max_depth) & (depth > min_depth)
    u = u[valid]
    v = v[valid]
    depth = depth[valid]

    depth_img = np.full((H, W), np.inf, dtype=np.float32)
    np.minimum.at(depth_img, (v, u), depth)
    depth_img[~np.isfinite(depth_img)] = 0.0

    return depth_img


def interpolate_depth(depth: np.ndarray, method: str = "none") -> np.ndarray:
    """Interpolate missing depth values.

    Args:
        depth: [H, W] depth map with 0 indicating missing depth.
        method: "none" or "nearest".
    """
    if method == "none":
        return depth
    if method != "nearest":
        raise ValueError(f"Unsupported interpolation method: {method}")

    try:
        from scipy.ndimage import distance_transform_edt
    except ImportError as exc:
        raise ImportError("scipy is required for nearest interpolation") from exc

    valid = depth > 0
    if not np.any(valid):
        return depth

    inv_valid = ~valid
    _, indices = distance_transform_edt(inv_valid, return_indices=True)
    filled = depth[tuple(indices)]
    return filled.astype(np.float32)
