#!/usr/bin/env python
"""Generate per-camera depth maps from T4 LiDAR.

Output structure:
    {output_dir}/{chunk_name}/{camera_name}/{frame_id}.npz

Example:
    python -m scripts.t4.generate_depth \
        --ann-file /path/to/t4_infos_train.pkl \
        --output-dir /path/to/T4_lidar_depth \
        --data-root /path/to/T4_datasets \
        --num-workers 8
"""

import argparse
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, Any, List, Tuple

import cv2
import numpy as np
from tqdm import tqdm

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import directly from the module file to avoid __init__.py import chain
import importlib.util
spec = importlib.util.spec_from_file_location("lidar_to_depth", project_root / "dataset" / "lidar_to_depth.py")
lidar_to_depth = importlib.util.module_from_spec(spec)
spec.loader.exec_module(lidar_to_depth)

load_lidar_pcd_bin = lidar_to_depth.load_lidar_pcd_bin
project_lidar_to_camera = lidar_to_depth.project_lidar_to_camera
interpolate_depth = lidar_to_depth.interpolate_depth

# T4 camera names (avoid importing t4_dataset which has heavy dependencies)
T4_CAMERA_NAMES = [
    "CAM_FRONT",
    "CAM_FRONT_WIDE",
    "CAM_FRONT_LEFT_WIDE",
    "CAM_FRONT_RIGHT_WIDE",
    "CAM_BACK_LEFT_WIDE",
    "CAM_BACK_RIGHT_WIDE",
]


def _load_annotations(path: str) -> List[Dict[str, Any]]:
    import pickle

    with open(path, "rb") as f:
        data = pickle.load(f)

    if isinstance(data, dict):
        if "data_list" in data:
            return data["data_list"]
        if "infos" in data:
            return data["infos"]
        raise ValueError(f"Unknown annotation format in {path}")
    if isinstance(data, list):
        return data
    raise ValueError(f"Unknown annotation format in {path}")


def _resolve_path(data_root: str, path: str) -> str:
    if not path:
        return path
    if os.path.isabs(path) or path.startswith(data_root):
        return path
    return os.path.join(data_root, path)


def _extract_chunk_name(img_path: str) -> str:
    """Extract chunk name from T4 image path.

    Path format: .../t4_datasets/{chunk_name}/data/{camera}/{filename}.jpg
    """
    parts = img_path.replace("\\", "/").split("/")
    for i, part in enumerate(parts):
        if part == "t4_datasets" and i + 1 < len(parts):
            return parts[i + 1]
    # Fallback: return empty string (flat structure)
    return ""


def _process_sample(
    info: Dict[str, Any],
    data_root: str,
    output_dir: str,
    camera_names: List[str],
    min_depth: float,
    max_depth: float,
    interpolation: str,
    overwrite: bool,
) -> Tuple[int, int]:
    """Process a single sample. Returns (generated_count, skipped_count)."""
    lidar_path = _resolve_path(data_root, info.get("lidar_path", ""))
    if not lidar_path or not os.path.exists(lidar_path):
        return 0, 0

    points = load_lidar_pcd_bin(lidar_path)
    images = info.get("images", {})
    generated = 0
    skipped = 0

    # Get lidar calibration data (lidar2ego and ego2global at lidar timestamp)
    lidar_info = info.get("lidar_points", {})
    lidar2ego = lidar_info.get("lidar2ego") if lidar_info else None
    lidar_ego2global = lidar_info.get("ego2global") if lidar_info else None

    # Extract chunk name from first available image path
    chunk_name = ""
    for cam_name in camera_names:
        cam_info = images.get(cam_name)
        if cam_info and cam_info.get("img_path"):
            chunk_name = _extract_chunk_name(cam_info["img_path"])
            break

    for cam_name in camera_names:
        cam_info = images.get(cam_name)
        if not cam_info:
            continue

        img_path = _resolve_path(data_root, cam_info.get("img_path", ""))
        if not img_path or not os.path.exists(img_path):
            continue

        basename = os.path.basename(img_path).split(".")[0]
        # Use chunk_name in output path to match SAM3/dinov3clip structure
        if chunk_name:
            out_dir = os.path.join(output_dir, chunk_name, cam_name)
        else:
            out_dir = os.path.join(output_dir, cam_name)
        out_path = os.path.join(out_dir, f"{basename}.npy")
        out_path_npz = out_path.replace('.npy', '.npz')

        if not overwrite and os.path.exists(out_path_npz):
            skipped += 1
            continue

        os.makedirs(out_dir, exist_ok=True)

        img = cv2.imread(img_path)
        if img is None:
            continue

        H, W = img.shape[:2]
        cam2img = np.asarray(cam_info.get("cam2img"), dtype=np.float32)
        cam2ego = np.asarray(cam_info.get("cam2ego"), dtype=np.float32)
        cam_ego2global = cam_info.get("ego2global")  # ego2global at camera timestamp

        # Project with proper time compensation (4-step transformation)
        depth = project_lidar_to_camera(
            points,
            cam2img=cam2img,
            cam2ego=cam2ego,
            image_size=(H, W),
            min_depth=min_depth,
            max_depth=max_depth,
            lidar2ego=lidar2ego,
            lidar_ego2global=lidar_ego2global,
            cam_ego2global=cam_ego2global,
        )
        depth = interpolate_depth(depth, method=interpolation)

        # Save as sparse format (98%+ compression for lidar depth)
        # Store only non-zero values and their indices
        nonzero_mask = depth > 0
        indices = np.argwhere(nonzero_mask).astype(np.uint16)  # [N, 2] (row, col)
        values = depth[nonzero_mask].astype(np.float16)  # [N] depth values
        shape = np.array(depth.shape, dtype=np.uint16)  # [2] (H, W)

        out_path_npz = out_path.replace('.npy', '.npz')
        np.savez_compressed(out_path_npz, indices=indices, values=values, shape=shape)
        generated += 1

    return generated, skipped


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate T4 depth maps from LiDAR.")
    parser.add_argument("--ann-file", required=True, help="Path to T4 annotation pickle.")
    parser.add_argument("--output-dir", required=True, help="Output directory for depth maps.")
    parser.add_argument("--data-root", default="", help="Optional data root for relative paths.")
    parser.add_argument("--num-workers", type=int, default=4, help="Number of workers.")
    parser.add_argument("--min-depth", type=float, default=0.1, help="Minimum depth.")
    parser.add_argument("--max-depth", type=float, default=80.0, help="Maximum depth.")
    parser.add_argument(
        "--interpolation",
        choices=["none", "nearest"],
        default="none",
        help="Interpolation method for sparse depth.",
    )
    parser.add_argument(
        "--camera-names",
        default="",
        help="Comma-separated camera list (default: T4_CAMERA_NAMES).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing depth maps.",
    )
    args = parser.parse_args()

    ann_file = args.ann_file
    output_dir = args.output_dir
    data_root = args.data_root.strip() or ""

    camera_names = T4_CAMERA_NAMES
    if args.camera_names:
        camera_names = [name.strip() for name in args.camera_names.split(",") if name.strip()]

    os.makedirs(output_dir, exist_ok=True)
    infos = _load_annotations(ann_file)

    print(f"Loaded {len(infos)} samples from {ann_file}")
    print(f"Output directory: {output_dir}")
    print(f"Cameras: {camera_names}")
    print(f"Workers: {args.num_workers}")
    print(f"Overwrite: {args.overwrite}")
    print("-" * 60)

    total_generated = 0
    total_skipped = 0

    if args.num_workers <= 1:
        for info in tqdm(infos, desc="Processing samples"):
            generated, skipped = _process_sample(
                info,
                data_root=data_root,
                output_dir=output_dir,
                camera_names=camera_names,
                min_depth=args.min_depth,
                max_depth=args.max_depth,
                interpolation=args.interpolation,
                overwrite=args.overwrite,
            )
            total_generated += generated
            total_skipped += skipped
    else:
        with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
            futures = [
                executor.submit(
                    _process_sample,
                    info,
                    data_root,
                    output_dir,
                    camera_names,
                    args.min_depth,
                    args.max_depth,
                    args.interpolation,
                    args.overwrite,
                )
                for info in infos
            ]
            for future in tqdm(as_completed(futures), total=len(futures), desc="Processing samples"):
                generated, skipped = future.result()
                total_generated += generated
                total_skipped += skipped

    print("-" * 60)
    print(f"Done! Generated: {total_generated}, Skipped (existing): {total_skipped}")
    print(f"Total depth maps: {total_generated + total_skipped}")


if __name__ == "__main__":
    main()
