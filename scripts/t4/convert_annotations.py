#!/usr/bin/env python
"""Convert T4 JSON annotations into GaussTR-compatible pickle files.

Example:
    python -m scripts.t4.convert_annotations \
        --input-dir /path/to/T4_datasets \
        --output-dir /path/to/T4_processed \
        --val-ratio 0.1
"""

import argparse
import json
import os
import pickle
import random
from pathlib import Path
from typing import Dict, List, Any

import numpy as np


T4_CAMERA_NAMES = [
    "CAM_FRONT",
    "CAM_FRONT_LEFT_WIDE",
    "CAM_FRONT_RIGHT_WIDE",
    "CAM_BACK_LEFT_WIDE",
    "CAM_BACK_RIGHT_WIDE",
    "CAM_FRONT_WIDE",
]


def quat_to_matrix(quat: List[float]) -> np.ndarray:
    """Convert quaternion [qw, qx, qy, qz] to 3x3 rotation matrix."""
    qw, qx, qy, qz = quat
    xx, yy, zz = qx * qx, qy * qy, qz * qz
    xy, xz, yz = qx * qy, qx * qz, qy * qz
    wx, wy, wz = qw * qx, qw * qy, qw * qz

    return np.array([
        [1 - 2 * (yy + zz), 2 * (xy - wz), 2 * (xz + wy)],
        [2 * (xy + wz), 1 - 2 * (xx + zz), 2 * (yz - wx)],
        [2 * (xz - wy), 2 * (yz + wx), 1 - 2 * (xx + yy)],
    ], dtype=np.float32)


def find_annotation_dirs(input_dir: Path) -> List[Path]:
    ann_dirs = []
    for ann_dir in input_dir.rglob("annotation"):
        if (ann_dir / "sample.json").exists() and (ann_dir / "sample_data.json").exists():
            ann_dirs.append(ann_dir)
    return ann_dirs


def build_infos_for_chunk(
    ann_dir: Path,
    data_root: Path,
    camera_names: List[str],
    require_all_cams: bool,
    absolute_paths: bool,
) -> List[Dict[str, Any]]:
    chunk_root = ann_dir.parent

    with (ann_dir / "sensor.json").open() as f:
        sensors = {s["token"]: s for s in json.load(f)}
    with (ann_dir / "calibrated_sensor.json").open() as f:
        calibs = {c["token"]: c for c in json.load(f)}
    with (ann_dir / "sample.json").open() as f:
        samples = json.load(f)
    with (ann_dir / "sample_data.json").open() as f:
        sample_data = json.load(f)
    with (ann_dir / "scene.json").open() as f:
        scenes = json.load(f)
    with (ann_dir / "ego_pose.json").open() as f:
        ego_poses = {e["token"]: e for e in json.load(f)}

    scene_idx_map = {scene["token"]: idx for idx, scene in enumerate(scenes)}

    sample_data_by_sample: Dict[str, List[Dict[str, Any]]] = {}
    for sd in sample_data:
        sample_data_by_sample.setdefault(sd["sample_token"], []).append(sd)

    infos = []
    for sample in samples:
        info = {
            "token": sample["token"],
            "timestamp": sample["timestamp"],
            "scene_token": sample.get("scene_token", ""),
            "scene_idx": scene_idx_map.get(sample.get("scene_token", ""), 0),
            "images": {},
            "lidar_path": None,
            "lidar_points": None,  # Will store lidar2ego and ego2global
        }

        sample_datas = sample_data_by_sample.get(sample["token"], [])
        for sd in sample_datas:
            calib = calibs[sd["calibrated_sensor_token"]]
            sensor = sensors[calib["sensor_token"]]
            modality = sensor["modality"]
            channel = sensor["channel"]

            file_path = chunk_root / sd["filename"]
            if absolute_paths:
                rel_path = str(file_path)
            else:
                rel_path = os.path.relpath(file_path, data_root)

            # Get ego_pose for this sample_data
            ego_pose = ego_poses.get(sd.get("ego_pose_token"))

            if modality == "camera" and channel in camera_names:
                # Camera calibrated_sensor (cam2ego)
                cam2ego = np.eye(4, dtype=np.float32)
                cam2ego[:3, :3] = quat_to_matrix(calib["rotation"])
                cam2ego[:3, 3] = np.asarray(calib["translation"], dtype=np.float32)

                # Camera ego_pose at camera timestamp (ego2global)
                cam_ego2global = None
                if ego_pose is not None:
                    cam_ego2global = np.eye(4, dtype=np.float32)
                    cam_ego2global[:3, :3] = quat_to_matrix(ego_pose["rotation"])
                    cam_ego2global[:3, 3] = np.asarray(ego_pose["translation"], dtype=np.float32)

                info["images"][channel] = {
                    "img_path": rel_path,
                    "cam2img": calib["camera_intrinsic"],
                    "cam2ego": cam2ego.tolist(),
                    "ego2global": cam_ego2global.tolist() if cam_ego2global is not None else None,
                    "timestamp": sd.get("timestamp"),
                }
            elif modality == "lidar" and channel == "LIDAR_CONCAT":
                info["lidar_path"] = rel_path

                # Lidar calibrated_sensor (lidar2ego)
                lidar2ego = np.eye(4, dtype=np.float32)
                lidar2ego[:3, :3] = quat_to_matrix(calib["rotation"])
                lidar2ego[:3, 3] = np.asarray(calib["translation"], dtype=np.float32)

                # Lidar ego_pose at lidar timestamp (ego2global)
                lidar_ego2global = None
                if ego_pose is not None:
                    lidar_ego2global = np.eye(4, dtype=np.float32)
                    lidar_ego2global[:3, :3] = quat_to_matrix(ego_pose["rotation"])
                    lidar_ego2global[:3, 3] = np.asarray(ego_pose["translation"], dtype=np.float32)

                info["lidar_points"] = {
                    "lidar_path": rel_path,
                    "lidar2ego": lidar2ego.tolist(),
                    "ego2global": lidar_ego2global.tolist() if lidar_ego2global is not None else None,
                    "timestamp": sd.get("timestamp"),
                }

        if require_all_cams and len(info["images"]) != len(camera_names):
            continue
        if info["lidar_path"] is None:
            continue

        infos.append(info)

    return infos


def split_by_scene(
    infos: List[Dict[str, Any]],
    val_ratio: float,
    seed: int,
):
    scene_tokens = sorted({info.get("scene_token", "") for info in infos})
    scene_tokens = [s for s in scene_tokens if s]

    random.Random(seed).shuffle(scene_tokens)
    val_count = int(len(scene_tokens) * val_ratio)
    val_scenes = set(scene_tokens[:val_count])

    train_infos = [info for info in infos if info.get("scene_token") not in val_scenes]
    val_infos = [info for info in infos if info.get("scene_token") in val_scenes]
    return train_infos, val_infos


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert T4 annotations to pickle format.")
    parser.add_argument("--input-dir", required=True, help="Root directory containing T4 datasets.")
    parser.add_argument("--output-dir", required=True, help="Output directory for pickle files.")
    parser.add_argument("--val-ratio", type=float, default=0.1, help="Validation split ratio.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for split.")
    parser.add_argument(
        "--camera-names",
        default="",
        help="Comma-separated camera list (default: T4_CAMERA_NAMES).",
    )
    parser.add_argument(
        "--allow-missing-cams",
        action="store_true",
        help="Keep samples even if some cameras are missing.",
    )
    parser.add_argument(
        "--absolute-paths",
        action="store_true",
        help="Store absolute file paths instead of paths relative to input-dir.",
    )
    parser.add_argument(
        "--no-val",
        action="store_true",
        help="Write a single output file without train/val split.",
    )
    parser.add_argument(
        "--output-prefix",
        default="t4_infos",
        help="Output prefix for pickle files.",
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    camera_names = T4_CAMERA_NAMES
    if args.camera_names:
        camera_names = [name.strip() for name in args.camera_names.split(",") if name.strip()]

    ann_dirs = find_annotation_dirs(input_dir)
    if not ann_dirs:
        raise FileNotFoundError(f"No annotation folders found under {input_dir}")

    all_infos: List[Dict[str, Any]] = []
    for ann_dir in ann_dirs:
        infos = build_infos_for_chunk(
            ann_dir=ann_dir,
            data_root=input_dir,
            camera_names=camera_names,
            require_all_cams=not args.allow_missing_cams,
            absolute_paths=args.absolute_paths,
        )
        all_infos.extend(infos)

    # Sort by scene and timestamp for consistency
    all_infos.sort(key=lambda x: (x.get("scene_token", ""), x.get("timestamp", 0)))
    for idx, info in enumerate(all_infos):
        info["sample_idx"] = idx

    if args.no_val:
        out_path = output_dir / f"{args.output_prefix}_all.pkl"
        with out_path.open("wb") as f:
            pickle.dump({"data_list": all_infos}, f)
        print(f"Saved {len(all_infos)} samples to {out_path}")
        return

    train_infos, val_infos = split_by_scene(all_infos, args.val_ratio, args.seed)

    train_path = output_dir / f"{args.output_prefix}_train.pkl"
    val_path = output_dir / f"{args.output_prefix}_val.pkl"

    with train_path.open("wb") as f:
        pickle.dump({"data_list": train_infos}, f)
    with val_path.open("wb") as f:
        pickle.dump({"data_list": val_infos}, f)

    print(f"Saved {len(train_infos)} train samples to {train_path}")
    print(f"Saved {len(val_infos)} val samples to {val_path}")


if __name__ == "__main__":
    main()
