#!/usr/bin/env python
"""Inference and visualization script for T4 dataset.

Usage:
    python -m scripts.t4.infer --checkpoint path/to/checkpoint.ckpt --num-samples 10
    python -m scripts.t4.infer --checkpoint path/to/checkpoint.ckpt --split train --num-samples 50
"""

import os
import sys
from pathlib import Path

import torch
import numpy as np
import cv2
from tqdm import tqdm
from omegaconf import OmegaConf

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

torch.set_float32_matmul_precision('high')

def load_t4_annotations(data_root: str):
    """Load T4 annotation data from sample's data_root."""
    import json

    # Find annotation path from data_root
    ann_path = os.path.join(data_root, 'annotation')
    if not os.path.exists(ann_path):
        return None

    try:
        with open(os.path.join(ann_path, 'sample_annotation.json')) as f:
            annotations = json.load(f)
        with open(os.path.join(ann_path, 'sample.json')) as f:
            samples = json.load(f)
        with open(os.path.join(ann_path, 'sample_data.json')) as f:
            sample_data = json.load(f)
        with open(os.path.join(ann_path, 'ego_pose.json')) as f:
            ego_poses = json.load(f)
        with open(os.path.join(ann_path, 'instance.json')) as f:
            instances = json.load(f)
        with open(os.path.join(ann_path, 'category.json')) as f:
            categories = json.load(f)
    except Exception as e:
        print(f"  Failed to load annotations from {ann_path}: {e}")
        return None

    # Build lookup dicts
    ego_pose_by_token = {e['token']: e for e in ego_poses}
    instance_by_token = {i['token']: i for i in instances}
    category_by_token = {c['token']: c for c in categories}

    # Build sample_token -> ego_pose mapping via sample_data (using LIDAR)
    sample_to_ego = {}
    for sd in sample_data:
        if 'LIDAR' in sd.get('filename', ''):
            sample_to_ego[sd['sample_token']] = sd['ego_pose_token']

    # Build sample_token -> annotations mapping
    sample_to_anns = {}
    for ann in annotations:
        st = ann['sample_token']
        if st not in sample_to_anns:
            sample_to_anns[st] = []
        sample_to_anns[st].append(ann)

    # Build timestamp -> sample_token mapping
    ts_to_sample = {s['timestamp']: s['token'] for s in samples}

    return {
        'sample_to_anns': sample_to_anns,
        'sample_to_ego': sample_to_ego,
        'ego_pose_by_token': ego_pose_by_token,
        'instance_by_token': instance_by_token,
        'category_by_token': category_by_token,
        'ts_to_sample': ts_to_sample,
    }


# Cache for loaded annotations per data_root
_ann_cache = {}


def quat_to_rotation_matrix(q):
    """Convert quaternion [w, x, y, z] to 3x3 rotation matrix."""
    w, x, y, z = q
    return np.array([
        [1 - 2*y*y - 2*z*z, 2*x*y - 2*z*w, 2*x*z + 2*y*w],
        [2*x*y + 2*z*w, 1 - 2*x*x - 2*z*z, 2*y*z - 2*x*w],
        [2*x*z - 2*y*w, 2*y*z + 2*x*w, 1 - 2*x*x - 2*y*y]
    ])


def get_bboxes_in_ego(timestamp, ann_data):
    """Get 3D bboxes in ego frame for given timestamp."""
    ts_to_sample = ann_data['ts_to_sample']
    sample_to_anns = ann_data['sample_to_anns']
    sample_to_ego = ann_data['sample_to_ego']
    ego_pose_by_token = ann_data['ego_pose_by_token']
    instance_by_token = ann_data['instance_by_token']
    category_by_token = ann_data['category_by_token']

    # Find sample token from timestamp
    if timestamp not in ts_to_sample:
        return []

    sample_token = ts_to_sample[timestamp]

    # Get annotations for this sample
    anns = sample_to_anns.get(sample_token, [])
    if not anns:
        return []

    # Get ego pose
    ego_token = sample_to_ego.get(sample_token)
    if not ego_token:
        return []
    ego_pose = ego_pose_by_token.get(ego_token)
    if not ego_pose:
        return []

    ego_trans = np.array(ego_pose['translation'])
    ego_rot = quat_to_rotation_matrix(ego_pose['rotation'])
    ego_rot_inv = ego_rot.T

    bboxes = []
    for ann in anns:
        # Get category
        inst = instance_by_token.get(ann['instance_token'], {})
        cat_token = inst.get('category_token', '')
        cat = category_by_token.get(cat_token, {})
        cat_name = cat.get('name', 'unknown')

        # Transform to ego frame
        global_trans = np.array(ann['translation'])
        local_trans = ego_rot_inv @ (global_trans - ego_trans)

        # Get size [width, length, height]
        size = ann['size']  # [w, l, h]

        # Get rotation (global) and transform to ego frame
        ann_rot = quat_to_rotation_matrix(ann['rotation'])
        local_rot = ego_rot_inv @ ann_rot

        # Get yaw from rotation matrix
        yaw = np.arctan2(local_rot[1, 0], local_rot[0, 0])

        bboxes.append({
            'center': local_trans[:2],  # x, y in ego frame
            'size': [size[1], size[0]],  # length, width
            'yaw': yaw,
            'category': cat_name,
        })

    return bboxes


def draw_bboxes_on_bev(bev_img, bboxes, vol_range=[-40, -40, 40, 40], output_size=(600, 600)):
    """Draw rotated bboxes on BEV image.

    BEV image orientation (with flip_vertical=True, flip_horizontal=True):
    - Up in image = forward in ego (positive X)
    - Left in image = left in ego (positive Y)
    """
    h, w = bev_img.shape[:2]

    # BEV coordinate mapping
    x_min, y_min, x_max, y_max = vol_range
    scale_x = h / (x_max - x_min)  # ego X -> image row
    scale_y = w / (y_max - y_min)  # ego Y -> image col

    # Category colors (BGR)
    cat_colors = {
        'car': (0, 255, 0),        # Green
        'truck': (0, 200, 0),      # Dark green
        'bus': (0, 150, 0),        # Darker green
        'pedestrian': (0, 0, 255), # Red
        'bicycle': (255, 0, 0),    # Blue
        'motorcycle': (255, 100, 0), # Light blue
        'trailer': (0, 100, 0),    # Very dark green
    }

    # Helper to convert ego coords to pixel coords
    def ego_to_pixel(ex, ey):
        py = int((x_max - ex) * scale_x)  # forward -> up
        px = int((y_max - ey) * scale_y)  # left -> left
        return px, py

    for bbox in bboxes:
        cx, cy = bbox['center']  # ego frame: cx=forward, cy=left
        length, width = bbox['size']
        yaw = bbox['yaw']
        cat = bbox['category']

        # Skip if center is behind ego (only show front area)
        if cx < 0:
            continue

        # Skip if center is outside range
        if cx < x_min or cx > x_max or cy < y_min or cy > y_max:
            continue

        # Box corners in local frame (length along X, width along Y)
        half_l, half_w = length / 2, width / 2
        corners = np.array([
            [half_l, -half_w],   # front-right
            [half_l, half_w],    # front-left
            [-half_l, half_w],   # rear-left
            [-half_l, -half_w],  # rear-right
        ])

        # Rotate corners by yaw
        cos_yaw, sin_yaw = np.cos(yaw), np.sin(yaw)
        rot_mat = np.array([[cos_yaw, -sin_yaw], [sin_yaw, cos_yaw]])
        corners_rot = corners @ rot_mat.T

        # Convert to pixel and clip to image bounds
        corners_px = []
        for corner in corners_rot:
            ex = cx + corner[0]
            ey = cy + corner[1]
            px, py = ego_to_pixel(ex, ey)
            # Clip to image bounds
            px = max(0, min(w - 1, px))
            py = max(0, min(h - 1, py))
            corners_px.append([px, py])

        corners_px = np.array(corners_px, dtype=np.int32)

        # Draw
        color = cat_colors.get(cat, (128, 128, 128))
        cv2.polylines(bev_img, [corners_px], True, color, 2)

    return bev_img


def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description='T4 Inference and Visualization')
    parser.add_argument('--checkpoint', '-c', type=str, required=True,
                        help='Path to checkpoint file')
    parser.add_argument('--config', type=str, default=None,
                        help='Path to config YAML (uses checkpoint dir config.yaml if not specified)')
    parser.add_argument('--num-samples', '-n', type=int, default=10,
                        help='Number of samples to visualize')
    parser.add_argument('--output-dir', '-o', type=str, default=None,
                        help='Output directory (default: checkpoint_dir/visualizations)')
    parser.add_argument('--batch-size', type=int, default=1,
                        help='Batch size for inference')
    parser.add_argument('--device', type=str, default='cuda:0',
                        help='Device to run inference on')
    parser.add_argument('--split', type=str, default='val', choices=['train', 'val', 'test'],
                        help='Dataset split to use (default: val)')
    return parser.parse_args()


def load_model_and_config(checkpoint_path: str, config_path: str = None, device: str = 'cuda:0'):
    """Load model from checkpoint and config."""
    from models import GaussTRLightning

    # Find config
    if config_path is None:
        # Check same directory as checkpoint
        ckpt_dir = os.path.dirname(checkpoint_path)
        config_path = os.path.join(ckpt_dir, 'config.yaml')
        if not os.path.exists(config_path):
            raise ValueError(f"Config not found at {config_path}. Please specify --config")

    print(f"Loading config from: {config_path}")
    cfg = OmegaConf.load(config_path)

    # Build model config
    model_cfg = OmegaConf.to_container(cfg.model, resolve=True)

    # Load checkpoint
    print(f"Loading checkpoint: {checkpoint_path}")
    if checkpoint_path.endswith('.ckpt'):
        model = GaussTRLightning.load_from_checkpoint(checkpoint_path, **model_cfg)
    else:
        model = GaussTRLightning(**model_cfg)
        ckpt = torch.load(checkpoint_path, map_location='cpu')
        state = ckpt.get('state_dict', ckpt)
        state = {k.removeprefix('model.'): v for k, v in state.items()}
        model.load_state_dict(state, strict=False)

    model = model.to(device).eval()
    return model, cfg


def create_t4_dataloader(cfg, num_samples: int, batch_size: int = 1, split: str = 'val'):
    """Create T4 dataloader for specified split."""
    from dataset import GaussTRDataModule

    data_cfg = OmegaConf.to_container(cfg.data, resolve=True)

    # Ensure dataset_type is t4
    data_cfg['dataset_type'] = 't4'

    # Create datamodule
    datamodule = GaussTRDataModule(**data_cfg)

    # Setup based on split
    if split == 'train':
        datamodule.setup('fit')
        dataset = datamodule.train_dataset
        full_dataset = datamodule.train_dataset
    else:
        datamodule.setup('test')
        dataset = datamodule.test_dataset
        full_dataset = datamodule.test_dataset

    # Limit to num_samples
    if num_samples > 0 and num_samples < len(dataset):
        # Create subset
        indices = list(range(0, len(dataset), len(dataset) // num_samples))[:num_samples]
        from torch.utils.data import Subset
        dataset = Subset(dataset, indices)

    # Create dataloader
    from torch.utils.data import DataLoader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    return dataloader, full_dataset


def run_inference(model, batch, device):
    """Run model inference on a batch."""
    # Move tensors to device
    images = batch['images'].to(device)
    feats = batch['feats'].to(device)
    depth = batch['depth'].to(device)
    cam2img = batch['cam2img'].to(device)
    cam2ego = batch['cam2ego'].to(device)
    img_aug_mat = batch.get('img_aug_mat')
    if img_aug_mat is not None:
        img_aug_mat = img_aug_mat.to(device)

    with torch.no_grad():
        # Forward pass
        outputs = model(
            images=images,
            feats=feats,
            depth=depth,
            cam2img=cam2img,
            cam2ego=cam2ego,
            img_aug_mat=img_aug_mat,
        )

    # In predict mode, output is directly the predictions tensor
    preds = outputs
    return preds


def create_multirow_legend(width: int, height: int, num_cols: int = 6) -> np.ndarray:
    """Create a multi-row legend with larger, clearer text."""
    from visualization.bev import OCC_CLASS_NAMES, OCC_COLORS

    legend = np.full((height, width, 3), 255, dtype=np.uint8)

    # Exclude 'free' class
    classes = [(name, color) for name, color in zip(OCC_CLASS_NAMES, OCC_COLORS) if name != 'free']
    n_classes = len(classes)
    num_rows = (n_classes + num_cols - 1) // num_cols

    cell_w = width // num_cols
    cell_h = height // num_rows
    box_size = min(cell_h - 10, 50)
    font_scale = 0.9
    font_thickness = 2

    for i, (name, color) in enumerate(classes):
        row = i // num_cols
        col = i % num_cols

        x_start = col * cell_w + 8
        y_start = row * cell_h + (cell_h - box_size) // 2

        # Draw color box
        cv2.rectangle(legend, (x_start, y_start), (x_start + box_size, y_start + box_size),
                      tuple(int(c) for c in color), -1)
        cv2.rectangle(legend, (x_start, y_start), (x_start + box_size, y_start + box_size),
                      (0, 0, 0), 1)

        # Draw class name
        text_x = x_start + box_size + 8
        text_y = y_start + box_size - 5
        cv2.putText(legend, name, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale, (0, 0, 0), font_thickness)

    return legend


def visualize_sample(
    pred: np.ndarray,
    images: dict,
    output_path: str,
    title: str = None,
    data_root: str = None,
    timestamp: int = None,
):
    """Create visualization for a single sample."""
    from visualization import (
        draw_bev_occupancy,
        render_occupancy_3d_to_array,
    )

    # Create output directory
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Create BEV visualization (forward=up, left=left, ego at bottom)
    # Crop to front half only (x >= 0) so ego is at bottom of image
    # pred shape is (X, Y, Z), X is forward direction
    # vol_range is [-40, -40, -1, 40, 40, 5.4] with voxel_size=0.4
    # So X dimension has 200 voxels, take second half (100:) for front
    pred_front = pred[pred.shape[0]//2:, :, :]  # front half only
    bev_img = draw_bev_occupancy(pred_front, output_size=(600, 600), rotate_deg=0, flip_vertical=True, flip_horizontal=True, draw_ego=False)

    # Load annotations and draw bboxes on BEV (front half range: x from 0 to 40)
    front_vol_range = [0, -40, 40, 40]  # x_min, y_min, x_max, y_max
    if data_root and timestamp:
        # Use cached annotations or load new
        if data_root not in _ann_cache:
            _ann_cache[data_root] = load_t4_annotations(data_root)

        ann_data = _ann_cache.get(data_root)
        if ann_data:
            bboxes = get_bboxes_in_ego(timestamp, ann_data)
            if bboxes:
                bev_img = draw_bboxes_on_bev(bev_img, bboxes, vol_range=front_vol_range)

    # Create 3D render (larger)
    try:
        img_3d = render_occupancy_3d_to_array(pred, image_size=(2000, 1500))
    except Exception as e:
        print(f"  3D render failed: {e}")
        img_3d = None

    # Create composite: Row 1: camera, Row 2: 3D (left) + BEV with legend (right)
    if images and 'CAM_FRONT_WIDE' in images and img_3d is not None:
        cam_img = images['CAM_FRONT_WIDE']
        cam_h, cam_w = cam_img.shape[:2]

        # Row 2 height = 3D height
        row2_h = img_3d.shape[0]  # 1500

        # Resize BEV to match 3D height
        bev_h = row2_h
        bev_w = bev_h  # Keep square
        bev_resized = cv2.resize(bev_img, (bev_w, bev_h), interpolation=cv2.INTER_LINEAR)

        # Create legend below BEV
        legend_h = 200
        legend = create_multirow_legend(width=bev_w, height=legend_h, num_cols=3)

        # Stack BEV and legend
        bev_with_legend = np.concatenate([bev_resized, legend], axis=0)

        # Row 2: 3D on left, BEV+legend on right
        row2_w = img_3d.shape[1] + bev_w
        row2_total_h = max(img_3d.shape[0], bev_with_legend.shape[0])
        row2 = np.ones((row2_total_h, row2_w, 3), dtype=np.uint8) * 255
        row2[:img_3d.shape[0], :img_3d.shape[1]] = img_3d
        row2[:bev_with_legend.shape[0], img_3d.shape[1]:] = bev_with_legend

        # Resize camera to match row2 width
        cam_resized_w = row2_w
        cam_resized_h = int(cam_h * cam_resized_w / cam_w)
        cam_resized = cv2.resize(cam_img, (cam_resized_w, cam_resized_h), interpolation=cv2.INTER_LINEAR)

        # Stack rows vertically
        composite = np.concatenate([cam_resized, row2], axis=0)

        # Add title if provided
        if title:
            cv2.putText(composite, title, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        cv2.imwrite(output_path.replace('.jpg', '_composite.jpg'), composite)

    # Save BEV
    cv2.imwrite(output_path.replace('.jpg', '_bev.jpg'), bev_img)

    # Save 3D
    if img_3d is not None:
        cv2.imwrite(output_path.replace('.jpg', '_3d.jpg'), img_3d)

    return output_path


def load_camera_images_t4(data_info: dict, data_root: str) -> dict:
    """Load camera images for T4 dataset."""
    images = {}

    if 'images' not in data_info:
        return images

    for cam_name, cam_info in data_info['images'].items():
        if cam_name.startswith('CAM_'):
            img_path = cam_info.get('img_path', '')
            if img_path and os.path.exists(img_path):
                img = cv2.imread(img_path)
                if img is not None:
                    images[cam_name] = img

    return images


def main():
    args = parse_args()

    # Load model and config
    model, cfg = load_model_and_config(args.checkpoint, args.config, args.device)

    # Setup output directory
    if args.output_dir is None:
        args.output_dir = os.path.join(os.path.dirname(args.checkpoint), 'visualizations')
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Output directory: {args.output_dir}")

    # Create dataloader
    print(f"Loading T4 {args.split} data...")
    dataloader, full_dataset = create_t4_dataloader(cfg, args.num_samples, args.batch_size, args.split)
    print(f"Processing {len(dataloader)} batches")

    # Get data root for loading camera images
    data_root = cfg.data.data_root

    # Run inference
    sample_idx = 0
    for batch_idx, batch in enumerate(tqdm(dataloader, desc="Inference")):
        # Run model
        preds = run_inference(model, batch, args.device)

        if preds is None:
            print(f"  Batch {batch_idx}: No predictions")
            continue

        # Process each sample in batch
        preds_np = preds.cpu().numpy()
        batch_size = preds_np.shape[0]

        for i in range(batch_size):
            pred = preds_np[i]

            # Get sample index for camera images
            if hasattr(dataloader.dataset, 'indices'):
                # Subset
                real_idx = dataloader.dataset.indices[sample_idx]
            else:
                real_idx = sample_idx

            # Load camera images and get data_root for this sample
            sample_data_root = None
            timestamp = None
            try:
                data_info = full_dataset.get_data_info(real_idx)
                images = load_camera_images_t4(data_info, data_root)
                # Get sample's data_root from image path
                if 'images' in data_info:
                    for cam_info in data_info['images'].values():
                        img_path = cam_info.get('img_path', '')
                        if img_path:
                            # data_root is parent of 'data' folder
                            # e.g., /path/to/t4_dataset/data/CAM_FRONT/xxx.jpg -> /path/to/t4_dataset
                            parts = img_path.split('/data/')
                            if len(parts) > 1:
                                sample_data_root = parts[0]
                            break
                # Get timestamp
                timestamp = data_info.get('timestamp')
            except Exception as e:
                print(f"  Sample {sample_idx}: Could not load camera images: {e}")
                images = {}

            # Get sample identifier
            if 'timestamp' in batch:
                ts = batch['timestamp'][i]
                if isinstance(ts, torch.Tensor):
                    ts = ts.item()
                sample_id = f"{ts:.6f}"
                if timestamp is None:
                    timestamp = int(ts)
            else:
                sample_id = f"{sample_idx:06d}"

            # Visualize
            output_path = os.path.join(args.output_dir, f'{sample_id}.jpg')
            visualize_sample(
                pred, images, output_path,
                title=f"Sample {sample_id}",
                data_root=sample_data_root,
                timestamp=timestamp,
            )

            sample_idx += 1

    print(f"\nVisualization complete!")
    print(f"  Output: {args.output_dir}")
    print(f"  Total samples: {sample_idx}")


if __name__ == '__main__':
    main()
