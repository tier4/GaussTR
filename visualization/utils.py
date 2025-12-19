"""Utility functions for visualization.

Data loading and preprocessing helpers.
"""

import os
import pickle
import cv2
import numpy as np
from typing import Dict, List, Optional, Tuple, Any


# Camera names for nuScenes
CAMERA_NAMES = [
    'CAM_FRONT', 'CAM_FRONT_LEFT', 'CAM_FRONT_RIGHT',
    'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT'
]


def load_gt_occupancy(
    occ_path: str,
) -> Dict[str, np.ndarray]:
    """Load ground truth occupancy data.

    Args:
        occ_path: Path to occupancy directory containing labels.npz.

    Returns:
        Dictionary with:
            - semantics: (X, Y, Z) semantic labels
            - mask_lidar: (X, Y, Z) lidar visibility mask
            - mask_camera: (X, Y, Z) camera visibility mask
    """
    labels_path = os.path.join(occ_path, 'labels.npz')
    if not os.path.exists(labels_path):
        raise FileNotFoundError(f"Occupancy labels not found: {labels_path}")

    data = np.load(labels_path)

    result = {
        'semantics': data['semantics'],
    }

    # Load masks if available
    if 'mask_lidar' in data:
        result['mask_lidar'] = data['mask_lidar'].astype(bool)
    if 'mask_camera' in data:
        result['mask_camera'] = data['mask_camera'].astype(bool)

    return result


def load_camera_images(
    data_info: Dict[str, Any],
    data_root: str = 'data/nuscenes',
    target_size: Optional[Tuple[int, int]] = None,
) -> Dict[str, np.ndarray]:
    """Load camera images for a sample.

    Args:
        data_info: Sample info dictionary from annotation file.
        data_root: Root directory for nuScenes data.
        target_size: Optional target size (width, height) for resizing.

    Returns:
        Dictionary mapping camera names to BGR images.
    """
    images = {}

    # Handle different annotation formats
    if 'images' in data_info:
        # V2 format
        for cam_name, cam_info in data_info['images'].items():
            img_path = cam_info.get('img_path', '')
            if img_path:
                # Try path as-is first
                if not os.path.exists(img_path):
                    # Try with data_root prepended
                    img_path = os.path.join(data_root, img_path)

                if os.path.exists(img_path):
                    img = cv2.imread(img_path)
                    if img is not None:
                        if target_size is not None:
                            img = cv2.resize(img, target_size, interpolation=cv2.INTER_LINEAR)
                        images[cam_name] = img

    elif 'cams' in data_info:
        # Old format (e.g., BEVDet annotations)
        for cam_name, cam_info in data_info['cams'].items():
            img_path = cam_info.get('data_path', '')
            if img_path:
                # Try path as-is first
                if not os.path.exists(img_path):
                    img_path = os.path.join(data_root, img_path)

                if os.path.exists(img_path):
                    img = cv2.imread(img_path)
                    if img is not None:
                        if target_size is not None:
                            img = cv2.resize(img, target_size, interpolation=cv2.INTER_LINEAR)
                        images[cam_name] = img

    return images


def get_scene_info(
    data_info: Dict[str, Any],
) -> Dict[str, Any]:
    """Extract scene information from data info.

    Args:
        data_info: Sample info dictionary.

    Returns:
        Dictionary with scene_name, token, etc.
    """
    return {
        'scene_name': data_info.get('scene_name', 'unknown'),
        'scene_idx': data_info.get('scene_idx', 0),
        'token': data_info.get('token', data_info.get('sample_idx', '')),
        'timestamp': data_info.get('timestamp', 0),
        'lidar_path': data_info.get('lidar_path', ''),
    }


def load_annotations(
    ann_file: str,
) -> List[Dict[str, Any]]:
    """Load annotation file.

    Args:
        ann_file: Path to annotation pickle file.

    Returns:
        List of data info dictionaries.
    """
    with open(ann_file, 'rb') as f:
        data = pickle.load(f)

    if isinstance(data, dict):
        if 'data_list' in data:
            return data['data_list']
        elif 'infos' in data:
            return data['infos']
        else:
            raise ValueError(f"Unknown annotation format")
    elif isinstance(data, list):
        return data
    else:
        raise ValueError(f"Unknown annotation format")


def get_occ_path(
    data_info: Dict[str, Any],
    data_root: str = 'data/nuscenes',
) -> str:
    """Get path to occupancy ground truth directory.

    Args:
        data_info: Sample info dictionary.
        data_root: Root directory for nuScenes data.

    Returns:
        Path to occupancy directory.
    """
    if 'occ_path' in data_info:
        occ_path = data_info['occ_path']
        if not os.path.isabs(occ_path):
            occ_path = os.path.join(data_root, occ_path)
        return occ_path

    scene_idx = data_info.get('scene_idx', 0)
    token = data_info.get('token', data_info.get('sample_idx', ''))
    return os.path.join(data_root, 'gts', str(scene_idx), token)


def create_output_dirs(
    base_dir: str,
    subdirs: List[str] = None,
) -> Dict[str, str]:
    """Create output directories for visualization.

    Args:
        base_dir: Base output directory.
        subdirs: List of subdirectory names.

    Returns:
        Dictionary mapping subdir names to full paths.
    """
    if subdirs is None:
        subdirs = ['bev', '3d', 'composite', 'video']

    dirs = {'base': base_dir}
    os.makedirs(base_dir, exist_ok=True)

    for subdir in subdirs:
        path = os.path.join(base_dir, subdir)
        os.makedirs(path, exist_ok=True)
        dirs[subdir] = path

    return dirs


def compute_metrics(
    pred_occ: np.ndarray,
    gt_occ: np.ndarray,
    mask: Optional[np.ndarray] = None,
    num_classes: int = 18,
    ignore_index: int = 17,
) -> Dict[str, float]:
    """Compute IoU metrics for visualization overlay.

    Args:
        pred_occ: Predicted occupancy (X, Y, Z).
        gt_occ: Ground truth occupancy (X, Y, Z).
        mask: Optional visibility mask.
        num_classes: Number of classes.
        ignore_index: Class index to ignore.

    Returns:
        Dictionary of metrics.
    """
    if mask is not None:
        pred = pred_occ[mask]
        gt = gt_occ[mask]
    else:
        pred = pred_occ.flatten()
        gt = gt_occ.flatten()

    # Remove ignored class
    valid = gt != ignore_index
    pred = pred[valid]
    gt = gt[valid]

    # Fast confusion matrix using numpy (vectorized)
    valid_mask = (pred < num_classes) & (gt < num_classes)
    pred = pred[valid_mask]
    gt = gt[valid_mask]
    conf_matrix = np.bincount(
        num_classes * gt.astype(np.int64) + pred.astype(np.int64),
        minlength=num_classes * num_classes
    ).reshape(num_classes, num_classes)

    # Compute IoU per class
    iou_per_class = []
    for c in range(num_classes):
        if c == ignore_index:
            continue
        tp = conf_matrix[c, c]
        fp = conf_matrix[:, c].sum() - tp
        fn = conf_matrix[c, :].sum() - tp

        if tp + fp + fn > 0:
            iou = tp / (tp + fp + fn)
            iou_per_class.append(iou)

    miou = np.mean(iou_per_class) if iou_per_class else 0.0

    return {
        'mIoU': miou,
        'num_classes': len(iou_per_class),
    }


def tensor_to_numpy(tensor) -> np.ndarray:
    """Convert PyTorch tensor to numpy array.

    Args:
        tensor: PyTorch tensor or numpy array.

    Returns:
        Numpy array.
    """
    if hasattr(tensor, 'cpu'):
        return tensor.cpu().numpy()
    return np.asarray(tensor)
