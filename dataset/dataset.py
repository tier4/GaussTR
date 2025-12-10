"""NuScenes Occupancy Dataset for GaussTR Lightning.

Pure PyTorch Dataset implementation without MMEngine/MMDet3D dependencies.
"""

import os
import pickle
from typing import Dict, List, Callable, Optional, Any

import numpy as np
import torch
from torch.utils.data import Dataset


# Occupancy class metadata
OCC_CLASSES = (
    'others', 'barrier', 'bicycle', 'bus', 'car', 'construction_vehicle',
    'motorcycle', 'pedestrian', 'traffic_cone', 'trailer', 'truck',
    'driveable_surface', 'other_flat', 'sidewalk', 'terrain', 'manmade',
    'vegetation', 'free'
)

LABEL2CAT = {i: cat for i, cat in enumerate(OCC_CLASSES)}


class NuScenesOccDataset(Dataset):
    """Pure PyTorch Dataset for nuScenes occupancy prediction.

    Args:
        ann_file: Path to annotation pickle file.
        data_root: Root directory for nuScenes data.
        transforms: Transform pipeline to apply.
        test_mode: Whether in test mode (no ground truth loading).
    """

    def __init__(
        self,
        ann_file: str,
        data_root: str = 'data/nuscenes',
        transforms: Optional[Callable] = None,
        test_mode: bool = False,
    ):
        self.ann_file = ann_file
        self.data_root = data_root
        self.transforms = transforms
        self.test_mode = test_mode

        # Load annotations
        self.data_infos = self._load_annotations(ann_file)

        # Metadata
        self.metainfo = {
            'classes': OCC_CLASSES,
            'label2cat': LABEL2CAT,
        }

    def _load_annotations(self, ann_file: str) -> List[Dict]:
        """Load annotations from pickle file.

        Args:
            ann_file: Path to annotation file.

        Returns:
            List of data info dictionaries.
        """
        with open(ann_file, 'rb') as f:
            data = pickle.load(f)

        # Handle both old and new annotation formats
        if isinstance(data, dict):
            if 'data_list' in data:
                return data['data_list']
            elif 'infos' in data:
                return data['infos']
            else:
                raise ValueError(f"Unknown annotation format in {ann_file}")
        elif isinstance(data, list):
            return data
        else:
            raise ValueError(f"Unknown annotation format in {ann_file}")

    def __len__(self) -> int:
        return len(self.data_infos)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get a single sample.

        Args:
            idx: Sample index.

        Returns:
            Transformed sample dictionary.
        """
        data_info = self.get_data_info(idx)

        # Apply transforms
        if self.transforms is not None:
            data = self.transforms(data_info)
            if data is None:
                # Transform failed, try next sample
                return self.__getitem__((idx + 1) % len(self))
            return data
        return data_info

    def get_data_info(self, idx: int) -> Dict[str, Any]:
        """Get raw data info for a sample.

        Args:
            idx: Sample index.

        Returns:
            Data info dictionary with image paths and camera parameters.
        """
        info = self.data_infos[idx].copy()

        # Ensure images dict has full paths
        if 'images' in info:
            for cam_name, cam_info in info['images'].items():
                img_path = cam_info['img_path']
                # Skip if already absolute or already contains data_root
                if not os.path.isabs(img_path) and not img_path.startswith(self.data_root):
                    cam_info['img_path'] = os.path.join(
                        self.data_root, img_path
                    )

        # Build occupancy ground truth path
        if 'scene_idx' in info and 'token' in info:
            info['occ_path'] = os.path.join(
                self.data_root,
                f"gts/{info['scene_idx']}/{info['token']}"
            )
        elif 'occ_path' not in info:
            # Try to construct from sample_idx if available
            sample_idx = info.get('sample_idx', info.get('token', ''))
            scene_idx = info.get('scene_idx', 0)
            info['occ_path'] = os.path.join(
                self.data_root,
                f"gts/{scene_idx}/{sample_idx}"
            )

        return info


class NuScenesOccDatasetV2(NuScenesOccDataset):
    """Dataset variant for OpenMMLab V2.0 annotation format.

    This handles the newer annotation format used in recent MMDet3D versions.
    """

    def get_data_info(self, idx: int) -> Dict[str, Any]:
        """Get raw data info for a sample (V2 format).

        Args:
            idx: Sample index.

        Returns:
            Data info dictionary.
        """
        info = self.data_infos[idx].copy()

        # V2 format stores images in 'images' dict with cam names as keys
        if 'images' in info:
            for cam_name, cam_info in info['images'].items():
                img_path = cam_info.get('img_path', '')
                # Skip if already absolute or already contains data_root
                if img_path and not os.path.isabs(img_path) and not img_path.startswith(self.data_root):
                    # Check if path already contains 'samples/' (some annotations have full relative paths)
                    if 'samples/' not in img_path:
                        cam_info['img_path'] = os.path.join(
                            self.data_root, 'samples', cam_name, img_path
                        )
                    else:
                        cam_info['img_path'] = os.path.join(
                            self.data_root, img_path
                        )

                # Ensure cam2img is 3x3
                if 'cam2img' in cam_info:
                    cam2img = np.array(cam_info['cam2img'])
                    if cam2img.shape == (4, 4):
                        cam_info['cam2img'] = cam2img[:3, :3].tolist()

        # Build occ path
        if 'scene_idx' in info:
            token = info.get('token', info.get('sample_idx', ''))
            info['occ_path'] = os.path.join(
                self.data_root,
                f"gts/{info['scene_idx']}/{token}"
            )

        return info


def create_nuscenes_dataset(
    ann_file: str,
    data_root: str = 'data/nuscenes',
    transforms: Optional[Callable] = None,
    test_mode: bool = False,
    use_v2_format: bool = True,
) -> Dataset:
    """Factory function to create NuScenes occupancy dataset.

    Args:
        ann_file: Path to annotation file.
        data_root: Root directory.
        transforms: Transform pipeline.
        test_mode: Whether in test mode.
        use_v2_format: Whether to use V2 annotation format.

    Returns:
        Dataset instance.
    """
    dataset_cls = NuScenesOccDatasetV2 if use_v2_format else NuScenesOccDataset
    return dataset_cls(
        ann_file=ann_file,
        data_root=data_root,
        transforms=transforms,
        test_mode=test_mode,
    )
