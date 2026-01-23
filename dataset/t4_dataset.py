"""T4 Dataset for GaussTR Lightning.

Loads T4 annotations and camera data without occupancy ground truth.
"""

import os
import pickle
from typing import Dict, List, Callable, Optional, Any

from torch.utils.data import Dataset

T4_CAMERA_NAMES = [
    "CAM_FRONT",
    "CAM_FRONT_LEFT_WIDE",
    "CAM_FRONT_RIGHT_WIDE",
    "CAM_BACK_LEFT_WIDE",
    "CAM_BACK_RIGHT_WIDE",
    "CAM_FRONT_WIDE",
]


class T4Dataset(Dataset):
    """PyTorch Dataset for the internal T4 dataset.

    Args:
        ann_file: Path to annotation pickle file.
        data_root: Root directory for T4 data.
        transforms: Transform pipeline to apply.
        test_mode: Whether in test mode.
        camera_names: Ordered list of camera names to load.
    """

    def __init__(
        self,
        ann_file: str,
        data_root: str = '/mnt/nvme2/T4_datasets',
        transforms: Optional[Callable] = None,
        test_mode: bool = False,
        camera_names: Optional[List[str]] = None,
    ):
        self.ann_file = ann_file
        self.data_root = data_root
        self.transforms = transforms
        self.test_mode = test_mode
        self.camera_names = camera_names or T4_CAMERA_NAMES

        self.data_infos = self._load_annotations(ann_file)

        # No occupancy GT for T4
        self.metainfo = {
            'classes': [],
            'has_gt': False,
        }

    def _load_annotations(self, ann_file: str) -> List[Dict]:
        with open(ann_file, 'rb') as f:
            data = pickle.load(f)

        if isinstance(data, dict):
            if 'data_list' in data:
                return data['data_list']
            if 'infos' in data:
                return data['infos']
            raise ValueError(f"Unknown annotation format in {ann_file}")
        if isinstance(data, list):
            return data
        raise ValueError(f"Unknown annotation format in {ann_file}")

    def __len__(self) -> int:
        return len(self.data_infos)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        data_info = self.get_data_info(idx)

        if self.transforms is not None:
            data = self.transforms(data_info)
            if data is None:
                return self.__getitem__((idx + 1) % len(self))
            return data
        return data_info

    def get_data_info(self, idx: int) -> Dict[str, Any]:
        info = self.data_infos[idx].copy()

        if 'images' in info:
            ordered_images = {
                name: info['images'][name]
                for name in self.camera_names
                if name in info['images']
            }
            info['images'] = ordered_images

            for cam_info in info['images'].values():
                img_path = cam_info.get('img_path', '')
                if img_path and not os.path.isabs(img_path) and not img_path.startswith(self.data_root):
                    cam_info['img_path'] = os.path.join(self.data_root, img_path)

        lidar_path = info.get('lidar_path', '')
        if lidar_path and not os.path.isabs(lidar_path) and not lidar_path.startswith(self.data_root):
            info['lidar_path'] = os.path.join(self.data_root, lidar_path)

        return info


def create_t4_dataset(
    ann_file: str,
    data_root: str = '/mnt/nvme2/T4_datasets',
    transforms: Optional[Callable] = None,
    test_mode: bool = False,
    camera_names: Optional[List[str]] = None,
) -> Dataset:
    """Factory function to create T4 dataset."""
    return T4Dataset(
        ann_file=ann_file,
        data_root=data_root,
        transforms=transforms,
        test_mode=test_mode,
        camera_names=camera_names,
    )
