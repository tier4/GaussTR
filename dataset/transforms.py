"""Data transforms for GaussTR Lightning.

Pure PyTorch implementations without MMEngine/MMCV dependencies.
Uses OpenCV for faster image operations instead of PIL.
"""

import os
from typing import Dict, List, Tuple, Optional, Any, Callable

import cv2
import numpy as np
import torch
import torch.nn.functional as F


class Compose:
    """Compose multiple transforms together."""

    def __init__(self, transforms: List[Callable]):
        self.transforms = transforms

    def __call__(self, data: Dict) -> Dict:
        for t in self.transforms:
            data = t(data)
            if data is None:
                return None
        return data


class LoadMultiViewImages:
    """Load multi-view images from file paths.

    Args:
        to_float32: Whether to convert images to float32.
        num_views: Number of camera views.
        data_root: Root directory for data.
    """

    def __init__(self, to_float32: bool = True, num_views: int = 6, data_root: str = 'data/nuscenes'):
        self.to_float32 = to_float32
        self.num_views = num_views
        self.data_root = data_root

    def __call__(self, results: Dict) -> Dict:
        """Load images and camera parameters.

        Args:
            results: Dictionary containing 'images' with camera info.

        Returns:
            Updated results dictionary.
        """
        filenames = []
        cam2imgs = []
        cam2egos = []
        lidar2cams = []
        imgs = []

        for cam_name, cam_item in results['images'].items():
            # Dataset already provides full path in img_path
            img_path = cam_item['img_path']
            filenames.append(img_path)

            # Load image with OpenCV (faster than PIL)
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # OpenCV loads as BGR
            imgs.append(img)

            # Camera intrinsics (3x3 -> 4x4)
            cam2img = np.eye(4, dtype=np.float32)
            cam2img[:3, :3] = np.array(cam_item['cam2img'], dtype=np.float32)
            cam2imgs.append(cam2img)

            # Camera extrinsics
            cam2ego = np.array(cam_item['cam2ego'], dtype=np.float32)
            cam2egos.append(cam2ego)

            # Lidar to camera (if available)
            if 'lidar2cam' in cam_item:
                lidar2cams.append(np.array(cam_item['lidar2cam'], dtype=np.float32))

        results['filename'] = filenames
        results['img_path'] = filenames
        results['img'] = imgs
        results['cam2img'] = np.stack(cam2imgs, axis=0)
        results['cam2ego'] = np.stack(cam2egos, axis=0)
        if lidar2cams:
            results['lidar2cam'] = np.stack(lidar2cams, axis=0)

        # Image shape info
        results['img_shape'] = imgs[0].shape[:2]
        results['ori_shape'] = imgs[0].shape[:2]
        results['num_views'] = self.num_views

        if self.to_float32:
            results['img'] = [img.astype(np.float32) for img in results['img']]

        return results


class ImageAug3D:
    """3D-aware image augmentation.

    Applies resize, crop, flip, and rotation while tracking the transformation
    matrix for updating camera parameters.

    Args:
        final_dim: Final image dimensions (H, W).
        resize_lim: Range for resize factor.
        bot_pct_lim: Range for bottom crop percentage.
        rot_lim: Range for rotation in degrees.
        rand_flip: Whether to apply random horizontal flip.
        is_train: Whether in training mode.
    """

    def __init__(
        self,
        final_dim: Tuple[int, int],
        resize_lim: Tuple[float, float],
        bot_pct_lim: Tuple[float, float] = (0.0, 0.0),
        rot_lim: Tuple[float, float] = (0.0, 0.0),
        rand_flip: bool = False,
        is_train: bool = False
    ):
        self.final_dim = final_dim
        self.resize_lim = resize_lim
        self.bot_pct_lim = bot_pct_lim
        self.rand_flip = rand_flip
        self.rot_lim = rot_lim
        self.is_train = is_train

    def sample_augmentation(self, results: Dict) -> Tuple:
        """Sample augmentation parameters."""
        H, W = results['ori_shape']
        fH, fW = self.final_dim

        if self.is_train:
            resize = np.random.uniform(*self.resize_lim)
            resize_dims = (int(W * resize), int(H * resize))
            newW, newH = resize_dims
            crop_h = int((1 - np.random.uniform(*self.bot_pct_lim)) * newH) - fH
            crop_w = int(np.random.uniform(0, max(0, newW - fW)))
            crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)
            flip = self.rand_flip and np.random.choice([0, 1])
            rotate = np.random.uniform(*self.rot_lim)
        else:
            resize = np.mean(self.resize_lim)
            resize_dims = (int(W * resize), int(H * resize))
            newW, newH = resize_dims
            crop_h = int((1 - np.mean(self.bot_pct_lim)) * newH) - fH
            crop_w = int(max(0, newW - fW) / 2)
            crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)
            flip = False
            rotate = 0

        return resize, resize_dims, crop, flip, rotate

    def img_transform(
        self,
        img: np.ndarray,
        rotation: torch.Tensor,
        translation: torch.Tensor,
        resize: float,
        resize_dims: Tuple[int, int],
        crop: Tuple[int, int, int, int],
        flip: bool,
        rotate: float
    ) -> Tuple[np.ndarray, torch.Tensor, torch.Tensor]:
        """Apply transform to image and update transformation matrix using OpenCV."""
        # Resize using OpenCV (much faster than PIL)
        img = img.astype(np.uint8)
        img = cv2.resize(img, resize_dims, interpolation=cv2.INTER_LINEAR)

        # Crop using array slicing
        x1, y1, x2, y2 = crop
        img = img[y1:y2, x1:x2]

        # Flip using OpenCV
        if flip:
            img = cv2.flip(img, 1)  # 1 = horizontal flip

        # Rotate using OpenCV warpAffine (10x faster than PIL rotate)
        if rotate != 0:
            h, w = img.shape[:2]
            center = (w / 2, h / 2)
            M = cv2.getRotationMatrix2D(center, rotate, 1.0)
            img = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR,
                                 borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))

        # Update transformation matrix
        rotation = rotation * resize
        translation = translation - torch.tensor(crop[:2], dtype=torch.float32)

        if flip:
            A = torch.tensor([[-1, 0], [0, 1]], dtype=torch.float32)
            b = torch.tensor([crop[2] - crop[0], 0], dtype=torch.float32)
            rotation = A @ rotation
            translation = A @ translation + b

        theta = rotate / 180 * np.pi
        A = torch.tensor([
            [np.cos(theta), np.sin(theta)],
            [-np.sin(theta), np.cos(theta)],
        ], dtype=torch.float32)
        b = torch.tensor([crop[2] - crop[0], crop[3] - crop[1]], dtype=torch.float32) / 2
        b = A @ (-b) + b
        rotation = A @ rotation
        translation = A @ translation + b

        return img, rotation, translation

    def __call__(self, data: Dict) -> Dict:
        """Apply augmentation to all views."""
        imgs = data['img']
        new_imgs = []
        transforms = []

        # Sample augmentation parameters ONCE for all views (not per-view)
        resize, resize_dims, crop, flip, rotate = self.sample_augmentation(data)

        for img in imgs:
            post_rot = torch.eye(2)
            post_tran = torch.zeros(2)

            new_img, rotation, translation = self.img_transform(
                img, post_rot, post_tran,
                resize=resize, resize_dims=resize_dims,
                crop=crop, flip=flip, rotate=rotate
            )

            transform = torch.eye(4)
            transform[:2, :2] = rotation
            transform[:2, 3] = translation
            new_imgs.append(new_img.astype(np.float32))
            transforms.append(transform.numpy())

        data['img'] = new_imgs
        data['img_aug_mat'] = transforms
        return data


class LoadFeatMaps:
    """Load pre-extracted feature maps.

    Args:
        data_root: Root directory containing feature maps.
        key: Key name for storing features in results.
        apply_aug: Whether to apply image augmentation to features.
        suffix: Optional suffix for feature filenames.
        use_mmap: Use memory-mapped loading for large datasets (reduces RAM usage).
    """

    def __init__(
        self,
        data_root: str,
        key: str,
        apply_aug: bool = False,
        suffix: str = '',
        use_mmap: bool = False
    ):
        self.data_root = data_root
        self.key = key
        self.apply_aug = apply_aug
        self.suffix = suffix
        self.use_mmap = use_mmap

    def __call__(self, results: Dict) -> Dict:
        """Load feature maps for all views."""
        feats = []
        img_aug_mats = results.get('img_aug_mat')

        for i, filename in enumerate(results['filename']):
            # Build feature path
            basename = os.path.basename(filename).split('.')[0]
            feat_path = os.path.join(self.data_root, basename + self.suffix + '.npy')

            # Use memory-mapped loading if enabled (lazy loading, OS handles caching)
            if self.use_mmap:
                feat = np.load(feat_path, mmap_mode='r')
                feat = np.array(feat)  # Copy to allow modification
            else:
                feat = np.load(feat_path)
            feat = torch.from_numpy(feat)

            # Apply augmentation if needed
            if self.apply_aug and img_aug_mats is not None:
                post_rot = img_aug_mats[i][:3, :3]
                post_tran = img_aug_mats[i][:3, 3]

                h, w = feat.shape[-2:]
                mode = 'nearest' if feat.dtype in [torch.long, torch.int] else 'bilinear'

                # Resize
                new_h = int(h * post_rot[1, 1] + 0.5)
                new_w = int(w * post_rot[0, 0] + 0.5)

                if feat.dim() == 2:
                    feat = feat.unsqueeze(0).unsqueeze(0)
                    feat = F.interpolate(feat.float(), (new_h, new_w), mode=mode)
                    feat = feat.squeeze(0).squeeze(0)
                else:
                    feat = feat.unsqueeze(0)
                    feat = F.interpolate(feat.float(), (new_h, new_w), mode=mode)
                    feat = feat.squeeze(0)

                # Crop
                start_h = int(post_tran[1])
                start_w = int(-post_tran[0])
                if feat.dim() == 2:
                    feat = feat[start_h:, start_w:]
                else:
                    feat = feat[:, start_h:, start_w:]

            feats.append(feat)

        results[self.key] = torch.stack(feats)
        return results


class LoadOccFromFile:
    """Load occupancy ground truth from file."""

    def __call__(self, results: Dict) -> Dict:
        """Load occupancy labels."""
        occ_path = os.path.join(results['occ_path'], 'labels.npz')
        occ_labels = np.load(occ_path)

        results['gt_semantic_seg'] = occ_labels['semantics']
        results['mask_lidar'] = occ_labels['mask_lidar']
        results['mask_camera'] = occ_labels['mask_camera']
        return results


class PackInputs:
    """Pack inputs into tensors for model.

    Args:
        keys: Keys to convert to tensors.
        meta_keys: Keys to keep as metadata.
    """

    def __init__(
        self,
        keys: List[str] = None,
        meta_keys: List[str] = None
    ):
        self.keys = keys or ['img']
        self.meta_keys = meta_keys or [
            'cam2img', 'cam2ego', 'ego2global', 'img_aug_mat',
            'sample_idx', 'token', 'timestamp', 'num_views', 'img_path', 'depth', 'feats', 'sem_seg'
        ]

    def __call__(self, results: Dict) -> Dict:
        """Pack results into tensor format."""
        packed = {}

        # Convert images to tensor
        if 'img' in results:
            imgs = results['img']
            # Stack images: [N, H, W, C] -> [N, C, H, W]
            imgs = np.stack(imgs, axis=0)
            imgs = torch.from_numpy(imgs).permute(0, 3, 1, 2)
            packed['images'] = imgs

        # Copy other keys
        for key in self.meta_keys:
            if key in results:
                val = results[key]
                if isinstance(val, np.ndarray):
                    val = torch.from_numpy(val)
                elif isinstance(val, list) and len(val) > 0:
                    if isinstance(val[0], np.ndarray):
                        val = torch.from_numpy(np.stack(val))
                packed[key] = val

        # Handle ground truth
        if 'gt_semantic_seg' in results:
            packed['gt_occ'] = torch.from_numpy(results['gt_semantic_seg'])
        if 'mask_camera' in results:
            packed['mask_camera'] = torch.from_numpy(results['mask_camera'])

        return packed


def get_train_transforms(
    input_size: Tuple[int, int] = (432, 768),
    resize_lim: Tuple[float, float] = (0.48, 0.48),
    depth_root: str = 'data/nuscenes_unidepth',
    feats_root: str = 'data/nuscenes_featup',
    sem_seg_root: Optional[str] = 'data/nuscenes_sam3',
    data_root: str = 'data/nuscenes'
) -> Compose:
    """Get training transforms.

    Args:
        input_size: Target image size.
        resize_lim: Resize limits.
        depth_root: Root for depth features.
        feats_root: Root for image features.
        sem_seg_root: Root for semantic segmentation.
        data_root: Root for nuScenes data.

    Returns:
        Composed transforms.
    """
    transforms = [
        LoadMultiViewImages(to_float32=True, num_views=6, data_root=data_root),
        ImageAug3D(
            final_dim=input_size,
            resize_lim=resize_lim,
            is_train=True
        ),
        LoadFeatMaps(data_root=depth_root, key='depth', apply_aug=True),
        LoadFeatMaps(data_root=feats_root, key='feats'),
    ]

    if sem_seg_root:
        transforms.append(
            LoadFeatMaps(data_root=sem_seg_root, key='sem_seg', apply_aug=True)
        )

    transforms.append(PackInputs())

    return Compose(transforms)


def get_val_transforms(
    input_size: Tuple[int, int] = (432, 768),
    resize_lim: Tuple[float, float] = (0.48, 0.48),
    depth_root: str = 'data/nuscenes_unidepth',
    feats_root: str = 'data/nuscenes_featup',
    data_root: str = 'data/nuscenes'
) -> Compose:
    """Get validation transforms.

    Args:
        input_size: Target image size.
        resize_lim: Resize limits.
        depth_root: Root for depth features.
        feats_root: Root for image features.
        data_root: Root for nuScenes data.

    Returns:
        Composed transforms.
    """
    transforms = [
        LoadMultiViewImages(to_float32=True, num_views=6, data_root=data_root),
        LoadOccFromFile(),
        ImageAug3D(
            final_dim=input_size,
            resize_lim=resize_lim,
            is_train=False
        ),
        LoadFeatMaps(data_root=depth_root, key='depth', apply_aug=True),
        LoadFeatMaps(data_root=feats_root, key='feats'),
        PackInputs(),
    ]

    return Compose(transforms)
