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
        to_rgb: Convert OpenCV BGR images to RGB.
    """

    def __init__(
        self,
        to_float32: bool = True,
        num_views: int = 6,
        data_root: str = 'data/nuscenes',
        to_rgb: bool = True,
    ):
        self.to_float32 = to_float32
        self.num_views = num_views
        self.data_root = data_root
        self.to_rgb = to_rgb

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
            if self.to_rgb:
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
        fixed_resize_min_side: If set, resize so the shorter side equals this value,
            then round H/W up to a multiple of fixed_resize_round. Disables crop/flip/rotate.
        fixed_resize_round: Round resized H/W up to this multiple when fixed_resize_min_side is set.
    """

    def __init__(
        self,
        final_dim: Tuple[int, int],
        resize_lim: Tuple[float, float],
        bot_pct_lim: Tuple[float, float] = (0.0, 0.0),
        rot_lim: Tuple[float, float] = (0.0, 0.0),
        rand_flip: bool = False,
        is_train: bool = False,
        fixed_resize_min_side: Optional[int] = None,
        fixed_resize_round: int = 16
    ):
        self.final_dim = final_dim
        self.resize_lim = resize_lim
        self.bot_pct_lim = bot_pct_lim
        self.rand_flip = rand_flip
        self.rot_lim = rot_lim
        self.is_train = is_train
        self.fixed_resize_min_side = fixed_resize_min_side
        self.fixed_resize_round = fixed_resize_round

    def sample_augmentation(self, results: Dict) -> Tuple:
        """Sample augmentation parameters."""
        H, W = results['ori_shape']
        fH, fW = self.final_dim

        if self.fixed_resize_min_side is not None:
            # Match dinov3clip keep_aspect_min_side_round16: resize only, no crop/flip/rotate
            resize = float(self.fixed_resize_min_side) / float(min(H, W))
            newW = int(round(W * resize))
            newH = int(round(H * resize))
            round_to = max(1, int(self.fixed_resize_round))
            newH = ((newH + round_to - 1) // round_to) * round_to
            newW = ((newW + round_to - 1) // round_to) * round_to
            resize_dims = (newW, newH)
            # Use anisotropic scales to reflect rounded dimensions exactly
            resize = (newW / float(W), newH / float(H))
            crop = (0, 0, newW, newH)
            flip = False
            rotate = 0.0
        elif self.is_train:
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
        resize: Any,
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

        # Update transformation matrix (support anisotropic resize)
        if isinstance(resize, (tuple, list, np.ndarray)):
            scale_x, scale_y = float(resize[0]), float(resize[1])
            scale = torch.tensor([[scale_x, 0.0], [0.0, scale_y]], dtype=torch.float32)
            rotation = rotation @ scale
        else:
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


def _extract_chunk_name(path: str) -> str:
    """Extract chunk name from T4 image path.

    Path format: .../t4_datasets/{chunk_name}/data/{camera}/{filename}.jpg
    """
    parts = path.replace("\\", "/").split("/")
    for i, part in enumerate(parts):
        if part == "t4_datasets" and i + 1 < len(parts):
            return parts[i + 1]
    return ""


def _load_sparse_depth(path: str) -> np.ndarray:
    """Load sparse depth from .npz file and reconstruct dense array."""
    data = np.load(path)
    indices = data['indices']  # [N, 2] (row, col)
    values = data['values']    # [N] depth values
    shape = tuple(data['shape'])  # (H, W)

    # Reconstruct dense depth map
    depth = np.zeros(shape, dtype=np.float32)
    if len(indices) > 0:
        depth[indices[:, 0], indices[:, 1]] = values.astype(np.float32)
    return depth


def _load_png_as_array(path: str, depth_scale: float = None) -> np.ndarray:
    """Load PNG image as numpy array.

    Args:
        path: Path to PNG file.
        depth_scale: If provided and image is uint16, convert to float32 depth in meters
                     by dividing by this scale (e.g., 650 for 1.54mm precision, max 100m).

    Returns:
        numpy array - uint8/int64 for segmentation masks, float32 for depth maps.
    """
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"Failed to load PNG: {path}")

    # Handle 16-bit depth PNGs (e.g., from Prior-Depth-Anything)
    if img.dtype == np.uint16 and depth_scale is not None:
        # ~3% of PriorDA frames were encoded with scale=1000 instead of 650,
        # detectable by saturated max value (65535 = 65.535m at scale=1000,
        # vs 100.8m at scale=650 which rarely saturates for driving scenes).
        if img.max() == 65535:
            return img.astype(np.float32) / 1000.0
        return img.astype(np.float32) / depth_scale

    return img


class LoadFeatMaps:
    """Load pre-extracted feature maps.

    Args:
        data_root: Root directory containing feature maps.
        key: Key name for storing features in results.
        apply_aug: Whether to apply image augmentation to features.
        suffix: Optional suffix for feature filenames.
        use_mmap: Use memory-mapped loading for large datasets (reduces RAM usage).
        use_camera_subdirs: If True, load from per-camera subdirectories.
        use_chunk_subdirs: If True, load from {chunk_name}/{camera} subdirectories (T4 format).
        png_format: If True, load PNG files instead of numpy (for SAM3 segmentation masks).
        depth_scale: Scale factor for 16-bit depth PNGs (e.g., 650 for Prior-Depth-Anything).
                     If provided, uint16 PNGs are converted to float32 meters.

    Note:
        File format is auto-detected based on extension (.npy, .npz, or .png).
    """

    def __init__(
        self,
        data_root: str,
        key: str,
        apply_aug: bool = False,
        suffix: str = '',
        use_mmap: bool = False,
        use_camera_subdirs: bool = False,
        use_chunk_subdirs: bool = False,
        png_format: bool = False,
        depth_scale: float = None
    ):
        self.data_root = data_root
        self.key = key
        self.apply_aug = apply_aug
        self.suffix = suffix
        self.use_mmap = use_mmap
        self.use_camera_subdirs = use_camera_subdirs
        self.use_chunk_subdirs = use_chunk_subdirs
        self.png_format = png_format
        self.depth_scale = depth_scale

    def __call__(self, results: Dict) -> Dict:
        """Load feature maps for all views."""
        feats = []
        img_aug_mats = results.get('img_aug_mat')

        for i, filename in enumerate(results['filename']):
            # Build feature path
            basename = os.path.basename(filename).split('.')[0]
            cam_name = os.path.basename(os.path.dirname(filename))

            # Build base path without extension
            if self.use_chunk_subdirs:
                # T4 format: {data_root}/{chunk_name}/{cam_name}/{basename}
                chunk_name = _extract_chunk_name(filename)
                base_path = os.path.join(
                    self.data_root,
                    chunk_name,
                    cam_name,
                    basename + self.suffix
                )
            elif self.use_camera_subdirs:
                base_path = os.path.join(
                    self.data_root,
                    cam_name,
                    basename + self.suffix
                )
            else:
                base_path = os.path.join(self.data_root, basename + self.suffix)

            # Auto-detect file format based on extension
            if self.png_format:
                feat_path = base_path + '.png'
                feat = _load_png_as_array(feat_path, depth_scale=self.depth_scale)
            elif os.path.exists(base_path + '.npy'):
                feat_path = base_path + '.npy'
                if self.use_mmap:
                    feat = np.load(feat_path, mmap_mode='r')
                    feat = np.array(feat)  # Copy to allow modification
                else:
                    feat = np.load(feat_path)
            elif os.path.exists(base_path + '.npz'):
                feat_path = base_path + '.npz'
                feat = _load_sparse_depth(feat_path)
            else:
                raise FileNotFoundError(
                    f"Feature file not found: {base_path}.[npy|npz]"
                )

            # Handle int8 quantized features (convert back to float32)
            # Skip for PNG segmentation masks which are uint8 class labels
            if feat.dtype == np.int8:
                feat = feat.astype(np.float32) / 127.0
            elif self.png_format and feat.dtype != np.float32:
                # PNG segmentation masks - keep as integer class labels
                # (depth PNGs with depth_scale are already float32)
                feat = feat.astype(np.int64)

            feat = torch.from_numpy(feat)

            # Apply augmentation if needed
            if self.apply_aug and img_aug_mats is not None:
                post_rot = img_aug_mats[i][:3, :3]
                post_tran = img_aug_mats[i][:3, 3]

                h, w = feat.shape[-2:]

                # Resize
                new_h = int(h * post_rot[1, 1] + 0.5)
                new_w = int(w * post_rot[0, 0] + 0.5)

                # Use nearest neighbor for segmentation masks; depth should stay continuous
                int_types = (torch.long, torch.int, torch.int64, torch.int32)
                is_segmentation = (self.png_format and self.key in ("sem_seg", "sem_segs")) or feat.dtype in int_types
                if is_segmentation:
                    mode = 'nearest'
                else:
                    # Prefer area for downsampling (anti-aliasing), bilinear for upsampling
                    mode = 'area' if (new_h < h or new_w < w) else 'bilinear'

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
            'sample_idx', 'token', 'scene_idx', 'timestamp', 'num_views', 'img_path', 'depth', 'feats', 'sem_seg'
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
    data_root: str = 'data/nuscenes',
    num_views: int = 6,
    use_camera_subdirs: bool = False,
    use_chunk_subdirs: bool = False,
    sam3_png_format: bool = False,
    depth_png_format: bool = False,
    depth_scale: Optional[float] = None,
    fixed_resize_min_side: Optional[int] = None,
    fixed_resize_round: int = 16,
) -> Compose:
    """Get training transforms.

    Args:
        input_size: Target image size.
        resize_lim: Resize limits.
        depth_root: Root for depth features.
        feats_root: Root for image features.
        sem_seg_root: Root for semantic segmentation.
        data_root: Root for nuScenes data.
        num_views: Number of camera views.
        use_camera_subdirs: Whether to load depth/features from per-camera subdirs.
        use_chunk_subdirs: Whether to use {chunk_name}/{camera} subdirs (T4 format).
        sam3_png_format: Whether SAM3 segmentation masks are PNG files.
        depth_png_format: Whether depth maps are stored as 16-bit PNG files.
        depth_scale: Scale factor for 16-bit depth PNGs (e.g., 650 for Prior-Depth-Anything).
        fixed_resize_min_side: If set, resize so the shorter side equals this value and round
            H/W to a multiple of fixed_resize_round. Disables crop/flip/rotate.
        fixed_resize_round: Round resized H/W up to this multiple when fixed_resize_min_side is set.

    Returns:
        Composed transforms.
    """
    transforms = [
        LoadMultiViewImages(to_float32=True, num_views=num_views, data_root=data_root),
        ImageAug3D(
            final_dim=input_size,
            resize_lim=resize_lim,
            is_train=True,
            fixed_resize_min_side=fixed_resize_min_side,
            fixed_resize_round=fixed_resize_round,
        ),
        LoadFeatMaps(
            data_root=depth_root,
            key='depth',
            apply_aug=True,
            use_camera_subdirs=use_camera_subdirs,
            use_chunk_subdirs=use_chunk_subdirs,
            png_format=depth_png_format,
            depth_scale=depth_scale,
        ),
        LoadFeatMaps(
            data_root=feats_root,
            key='feats',
            use_camera_subdirs=use_camera_subdirs,
            use_chunk_subdirs=use_chunk_subdirs,
        ),
    ]

    if sem_seg_root:
        transforms.append(
            LoadFeatMaps(
                data_root=sem_seg_root,
                key='sem_seg',
                apply_aug=True,
                use_camera_subdirs=use_camera_subdirs,
                use_chunk_subdirs=use_chunk_subdirs,
                png_format=sam3_png_format,
            )
        )

    transforms.append(PackInputs())

    return Compose(transforms)


def get_val_transforms(
    input_size: Tuple[int, int] = (432, 768),
    resize_lim: Tuple[float, float] = (0.48, 0.48),
    depth_root: str = 'data/nuscenes_unidepth',
    feats_root: str = 'data/nuscenes_featup',
    sem_seg_root: Optional[str] = None,
    data_root: str = 'data/nuscenes',
    num_views: int = 6,
    use_camera_subdirs: bool = False,
    use_chunk_subdirs: bool = False,
    sam3_png_format: bool = False,
    depth_png_format: bool = False,
    depth_scale: Optional[float] = None,
    fixed_resize_min_side: Optional[int] = None,
    fixed_resize_round: int = 16,
    load_gt: bool = True
) -> Compose:
    """Get validation transforms.

    Args:
        input_size: Target image size.
        resize_lim: Resize limits.
        depth_root: Root for depth features.
        feats_root: Root for image features.
        sem_seg_root: Root for semantic segmentation (optional).
        data_root: Root for nuScenes data.
        num_views: Number of camera views.
        use_camera_subdirs: Whether to load depth/features from per-camera subdirs.
        use_chunk_subdirs: Whether to use {chunk_name}/{camera} subdirs (T4 format).
        sam3_png_format: Whether SAM3 segmentation masks are PNG files.
        depth_png_format: Whether depth maps are stored as 16-bit PNG files.
        depth_scale: Scale factor for 16-bit depth PNGs (e.g., 650 for Prior-Depth-Anything).
        fixed_resize_min_side: If set, resize so the shorter side equals this value and round
            H/W to a multiple of fixed_resize_round. Disables crop/flip/rotate.
        fixed_resize_round: Round resized H/W up to this multiple when fixed_resize_min_side is set.
        load_gt: Whether to load occupancy ground truth.

    Returns:
        Composed transforms.
    """
    transforms = [
        LoadMultiViewImages(to_float32=True, num_views=num_views, data_root=data_root),
    ]

    if load_gt:
        transforms.append(LoadOccFromFile())

    transforms.extend([
        ImageAug3D(
            final_dim=input_size,
            resize_lim=resize_lim,
            is_train=False,
            fixed_resize_min_side=fixed_resize_min_side,
            fixed_resize_round=fixed_resize_round,
        ),
        LoadFeatMaps(
            data_root=depth_root,
            key='depth',
            apply_aug=True,
            use_camera_subdirs=use_camera_subdirs,
            use_chunk_subdirs=use_chunk_subdirs,
            png_format=depth_png_format,
            depth_scale=depth_scale,
        ),
        LoadFeatMaps(
            data_root=feats_root,
            key='feats',
            use_camera_subdirs=use_camera_subdirs,
            use_chunk_subdirs=use_chunk_subdirs,
        ),
    ])

    if sem_seg_root:
        transforms.append(
            LoadFeatMaps(
                data_root=sem_seg_root,
                key='sem_seg',
                apply_aug=True,
                use_camera_subdirs=use_camera_subdirs,
                use_chunk_subdirs=use_chunk_subdirs,
                png_format=sam3_png_format,
            )
        )

    transforms.append(PackInputs())

    return Compose(transforms)


# === PG-Occ transforms ===


class LoadMultiSweepImages:
    """Load temporal sweep images and compute ego motion transforms.

    Loads images from past frames (stored in annotation 'sweeps' field),
    computes ego-to-image projections for all frames, and ego-to-ego
    transforms for temporal depth warping.

    Args:
        num_sweeps: Number of past sweep frames to load.
        num_cams: Number of cameras.
        render_h: Render target height.
        render_w: Render target width.
        data_root: Root directory for T4 data.
        to_rgb: Convert OpenCV BGR images to RGB.
    """

    def __init__(
        self,
        num_sweeps=7,
        num_cams=5,
        render_h=180,
        render_w=320,
        data_root='',
        to_rgb=False,
    ):
        self.num_sweeps = num_sweeps
        self.num_cams = num_cams
        self.render_h = render_h
        self.render_w = render_w
        self.data_root = data_root
        self.to_rgb = to_rgb

    def __call__(self, results):
        N = self.num_cams
        cam_names = list(results['images'].keys())[:N]
        sweeps = results.get('sweeps', [])

        # Current frame ego2global and cam2ego per camera
        cur_ego2global = []
        cur_cam2ego = []
        cur_cam2img = []
        for cam_name in cam_names:
            cam = results['images'][cam_name]
            cur_ego2global.append(np.array(cam['ego2global'], dtype=np.float32))
            cur_cam2ego.append(np.array(cam['cam2ego'], dtype=np.float32))
            c2i_3x3 = np.array(cam['cam2img'], dtype=np.float32)
            c2i = np.eye(4, dtype=np.float32)
            c2i[:3, :3] = c2i_3x3 if c2i_3x3.shape == (3, 3) else c2i_3x3[:3, :3]
            cur_cam2img.append(c2i)

        cur_ego2global = np.stack(cur_ego2global)  # [N, 4, 4]
        cur_cam2ego = np.stack(cur_cam2ego)  # [N, 4, 4]
        cur_cam2img = np.stack(cur_cam2img)  # [N, 4, 4]

        # Compute ego2img for current frame: cam2img @ inv(cam2ego)
        ego2img_list = []
        for n in range(N):
            ego2cam = np.linalg.inv(cur_cam2ego[n])
            ego2img_list.append(cur_cam2img[n] @ ego2cam)

        # Store render_k scaled to render resolution for gsplat rendering.
        img_h, img_w = results['img'][0].shape[:2]
        render_k = cur_cam2img.copy()
        render_k[..., 0, :] *= float(self.render_w) / float(img_w)
        render_k[..., 1, :] *= float(self.render_h) / float(img_h)
        results['render_k'] = render_k

        # Match PG-Occ sweep selection behavior:
        # if fewer sweeps are available, repeat the last available sweep.
        if len(sweeps) == 0:
            sweep_indices = [None] * self.num_sweeps
        elif len(sweeps) <= self.num_sweeps:
            sweep_indices = list(range(len(sweeps))) + [len(sweeps) - 1] * (self.num_sweeps - len(sweeps))
        else:
            sweep_indices = list(range(self.num_sweeps))

        # Load sweep images and compute transforms.
        sweep_imgs = []
        t0_2_x_geo = []

        for sweep_idx in sweep_indices:
            sweep = sweeps[sweep_idx] if sweep_idx is not None else None
            for n, cam_name in enumerate(cam_names):
                if sweep is not None and cam_name in sweep.get('images', {}):
                    s_cam = sweep['images'][cam_name]
                else:
                    # Fallback to current frame for missing/empty sweep entries.
                    s_cam = results['images'][cam_name]

                # Load sweep image
                img_path = s_cam['img_path']
                if not os.path.isabs(img_path) and self.data_root:
                    img_path = os.path.join(self.data_root, img_path)
                img = cv2.imread(img_path)
                if img is not None:
                    if self.to_rgb:
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    img = img.astype(np.float32)
                    sweep_imgs.append(img)
                else:
                    h, w = results['img'][0].shape[:2]
                    sweep_imgs.append(np.zeros((h, w, 3), dtype=np.float32))

                # ego2img for sweep frame
                s_c2i_3x3 = np.array(s_cam['cam2img'], dtype=np.float32)
                s_c2i = np.eye(4, dtype=np.float32)
                s_c2i[:3, :3] = s_c2i_3x3 if s_c2i_3x3.shape == (3, 3) else s_c2i_3x3[:3, :3]
                s_cam2ego = np.array(s_cam['cam2ego'], dtype=np.float32)
                s_ego2global = np.array(s_cam['ego2global'], dtype=np.float32)

                # Map current-ego points to sweep image:
                # current ego -> global -> sweep ego -> sweep cam -> image.
                s_ego2img = (
                    s_c2i
                    @ np.linalg.inv(s_cam2ego)
                    @ np.linalg.inv(s_ego2global)
                    @ cur_ego2global[n]
                )
                ego2img_list.append(s_ego2img)

                # T = inv(s_cam2ego) @ inv(s_ego2global) @ cur_ego2global[n] @ cur_cam2ego[n]
                T = (np.linalg.inv(s_cam2ego) @
                     np.linalg.inv(s_ego2global) @
                     cur_ego2global[n] @
                     cur_cam2ego[n])
                t0_2_x_geo.append(T)

        # Generate render_gt: current + two temporal frames (PG-Occ loss contract).
        render_gt_imgs = []
        all_imgs = list(results['img'][:N]) + sweep_imgs[:2 * N]
        for img in all_imgs:
            img_u8 = img.astype(np.uint8) if img.max() > 1.0 else (img * 255).astype(np.uint8)
            resized = cv2.resize(img_u8, (self.render_w, self.render_h),
                                 interpolation=cv2.INTER_LINEAR)
            render_gt_imgs.append(resized.astype(np.float32))

        # Append sweep images to results (img now has T*N images)
        results['img'] = list(results['img'][:N]) + sweep_imgs[:self.num_sweeps * N]

        # Store computed arrays
        results['ego2img'] = np.stack(ego2img_list[:(1 + self.num_sweeps) * N])
        results['t0_2_x_geo'] = np.stack(t0_2_x_geo[:2 * N])
        results['render_gt'] = np.stack(render_gt_imgs)
        results['cam2ego'] = cur_cam2ego  # [N, 4, 4] current frame only

        return results


class ResizePGOccImages:
    """Resize images to target resolution for PG-Occ.

    Matches original PG-Occ deterministic IDA transform (training=False):
    isotropic resize + crop, without flip/rotate.

    Args:
        target_size: Target (H, W) for backbone input.
    """

    def __init__(self, target_size=(256, 704)):
        self.target_h, self.target_w = target_size

    def __call__(self, results):
        src_h, src_w = results['img'][0].shape[:2]

        # Same policy as RandomTransformImage(training=False):
        # resize = max(fH/H, fW/W), then center-crop width and bottom-crop height.
        resize = max(float(self.target_h) / float(src_h), float(self.target_w) / float(src_w))
        new_w = max(self.target_w, int(src_w * resize))
        new_h = max(self.target_h, int(src_h * resize))
        crop_h = int(new_h - self.target_h)
        crop_w = int(max(0, new_w - self.target_w) / 2)
        x1, y1 = crop_w, crop_h
        x2, y2 = crop_w + self.target_w, crop_h + self.target_h

        resized = []
        for img in results['img']:
            if img.shape[:2] != (self.target_h, self.target_w):
                img_u8 = img.astype(np.uint8) if img.max() > 1.0 else (img * 255).astype(np.uint8)
                img = cv2.resize(img_u8, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
                img = img[y1:y2, x1:x2].astype(np.float32)
            resized.append(img)
        results['img'] = resized

        # Keep projection matrices consistent with resized + cropped images.
        if 'ego2img' in results:
            ida_mat = np.eye(4, dtype=np.float32)
            ida_mat[0, 0] = resize
            ida_mat[1, 1] = resize
            ida_mat[0, 2] = -float(crop_w)
            ida_mat[1, 2] = -float(crop_h)
            ego2img = np.array(results['ego2img'], dtype=np.float32, copy=True)
            ego2img = np.matmul(ida_mat[None, ...], ego2img)
            results['ego2img'] = ego2img

        # Store crop info for computing blind region in render space.
        # The top crop_h/resize original rows have no backbone features.
        results['backbone_crop_top_orig'] = crop_h / resize
        results['orig_img_h'] = src_h
        results['img_shape'] = (self.target_h, self.target_w)
        return results


class PackPGOccInputs:
    """Pack PG-Occ inputs into tensor format for the model.

    Produces the batch dict expected by PGOccLightning:
    - img: [T*N, 3, H, W] images
    - depth: [N, 1, Hd, Wd] foundation depth
    - text_vision: [N, C, Hf, Wf] DINOv3CLIP features
    - render_gt: [3*N, Rh, Rw, 3] warping targets (current + 2 temporal)
    - t0_2_x_geo: [2*N, 4, 4] ego-to-ego transforms for temporal warping
    - img_metas: dict with ego2img, cam2ego, render_k
    """

    def __init__(self, num_cams=5, render_h=180, render_w=320):
        self.num_cams = num_cams
        self.render_h = render_h
        self.render_w = render_w

    def __call__(self, results):
        packed = {}

        # Images: [T*N, H, W, C] -> [T*N, C, H, W]
        imgs = np.stack(results['img'], axis=0)
        packed['img'] = torch.from_numpy(imgs).permute(0, 3, 1, 2)

        # Foundation depth: resize to render resolution for BackprojectDepth
        if 'depth' in results:
            depth = results['depth']
            if isinstance(depth, torch.Tensor):
                if depth.dim() == 3:
                    depth = depth.unsqueeze(1)
            elif isinstance(depth, np.ndarray):
                depth = torch.from_numpy(depth)
                if depth.dim() == 3:
                    depth = depth.unsqueeze(1)
            # Resize to render resolution: [N, 1, Hd, Wd] -> [N, 1, render_h, render_w]
            if depth.shape[-2] != self.render_h or depth.shape[-1] != self.render_w:
                depth = F.interpolate(
                    depth.float(),
                    size=(self.render_h, self.render_w),
                    mode='bilinear',
                    align_corners=False,
                )
            packed['depth'] = depth

        # DINOv3CLIP features
        if 'feats' in results:
            packed['text_vision'] = results['feats']

        # Render targets
        if 'render_gt' in results:
            packed['render_gt'] = torch.from_numpy(results['render_gt'])

        # Ego-to-ego transforms
        if 't0_2_x_geo' in results:
            packed['t0_2_x_geo'] = torch.from_numpy(results['t0_2_x_geo']).float()

        # Image metadata
        img_metas = {}
        for key in ['ego2img', 'cam2ego', 'render_k']:
            if key in results:
                val = results[key]
                if isinstance(val, np.ndarray):
                    img_metas[key] = torch.from_numpy(val).float()
                elif isinstance(val, torch.Tensor):
                    img_metas[key] = val.float()

        # Compute which render rows have backbone coverage.
        # Rows above this boundary have no backbone features (blind region).
        if 'backbone_crop_top_orig' in results and 'orig_img_h' in results:
            crop_top_orig = results['backbone_crop_top_orig']
            orig_h = results['orig_img_h']
            img_metas['backbone_valid_row'] = int(
                crop_top_orig / orig_h * self.render_h)

        # Shape info needed by pad_multiple
        num_cams = self.num_cams
        if 'img' in packed:
            _, C, H, W = packed['img'].shape
            img_metas['ori_shape'] = [(H, W, C)] * num_cams
            img_metas['img_shape'] = [(H, W, C)] * num_cams

        packed['img_metas'] = img_metas

        for key in ['token', 'scene_token', 'timestamp', 'sample_idx']:
            if key in results:
                packed[key] = results[key]

        return packed


def get_pgocc_train_transforms(
    data_root='/mnt/nvme2/T4_datasets',
    depth_root='/mnt/nvme1/data/T4_datasets_priorda_depth',
    feats_root='/mnt/nvme3/T4_datasets_dinov3clip',
    input_size=(256, 704),
    render_h=180,
    render_w=320,
    num_sweeps=7,
    num_cams=5,
    num_views=5,
):
    """Get PG-Occ training transforms."""
    return Compose([
        LoadMultiViewImages(to_float32=True, num_views=num_views, data_root=data_root, to_rgb=False),
        LoadMultiSweepImages(
            num_sweeps=num_sweeps, num_cams=num_cams,
            render_h=render_h, render_w=render_w, data_root=data_root, to_rgb=False),
        ResizePGOccImages(target_size=input_size),
        LoadFeatMaps(
            data_root=depth_root, key='depth', apply_aug=False,
            use_chunk_subdirs=True, png_format=True, depth_scale=650.0),
        LoadFeatMaps(
            data_root=feats_root, key='feats', apply_aug=False,
            use_chunk_subdirs=True),
        PackPGOccInputs(num_cams=num_cams, render_h=render_h, render_w=render_w),
    ])


def get_pgocc_val_transforms(
    data_root='/mnt/nvme2/T4_datasets',
    depth_root='/mnt/nvme1/data/T4_datasets_priorda_depth',
    feats_root='/mnt/nvme3/T4_datasets_dinov3clip',
    input_size=(256, 704),
    render_h=180,
    render_w=320,
    num_sweeps=7,
    num_cams=5,
    num_views=5,
):
    """Get PG-Occ validation transforms (same pipeline, no random aug)."""
    return Compose([
        LoadMultiViewImages(to_float32=True, num_views=num_views, data_root=data_root, to_rgb=False),
        LoadMultiSweepImages(
            num_sweeps=num_sweeps, num_cams=num_cams,
            render_h=render_h, render_w=render_w, data_root=data_root, to_rgb=False),
        ResizePGOccImages(target_size=input_size),
        LoadFeatMaps(
            data_root=depth_root, key='depth', apply_aug=False,
            use_chunk_subdirs=True, png_format=True, depth_scale=650.0),
        LoadFeatMaps(
            data_root=feats_root, key='feats', apply_aug=False,
            use_chunk_subdirs=True),
        PackPGOccInputs(num_cams=num_cams, render_h=render_h, render_w=render_w),
    ])
