"""PyTorch Lightning DataModule for GaussTR.

Handles data loading, transforms, and dataloader creation.
"""

from typing import Optional, Dict, Any, Tuple, List

import pytorch_lightning as pl
from torch.utils.data import DataLoader

from .dataset import NuScenesOccDataset, NuScenesOccDatasetV2
from .t4_dataset import T4Dataset
from .transforms import (
    get_train_transforms, get_val_transforms,
    get_pgocc_train_transforms, get_pgocc_val_transforms,
    Compose,
)
from .collate import collate_gausstr, collate_pgocc


class GaussTRDataModule(pl.LightningDataModule):
    """Lightning DataModule for GaussTR training and evaluation.

    Args:
        data_root: Root directory for nuScenes data.
        train_ann_file: Path to training annotation file.
        val_ann_file: Path to validation annotation file.
        input_size: Target image size (H, W).
        resize_lim: Resize factor limits (min, max).
        depth_root: Root directory for depth features.
        feats_root: Root directory for image features.
        sem_seg_root: Root directory for semantic segmentation (optional).
        dataset_type: Dataset identifier ("nuscenes" or "t4").
        camera_names: Optional ordered list of camera names to load.
        num_views: Number of camera views.
        use_camera_subdirs: Whether depth/feature maps are stored per camera.
        use_chunk_subdirs: Whether to use {chunk_name}/{camera} subdirs (T4 format).
        sam3_png_format: Whether SAM3 segmentation masks are PNG files.
        has_gt: Whether occupancy ground truth is available.
        batch_size: Batch size per GPU.
        num_workers: Number of data loading workers.
        pin_memory: Whether to pin memory.
        persistent_workers: Whether to keep workers alive between epochs.
    """

    def __init__(
        self,
        data_root: str = 'data/nuscenes',
        train_ann_file: str = 'data/nuscenes/nuscenes_infos_train.pkl',
        val_ann_file: str = 'data/nuscenes/nuscenes_infos_val.pkl',
        input_size: Tuple[int, int] = (432, 768),
        resize_lim: Tuple[float, float] = (0.48, 0.48),
        depth_root: str = 'data/nuscenes_metric3d',
        feats_root: str = 'data/nuscenes_featup',
        sem_seg_root: Optional[str] = None,
        dataset_type: str = 'nuscenes',
        camera_names: Optional[List[str]] = None,
        num_views: int = 6,
        use_camera_subdirs: bool = False,
        use_chunk_subdirs: bool = False,
        sam3_png_format: bool = False,
        depth_png_format: bool = False,
        depth_scale: Optional[float] = None,
        fixed_resize_min_side: Optional[int] = None,
        fixed_resize_round: int = 16,
        has_gt: Optional[bool] = None,
        batch_size: int = 2,
        num_workers: int = 4,
        pin_memory: bool = True,
        persistent_workers: bool = True,
        prefetch_factor: int = 3,
        # PG-Occ specific
        model_type: str = 'gausstr',
        render_h: int = 180,
        render_w: int = 320,
        num_sweeps: int = 7,
        warp_sweep_indices: list = None,
        sam3_root: str = '',
        lidar_depth_root: str = '',
        flow_root: str = '',
    ):
        super().__init__()
        self.save_hyperparameters()

        self.data_root = data_root
        self.train_ann_file = train_ann_file
        self.val_ann_file = val_ann_file
        self.input_size = input_size
        self.resize_lim = resize_lim
        self.depth_root = depth_root
        self.feats_root = feats_root
        self.sem_seg_root = sem_seg_root
        self.dataset_type = (dataset_type or 'nuscenes').lower()
        self.camera_names = camera_names
        self.num_views = num_views
        self.use_camera_subdirs = use_camera_subdirs
        self.use_chunk_subdirs = use_chunk_subdirs
        self.sam3_png_format = sam3_png_format
        self.depth_png_format = depth_png_format
        self.depth_scale = depth_scale
        self.fixed_resize_min_side = fixed_resize_min_side
        self.fixed_resize_round = fixed_resize_round
        self.has_gt = has_gt if has_gt is not None else self.dataset_type != 't4'
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.persistent_workers = persistent_workers and num_workers > 0
        self.prefetch_factor = prefetch_factor if num_workers > 0 else None
        self.model_type = (model_type or 'gausstr').lower()
        self.render_h = render_h
        self.render_w = render_w
        self.num_sweeps = num_sweeps
        self.warp_sweep_indices = warp_sweep_indices
        self.sam3_root = sam3_root
        self.lidar_depth_root = lidar_depth_root
        self.flow_root = flow_root

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def _get_collate_fn(self):
        """Return the appropriate collate function."""
        if self.model_type == 'pgocc':
            return collate_pgocc
        return collate_gausstr

    def setup(self, stage: Optional[str] = None):
        """Set up datasets for each stage.

        Args:
            stage: Current stage ('fit', 'validate', 'test', or None for all).
        """
        if self.model_type == 'pgocc':
            self._setup_pgocc(stage)
            return

        if stage == 'fit' or stage is None:
            # Training transforms with augmentation
            train_transforms = get_train_transforms(
                input_size=self.input_size,
                resize_lim=self.resize_lim,
                depth_root=self.depth_root,
                feats_root=self.feats_root,
                sem_seg_root=self.sem_seg_root,
                data_root=self.data_root,
                num_views=self.num_views,
                use_camera_subdirs=self.use_camera_subdirs,
                use_chunk_subdirs=self.use_chunk_subdirs,
                sam3_png_format=self.sam3_png_format,
                depth_png_format=self.depth_png_format,
                depth_scale=self.depth_scale,
                fixed_resize_min_side=self.fixed_resize_min_side,
                fixed_resize_round=self.fixed_resize_round,
            )

            self.train_dataset = self._build_dataset(
                ann_file=self.train_ann_file,
                transforms=train_transforms,
                test_mode=False,
            )

        if (stage in ('fit', 'validate') or stage is None) and self.has_gt:
            # Validation transforms (deterministic, no random augmentation)
            # Only create validation dataset if GT is available
            val_transforms = get_val_transforms(
                input_size=self.input_size,
                resize_lim=self.resize_lim,
                depth_root=self.depth_root,
                feats_root=self.feats_root,
                sem_seg_root=self.sem_seg_root,
                data_root=self.data_root,
                num_views=self.num_views,
                use_camera_subdirs=self.use_camera_subdirs,
                use_chunk_subdirs=self.use_chunk_subdirs,
                sam3_png_format=self.sam3_png_format,
                depth_png_format=self.depth_png_format,
                depth_scale=self.depth_scale,
                fixed_resize_min_side=self.fixed_resize_min_side,
                fixed_resize_round=self.fixed_resize_round,
                load_gt=self.has_gt,
            )

            self.val_dataset = self._build_dataset(
                ann_file=self.val_ann_file,
                transforms=val_transforms,
                test_mode=False,
            )

        if stage == 'test':
            # Test uses same transforms as validation
            val_transforms = get_val_transforms(
                input_size=self.input_size,
                resize_lim=self.resize_lim,
                depth_root=self.depth_root,
                feats_root=self.feats_root,
                sem_seg_root=self.sem_seg_root,
                data_root=self.data_root,
                num_views=self.num_views,
                use_camera_subdirs=self.use_camera_subdirs,
                use_chunk_subdirs=self.use_chunk_subdirs,
                sam3_png_format=self.sam3_png_format,
                depth_png_format=self.depth_png_format,
                depth_scale=self.depth_scale,
                fixed_resize_min_side=self.fixed_resize_min_side,
                fixed_resize_round=self.fixed_resize_round,
                load_gt=self.has_gt,
            )

            self.test_dataset = self._build_dataset(
                ann_file=self.val_ann_file,
                transforms=val_transforms,
                test_mode=True,
            )

    def _setup_pgocc(self, stage):
        """Set up PG-Occ specific datasets with temporal sweep support."""
        pgocc_kwargs = dict(
            data_root=self.data_root,
            depth_root=self.depth_root,
            feats_root=self.feats_root,
            sam3_root=self.sam3_root,
            lidar_depth_root=self.lidar_depth_root,
            flow_root=self.flow_root,
            input_size=self.input_size,
            render_h=self.render_h,
            render_w=self.render_w,
            num_sweeps=self.num_sweeps,
            num_cams=self.num_views,
            num_views=self.num_views,
            warp_sweep_indices=self.warp_sweep_indices,
        )

        if stage == 'fit' or stage is None:
            train_transforms = get_pgocc_train_transforms(**pgocc_kwargs)
            self.train_dataset = self._build_dataset(
                ann_file=self.train_ann_file,
                transforms=train_transforms,
                test_mode=False,
            )

        if (stage in ('fit', 'validate') or stage is None) and self.has_gt:
            val_transforms = get_pgocc_val_transforms(**pgocc_kwargs)
            self.val_dataset = self._build_dataset(
                ann_file=self.val_ann_file,
                transforms=val_transforms,
                test_mode=False,
            )

        if stage == 'test':
            val_transforms = get_pgocc_val_transforms(**pgocc_kwargs)
            self.test_dataset = self._build_dataset(
                ann_file=self.val_ann_file,
                transforms=val_transforms,
                test_mode=True,
            )

    def _build_dataset(
        self,
        ann_file: str,
        transforms: Optional[Compose],
        test_mode: bool,
    ):
        if self.dataset_type == 't4':
            return T4Dataset(
                ann_file=ann_file,
                data_root=self.data_root,
                transforms=transforms,
                test_mode=test_mode,
                camera_names=self.camera_names,
            )

        return NuScenesOccDatasetV2(
            ann_file=ann_file,
            data_root=self.data_root,
            transforms=transforms,
            test_mode=test_mode,
            camera_names=self.camera_names,
        )

    def train_dataloader(self) -> DataLoader:
        """Create training dataloader."""
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=self._get_collate_fn(),
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
            prefetch_factor=self.prefetch_factor,
            drop_last=True,
        )

    def val_dataloader(self) -> Optional[DataLoader]:
        """Create validation dataloader.

        Returns train dataloader if has_gt is False (validation will be skipped
        via limit_val_batches=0 in Trainer).
        """
        # Use train dataset as fallback when no validation data
        dataset = self.val_dataset if self.has_gt else self.train_dataset
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=self._get_collate_fn(),
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
            prefetch_factor=self.prefetch_factor,
            drop_last=False,
        )

    def test_dataloader(self) -> DataLoader:
        """Create test dataloader."""
        dataset = self.test_dataset or self.val_dataset
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=self._get_collate_fn(),
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
            prefetch_factor=self.prefetch_factor,
            drop_last=False,
        )

    def predict_dataloader(self) -> DataLoader:
        """Create prediction dataloader (same as test)."""
        return self.test_dataloader()

    @property
    def num_train_samples(self) -> int:
        """Get number of training samples."""
        if self.train_dataset is None:
            return 0
        return len(self.train_dataset)

    @property
    def num_val_samples(self) -> int:
        """Get number of validation samples."""
        if self.val_dataset is None:
            return 0
        return len(self.val_dataset)


class GaussTRDataModuleFromConfig(GaussTRDataModule):
    """DataModule initialized from GaussTRConfig dataclass."""

    def __init__(self, config: "GaussTRConfig"):
        """Initialize from config dataclass.

        Args:
            config: GaussTRConfig with data configuration.
        """
        data_cfg = config.data
        super().__init__(
            data_root=data_cfg.data_root,
            train_ann_file=data_cfg.train_ann_file,
            val_ann_file=data_cfg.val_ann_file,
            input_size=data_cfg.input_size,
            resize_lim=data_cfg.resize_lim,
            depth_root=data_cfg.depth_root,
            feats_root=data_cfg.feats_root,
            sem_seg_root=data_cfg.sem_seg_root,
            dataset_type=getattr(data_cfg, 'dataset_type', 'nuscenes'),
            camera_names=getattr(data_cfg, 'camera_names', None),
            num_views=getattr(data_cfg, 'num_views', 6),
            use_camera_subdirs=getattr(data_cfg, 'use_camera_subdirs', False),
            use_chunk_subdirs=getattr(data_cfg, 'use_chunk_subdirs', False),
            sam3_png_format=getattr(data_cfg, 'sam3_png_format', False),
            depth_png_format=getattr(data_cfg, 'depth_png_format', False),
            depth_scale=getattr(data_cfg, 'depth_scale', None),
            fixed_resize_min_side=getattr(data_cfg, 'fixed_resize_min_side', None),
            fixed_resize_round=getattr(data_cfg, 'fixed_resize_round', 16),
            has_gt=getattr(data_cfg, 'has_gt', None),
            batch_size=data_cfg.batch_size,
            num_workers=data_cfg.num_workers,
            pin_memory=getattr(data_cfg, 'pin_memory', True),
            persistent_workers=getattr(data_cfg, 'persistent_workers', True),
            prefetch_factor=getattr(data_cfg, 'prefetch_factor', 3),
        )
