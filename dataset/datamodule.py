"""PyTorch Lightning DataModule for GaussTR.

Handles data loading, transforms, and dataloader creation.
"""

from typing import Optional, Dict, Any, Tuple

import pytorch_lightning as pl
from torch.utils.data import DataLoader

from .dataset import NuScenesOccDataset, NuScenesOccDatasetV2
from .transforms import get_train_transforms, get_val_transforms, Compose
from .collate import collate_gausstr


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
        batch_size: int = 2,
        num_workers: int = 4,
        pin_memory: bool = True,
        persistent_workers: bool = True,
        prefetch_factor: int = 3,
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
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.persistent_workers = persistent_workers and num_workers > 0
        self.prefetch_factor = prefetch_factor if num_workers > 0 else None

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def setup(self, stage: Optional[str] = None):
        """Set up datasets for each stage.

        Args:
            stage: Current stage ('fit', 'validate', 'test', or None for all).
        """
        if stage == 'fit' or stage is None:
            # Training transforms with augmentation
            train_transforms = get_train_transforms(
                input_size=self.input_size,
                resize_lim=self.resize_lim,
                depth_root=self.depth_root,
                feats_root=self.feats_root,
                sem_seg_root=self.sem_seg_root,
                data_root=self.data_root,
            )

            self.train_dataset = NuScenesOccDatasetV2(
                ann_file=self.train_ann_file,
                data_root=self.data_root,
                transforms=train_transforms,
                test_mode=False,
            )

        if stage in ('fit', 'validate') or stage is None:
            # Validation transforms (deterministic, no random augmentation)
            val_transforms = get_val_transforms(
                input_size=self.input_size,
                resize_lim=self.resize_lim,
                depth_root=self.depth_root,
                feats_root=self.feats_root,
                data_root=self.data_root,
            )

            self.val_dataset = NuScenesOccDatasetV2(
                ann_file=self.val_ann_file,
                data_root=self.data_root,
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
                data_root=self.data_root,
            )

            self.test_dataset = NuScenesOccDatasetV2(
                ann_file=self.val_ann_file,
                data_root=self.data_root,
                transforms=val_transforms,
                test_mode=True,
            )

    def train_dataloader(self) -> DataLoader:
        """Create training dataloader."""
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=collate_gausstr,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
            prefetch_factor=self.prefetch_factor,
            drop_last=True,
        )

    def val_dataloader(self) -> DataLoader:
        """Create validation dataloader."""
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=collate_gausstr,
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
            collate_fn=collate_gausstr,
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
            batch_size=data_cfg.batch_size,
            num_workers=data_cfg.num_workers,
            pin_memory=getattr(data_cfg, 'pin_memory', True),
            persistent_workers=getattr(data_cfg, 'persistent_workers', True),
            prefetch_factor=getattr(data_cfg, 'prefetch_factor', 3),
        )
