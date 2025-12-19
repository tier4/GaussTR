"""PyTorch Lightning callbacks for automatic visualization.

Usage in training config:
    callbacks:
      - _target_: visualization.callbacks.VisualizationCallback
        num_samples: 50
        mode: composite
        output_format: image
"""

import os
from typing import Optional, List, Dict, Any
from pathlib import Path

import pytorch_lightning as pl
import torch
import numpy as np


class VisualizationCallback(pl.Callback):
    """Lightning callback to generate visualizations after test/validation.

    Args:
        num_samples: Number of samples to visualize (-1 for all).
        mode: Visualization mode ('bev', 'composite', 'full').
        output_format: Output format ('image' or 'video').
        fps: Video FPS if output_format='video'.
        save_predictions: Whether to save raw predictions.
        output_dir: Output directory (auto-generated if None).
    """

    def __init__(
        self,
        num_samples: int = -1,
        mode: str = "composite",
        output_format: str = "image",
        fps: int = 10,
        save_predictions: bool = False,
        output_dir: Optional[str] = None,
    ):
        super().__init__()
        self.num_samples = num_samples
        self.mode = mode
        self.output_format = output_format
        self.fps = fps
        self.save_predictions = save_predictions
        self.output_dir = output_dir

        # Storage for collecting predictions during test
        self._predictions: List[Dict[str, Any]] = []
        self._datamodule = None

    def on_test_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """Reset predictions at test start."""
        self._predictions = []
        self._datamodule = trainer.datamodule

        # Determine output directory
        if self.output_dir is None:
            # Use trainer log directory or work directory
            if trainer.log_dir:
                self.output_dir = os.path.join(trainer.log_dir, "visualizations")
            elif hasattr(trainer, 'default_root_dir'):
                self.output_dir = os.path.join(trainer.default_root_dir, "visualizations")
            else:
                self.output_dir = "visualizations"

        os.makedirs(self.output_dir, exist_ok=True)
        for subdir in ['bev', 'composite', '3d', 'predictions']:
            os.makedirs(os.path.join(self.output_dir, subdir), exist_ok=True)

        print(f"Visualization output: {self.output_dir}")

    def on_test_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: Dict[str, Any],
        batch: Dict[str, torch.Tensor],
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        """Collect predictions from test batches."""
        # num_samples == -1 means all samples
        if self.num_samples > 0 and batch_idx >= self.num_samples:
            return

        # Get prediction
        if 'preds' in outputs:
            pred_occ = outputs['preds']
            if isinstance(pred_occ, torch.Tensor):
                pred_occ = pred_occ.cpu().numpy()
        else:
            return

        result = {
            'idx': batch_idx,
            'pred_occ': pred_occ[0] if pred_occ.ndim == 4 else pred_occ,
        }

        # Add GT if available
        if 'gt_occ' in batch:
            result['gt_occ'] = batch['gt_occ'].cpu().numpy()[0]
        if 'mask_camera' in batch:
            result['mask_camera'] = batch['mask_camera'].cpu().numpy()[0]

        self._predictions.append(result)

    def on_test_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """Generate visualizations after test completes."""
        if not self._predictions:
            print("No predictions collected for visualization")
            return

        print(f"\nGenerating visualizations for {len(self._predictions)} samples...")

        self._generate_visualizations()

        print(f"Visualizations saved to: {self.output_dir}")

    def _generate_visualizations(self) -> None:
        """Generate visualization images/video."""
        import cv2
        from .bev import draw_bev_occupancy
        from .composite import (
            create_composite_visualization,
            create_comparison_visualization,
            save_video,
        )
        from .utils import load_camera_images, compute_metrics

        video_frames = []
        dataset = self._datamodule.test_dataset if self._datamodule else None

        for result in self._predictions:
            idx = result['idx']
            pred_occ = result['pred_occ']
            gt_occ = result.get('gt_occ')
            mask = result.get('mask_camera')

            # Save predictions if requested
            if self.save_predictions:
                np.savez_compressed(
                    os.path.join(self.output_dir, 'predictions', f'{idx:06d}.npz'),
                    pred=pred_occ,
                )

            # Load camera images
            images = {}
            if dataset:
                try:
                    data_info = dataset.get_data_info(idx)
                    data_root = getattr(self._datamodule, 'data_root', 'data/nuscenes')
                    images = load_camera_images(data_info, data_root)
                except Exception as e:
                    pass

            # Compute metrics
            metrics = {}
            if gt_occ is not None:
                metrics = compute_metrics(pred_occ, gt_occ, mask)

            # BEV visualization
            if self.mode in ['bev', 'composite', 'full']:
                bev_img = draw_bev_occupancy(pred_occ, output_size=(800, 800))
                cv2.imwrite(
                    os.path.join(self.output_dir, 'bev', f'{idx:06d}_bev.jpg'),
                    bev_img
                )

            # Composite visualization
            if self.mode in ['composite', 'full'] and images:
                title = f"Sample {idx}"
                if metrics:
                    title += f" | mIoU: {metrics.get('mIoU', 0):.4f}"

                if gt_occ is not None:
                    composite = create_comparison_visualization(
                        images, pred_occ, gt_occ, title=title
                    )
                else:
                    composite = create_composite_visualization(
                        images, pred_occ, title=title
                    )

                cv2.imwrite(
                    os.path.join(self.output_dir, 'composite', f'{idx:06d}_composite.jpg'),
                    composite
                )

                if self.output_format == 'video':
                    video_frames.append(composite)

            # 3D visualization
            if self.mode == 'full':
                try:
                    from .voxel_3d import render_occupancy_3d
                    render_occupancy_3d(
                        pred_occ,
                        os.path.join(self.output_dir, '3d', f'{idx:06d}_3d.jpg'),
                        mask=mask
                    )
                except Exception:
                    pass

        # Save video
        if self.output_format == 'video' and video_frames:
            video_path = os.path.join(self.output_dir, 'visualization.mp4')
            save_video(video_frames, video_path, fps=self.fps)


class ValidationVisualizationCallback(VisualizationCallback):
    """Callback to visualize validation samples during training.

    Runs at the end of each validation epoch.
    """

    def __init__(
        self,
        num_samples: int = 10,
        every_n_epochs: int = 5,
        **kwargs
    ):
        super().__init__(num_samples=num_samples, **kwargs)
        self.every_n_epochs = every_n_epochs

    def on_validation_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """Check if we should visualize this epoch."""
        current_epoch = trainer.current_epoch
        self._should_visualize = (current_epoch + 1) % self.every_n_epochs == 0

        if self._should_visualize:
            self._predictions = []
            self._datamodule = trainer.datamodule

            if self.output_dir is None:
                if trainer.log_dir:
                    self.output_dir = os.path.join(trainer.log_dir, "val_visualizations")
                else:
                    self.output_dir = "val_visualizations"

            epoch_dir = os.path.join(self.output_dir, f"epoch_{current_epoch:03d}")
            os.makedirs(epoch_dir, exist_ok=True)
            for subdir in ['bev', 'composite']:
                os.makedirs(os.path.join(epoch_dir, subdir), exist_ok=True)
            self._epoch_dir = epoch_dir

    def on_validation_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: Dict[str, Any],
        batch: Dict[str, torch.Tensor],
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        """Collect validation predictions."""
        if not getattr(self, '_should_visualize', False):
            return
        if batch_idx >= self.num_samples:
            return

        # Get prediction from outputs
        if 'preds' in outputs:
            pred_occ = outputs['preds']
            if isinstance(pred_occ, torch.Tensor):
                pred_occ = pred_occ.cpu().numpy()
        else:
            return

        result = {
            'idx': batch_idx,
            'pred_occ': pred_occ[0] if pred_occ.ndim == 4 else pred_occ,
        }

        if 'gt_occ' in batch:
            result['gt_occ'] = batch['gt_occ'].cpu().numpy()[0]
        if 'mask' in outputs:
            result['mask_camera'] = outputs['mask'].cpu().numpy()[0]

        self._predictions.append(result)

    def on_validation_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """Generate visualizations at end of validation."""
        if not getattr(self, '_should_visualize', False):
            return
        if not self._predictions:
            return

        # Temporarily override output_dir
        original_output_dir = self.output_dir
        self.output_dir = self._epoch_dir

        self._generate_visualizations()

        self.output_dir = original_output_dir
        print(f"Validation visualizations saved to: {self._epoch_dir}")
