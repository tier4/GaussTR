#!/usr/bin/env python
"""Visualization script for GaussTR occupancy predictions.

Usage:
    # Visualize from checkpoint
    python -m scripts.visualize checkpoint=path/to/checkpoint.ckpt

    # Visualize specific samples
    python -m scripts.visualize checkpoint=path/to/checkpoint.ckpt vis.num_samples=100

    # Generate video
    python -m scripts.visualize checkpoint=path/to/checkpoint.ckpt vis.output_format=video

    # BEV only (fast)
    python -m scripts.visualize checkpoint=path/to/checkpoint.ckpt vis.mode=bev

    # Full visualization with 3D
    python -m scripts.visualize checkpoint=path/to/checkpoint.ckpt vis.mode=full
"""

import os
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional, List, Dict, Any

import hydra
import torch
import numpy as np
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def setup_model_and_data(cfg: DictConfig):
    """Setup model and datamodule for visualization."""
    from models import GaussTRLightning
    from dataset import GaussTRDataModule

    # Load model
    checkpoint_path = cfg.get('checkpoint')
    if not checkpoint_path:
        raise ValueError("checkpoint path is required")

    print(f"Loading checkpoint from: {checkpoint_path}")

    model_cfg = OmegaConf.to_container(cfg.model, resolve=True)

    if checkpoint_path.endswith('.ckpt'):
        model = GaussTRLightning.load_from_checkpoint(
            checkpoint_path,
            **model_cfg
        )
    else:
        model = GaussTRLightning(**model_cfg)
        ckpt = torch.load(checkpoint_path, map_location='cpu')
        state_dict = ckpt.get('state_dict', ckpt)
        state_dict = {k.replace('model.', ''): v for k, v in state_dict.items()}
        model.load_state_dict(state_dict, strict=False)

    model.eval()
    model.cuda()

    # Setup datamodule
    data_cfg = OmegaConf.to_container(cfg.data, resolve=True)
    datamodule = GaussTRDataModule(**data_cfg)
    datamodule.setup('test')

    return model, datamodule


@torch.no_grad()
def generate_predictions(
    model,
    dataloader,
    num_samples: int = -1,
    save_predictions: bool = False,
    output_dir: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Generate predictions for visualization.

    Args:
        model: GaussTR model.
        dataloader: DataLoader for test/val set.
        num_samples: Number of samples to process (-1 for all).
        save_predictions: Whether to save predictions to disk.
        output_dir: Output directory for saved predictions.

    Returns:
        List of prediction dictionaries.
    """
    results = []

    if save_predictions and output_dir:
        pred_dir = os.path.join(output_dir, 'predictions')
        os.makedirs(pred_dir, exist_ok=True)

    total = len(dataloader) if num_samples < 0 else min(num_samples, len(dataloader))

    for idx, batch in enumerate(tqdm(dataloader, total=total, desc="Generating predictions")):
        if num_samples > 0 and idx >= num_samples:
            break

        # Move to GPU
        batch_gpu = {k: v.cuda() if isinstance(v, torch.Tensor) else v
                     for k, v in batch.items()}

        # Forward pass
        pred_occ = model(
            images=batch_gpu['images'],
            feats=batch_gpu['feats'],
            depth=batch_gpu['depth'],
            cam2img=batch_gpu['cam2img'],
            cam2ego=batch_gpu['cam2ego'],
            img_aug_mat=batch_gpu.get('img_aug_mat'),
        )

        # Convert to numpy
        pred_np = pred_occ.cpu().numpy()

        result = {
            'idx': idx,
            'pred_occ': pred_np[0] if pred_np.ndim == 4 else pred_np,
        }

        # Add ground truth if available
        if 'gt_occ' in batch:
            result['gt_occ'] = batch['gt_occ'].cpu().numpy()[0]
        if 'mask_camera' in batch:
            result['mask_camera'] = batch['mask_camera'].cpu().numpy()[0]

        # Save prediction
        if save_predictions and output_dir:
            np.savez_compressed(
                os.path.join(pred_dir, f'{idx:06d}.npz'),
                pred=result['pred_occ'],
            )

        results.append(result)

    return results


def visualize_samples(
    predictions: List[Dict[str, Any]],
    datamodule,
    output_dir: str,
    mode: str = 'bev',
    output_format: str = 'image',
    fps: int = 10,
):
    """Generate visualizations from predictions.

    Args:
        predictions: List of prediction dictionaries.
        datamodule: DataModule for loading images.
        output_dir: Output directory.
        mode: Visualization mode ('bev', 'composite', 'full').
        output_format: Output format ('image' or 'video').
        fps: Video FPS if output_format='video'.
    """
    from visualization import (
        draw_bev_occupancy,
        create_composite_visualization,
        create_comparison_visualization,
        render_occupancy_3d,
        save_video,
    )
    from visualization.utils import (
        load_camera_images,
        create_output_dirs,
        compute_metrics,
    )

    # Create output directories
    dirs = create_output_dirs(output_dir)

    # Get dataset for loading images
    dataset = datamodule.test_dataset

    video_frames = []

    for result in tqdm(predictions, desc=f"Generating {mode} visualizations"):
        idx = result['idx']
        pred_occ = result['pred_occ']
        gt_occ = result.get('gt_occ')
        mask = result.get('mask_camera')

        # Load camera images
        try:
            data_info = dataset.get_data_info(idx)
            images = load_camera_images(data_info, datamodule.data_root)
        except Exception as e:
            print(f"Warning: Could not load images for sample {idx}: {e}")
            images = {}

        # Compute metrics if GT available
        metrics = {}
        if gt_occ is not None:
            metrics = compute_metrics(pred_occ, gt_occ, mask)

        # Generate visualizations based on mode
        if mode in ['bev', 'composite', 'full']:
            # BEV visualization
            bev_path = os.path.join(dirs['bev'], f'{idx:06d}_bev.jpg')
            bev_img = draw_bev_occupancy(pred_occ, output_size=(800, 800))
            import cv2
            cv2.imwrite(bev_path, bev_img)

            # Save GT BEV if available
            if gt_occ is not None:
                gt_bev_path = os.path.join(dirs['bev'], f'{idx:06d}_gt_bev.jpg')
                gt_bev_img = draw_bev_occupancy(gt_occ, output_size=(800, 800))
                cv2.imwrite(gt_bev_path, gt_bev_img)

        if mode in ['composite', 'full'] and images:
            # Composite visualization
            if gt_occ is not None:
                composite = create_comparison_visualization(
                    images, pred_occ, gt_occ,
                    mask=mask,
                    title=f"Sample {idx} | mIoU: {metrics.get('mIoU', 0):.4f}"
                )
            else:
                composite = create_composite_visualization(
                    images, pred_occ,
                    title=f"Sample {idx}"
                )
            comp_path = os.path.join(dirs['composite'], f'{idx:06d}_composite.jpg')
            cv2.imwrite(comp_path, composite)

            if output_format == 'video':
                video_frames.append(composite)

        if mode == 'full':
            # 3D visualization
            try:
                render_path = os.path.join(dirs['3d'], f'{idx:06d}_3d.jpg')
                render_occupancy_3d(pred_occ, render_path, mask=mask)
            except Exception as e:
                print(f"Warning: 3D rendering failed for sample {idx}: {e}")

    # Save video if requested
    if output_format == 'video' and video_frames:
        video_path = os.path.join(dirs['video'], 'visualization.mp4')
        save_video(video_frames, video_path, fps=fps)
        print(f"Saved video to: {video_path}")


@hydra.main(
    version_base=None,
    config_path="../config",
    config_name="gausstr_featup"
)
def main(cfg: DictConfig) -> None:
    """Main visualization function."""
    print(OmegaConf.to_yaml(cfg))

    # Get visualization config with defaults
    vis_cfg = cfg.get('vis', {})
    num_samples = vis_cfg.get('num_samples', 100)
    mode = vis_cfg.get('mode', 'composite')
    output_format = vis_cfg.get('output_format', 'image')
    fps = vis_cfg.get('fps', 10)
    save_predictions = vis_cfg.get('save_predictions', False)

    # Setup output directory
    checkpoint_path = cfg.get('checkpoint', '')
    if checkpoint_path:
        ckpt_dir = Path(checkpoint_path).parent
        output_dir = os.path.join(ckpt_dir, 'visualizations')
    else:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_dir = f'work_dirs/visualizations_{timestamp}'

    output_dir = vis_cfg.get('output_dir', output_dir)
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory: {output_dir}")

    # Setup model and data
    model, datamodule = setup_model_and_data(cfg)

    # Generate predictions
    dataloader = datamodule.test_dataloader()
    predictions = generate_predictions(
        model, dataloader,
        num_samples=num_samples,
        save_predictions=save_predictions,
        output_dir=output_dir,
    )

    # Generate visualizations
    visualize_samples(
        predictions, datamodule, output_dir,
        mode=mode,
        output_format=output_format,
        fps=fps,
    )

    print(f"\nVisualization complete!")
    print(f"  Output: {output_dir}")
    print(f"  Samples: {len(predictions)}")
    print(f"  Mode: {mode}")


if __name__ == '__main__':
    main()
