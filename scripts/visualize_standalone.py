#!/usr/bin/env python
"""Standalone visualization script - generates visualizations from saved predictions.

Usage:
    # Step 1: Run test.py with save_predictions=true to save predictions
    python -m scripts.test checkpoint=path/to/checkpoint.ckpt save_predictions=true trainer.devices=8

    # Step 2: Generate visualizations from saved predictions
    python -m scripts.visualize_standalone --pred-dir path/to/visualizations --bev
    python -m scripts.visualize_standalone --pred-dir path/to/visualizations --3d
    python -m scripts.visualize_standalone --pred-dir path/to/visualizations --bev --3d
"""

import os
# Suppress Filament/Open3D info messages before any imports
os.environ['OPEN3D_VERBOSITY_LEVEL'] = 'warning'

import argparse
import sys
from pathlib import Path
from typing import Optional

import numpy as np
from tqdm import tqdm

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def parse_args():
    parser = argparse.ArgumentParser(description='GaussTR Visualization from Saved Predictions')
    parser.add_argument('--pred-dir', '-p', type=str, required=True,
                        help='Path to directory containing predictions/ folder (output from test.py)')
    parser.add_argument('--num-samples', '-n', type=int, default=-1,
                        help='Number of samples to visualize (-1 for all, default: all)')
    parser.add_argument('--bev', action='store_true',
                        help='Generate BEV composite visualizations')
    parser.add_argument('--3d', dest='render_3d', action='store_true',
                        help='Generate 3D composite visualizations')
    parser.add_argument('--both', action='store_true',
                        help='Generate full composite visualizations (cameras + 3D + BEV + legend)')
    parser.add_argument('--video', action='store_true',
                        help='Generate video from visualizations')
    parser.add_argument('--fps', type=int, default=10,
                        help='Video FPS')
    parser.add_argument('--data-root', type=str, default='data/nuscenes',
                        help='Path to nuScenes data (for camera images)')
    parser.add_argument('--ann-file', type=str,
                        default='data/nuscenes/nuscenes_infos_val.pkl',
                        help='Path to annotation file (for camera paths)')
    parser.add_argument('--num-workers', type=int, default=16,
                        help='Number of parallel workers for visualization')
    return parser.parse_args()


def generate_vis_for_sample(args_tuple):
    """Generate visualization for a single sample - loads data on demand."""
    pred_file_path, data_info, data_root, output_dir, render_bev, render_3d, render_both = args_tuple

    import cv2
    from visualization import draw_bev_occupancy
    from visualization.bev import draw_bev_comparison
    from visualization.composite import (
        create_comparison_visualization,
        create_composite_visualization,
        create_3d_composite_visualization,
        create_full_composite_visualization,
    )
    from visualization.utils import load_camera_images

    # Load prediction on demand (not pre-loaded)
    data = np.load(pred_file_path)
    pred_occ = data['pred']
    gt_occ = data['gt'] if 'gt' in data else None
    mask = data['mask'] if 'mask' in data else None

    token = os.path.basename(pred_file_path).replace('.npz', '')

    # Load camera images for composite
    images = {}
    try:
        images = load_camera_images(data_info, data_root)
    except:
        pass

    # Create title from timestamp
    title = f"Sample {token[:15]}..."

    # Full composite visualization (cameras + 3D + BEV + legend) - saved to both/
    if render_both:
        try:
            img_both = create_full_composite_visualization(
                images, pred_occ, gt_occ, mask=mask, title=title
            )
            cv2.imwrite(os.path.join(output_dir, 'both', f'{token}.jpg'), img_both)
        except Exception as e:
            print(f"Full composite failed for {token}: {e}")

    # BEV composite visualization (cameras + BEV) - saved to bev/
    if render_bev:
        if images:
            if gt_occ is not None:
                bev_comp = create_comparison_visualization(images, pred_occ, gt_occ, title=title)
            else:
                bev_comp = create_composite_visualization(images, pred_occ, title=title)
            cv2.imwrite(os.path.join(output_dir, 'bev', f'{token}.jpg'), bev_comp)
        else:
            # Fallback: just BEV if no camera images
            if gt_occ is not None:
                bev_img = draw_bev_comparison(pred_occ, gt_occ, output_size=(400, 400))
            else:
                bev_img = draw_bev_occupancy(pred_occ, output_size=(800, 800))
            cv2.imwrite(os.path.join(output_dir, 'bev', f'{token}.jpg'), bev_img)

    # 3D composite visualization (cameras + 3D render) - saved to 3d/
    if render_3d:
        try:
            img_3d = create_3d_composite_visualization(images, pred_occ, gt_occ, title=title, mask=mask)
            cv2.imwrite(os.path.join(output_dir, '3d', f'{token}.jpg'), img_3d)
        except Exception as e:
            # Fallback: just 3D render if composite fails
            try:
                from visualization import render_occupancy_3d
                render_occupancy_3d(pred_occ, os.path.join(output_dir, '3d', f'{token}.jpg'), mask=mask)
            except:
                pass

    return token


def main():
    args = parse_args()

    # Check predictions directory
    pred_dir = os.path.join(args.pred_dir, 'predictions')
    if not os.path.exists(pred_dir):
        # Maybe user passed the predictions/ folder directly
        if os.path.exists(args.pred_dir) and any(f.endswith('.npz') for f in os.listdir(args.pred_dir)):
            pred_dir = args.pred_dir
            args.pred_dir = os.path.dirname(pred_dir)
        else:
            print(f"Error: No predictions found at {pred_dir}")
            print("Run test.py with save_predictions=true first to generate predictions.")
            return

    output_dir = args.pred_dir

    # Create output subdirectories
    for subdir in ['bev', '3d', 'both']:
        os.makedirs(os.path.join(output_dir, subdir), exist_ok=True)

    # Find all prediction files
    pred_files = sorted([f for f in os.listdir(pred_dir) if f.endswith('.npz')])
    if args.num_samples > 0:
        pred_files = pred_files[:args.num_samples]

    print(f"Found {len(pred_files)} predictions in {pred_dir}")

    # Determine what to render
    render_bev = args.bev
    render_3d = args.render_3d
    render_both = args.both
    if not render_bev and not render_3d and not render_both:
        # Default to 'both' mode if no mode specified
        render_both = True

    # Try to load dataset for camera image paths
    dataset = None
    try:
        from dataset import GaussTRDataModule
        datamodule = GaussTRDataModule(
            train_ann_file=args.ann_file,
            val_ann_file=args.ann_file,
            data_root=args.data_root,
            batch_size=1,
            num_workers=0,
        )
        datamodule.setup('test')
        dataset = datamodule.test_dataset
        print(f"Loaded dataset with {len(dataset)} samples")
    except Exception as e:
        print(f"Warning: Could not load dataset for camera images: {e}")
        print("Visualizations will not include camera images.")

    # Build timestamp to index mapping if dataset is available
    timestamp_to_idx = {}
    if dataset is not None:
        for i in tqdm(range(len(dataset)), desc="Building index"):
            try:
                info = dataset.get_data_info(i)
                ts = info.get('timestamp', 0)
                # Format timestamp same as filename
                ts_str = f"{ts:.6f}"
                timestamp_to_idx[ts_str] = i
            except:
                pass

    # Prepare visualization arguments with FILE PATHS (not pre-loaded data)
    vis_args = []
    for pred_file in pred_files:
        filename = pred_file.replace('.npz', '')
        pred_file_path = os.path.join(pred_dir, pred_file)

        # Get data info for camera images using timestamp
        data_info = {}
        if dataset is not None and filename in timestamp_to_idx:
            try:
                idx = timestamp_to_idx[filename]
                data_info = dataset.get_data_info(idx)
            except:
                pass

        vis_args.append((
            pred_file_path,  # Pass file path, load inside worker
            data_info,
            args.data_root,
            output_dir,
            render_bev,
            render_3d,
            render_both,
        ))

    # CUDA warmup for fast 3D rendering
    import torch
    if torch.cuda.is_available() and (render_3d or render_both):
        from visualization.voxel_3d import render_occupancy_3d_to_array
        dummy = np.zeros((200, 200, 16), dtype=np.int64)
        _ = render_occupancy_3d_to_array(dummy, image_size=(100, 100))
        torch.cuda.synchronize()
        print("CUDA warmup done")

    # Sequential processing (faster than ThreadPoolExecutor due to CUDA overhead)
    print(f"Generating visualizations (BEV={render_bev}, 3D={render_3d}, Both={render_both})...")
    for args_tuple in tqdm(vis_args, desc="Visualizing"):
        generate_vis_for_sample(args_tuple)

    # Generate video if requested
    if args.video:
        import cv2
        from visualization.composite import save_video

        print("Generating video...")

        # Timestamps naturally sort in chronological order
        filenames = sorted([pred_file.replace('.npz', '') for pred_file in pred_files])

        frames = []
        for filename in filenames:
            bev_path = os.path.join(output_dir, 'bev', f'{filename}.jpg')
            if os.path.exists(bev_path):
                frames.append(cv2.imread(bev_path))

        if frames:
            video_path = os.path.join(output_dir, 'visualization.mp4')
            save_video(frames, video_path, fps=args.fps)
            print(f"Saved video: {video_path}")

    print(f"\nVisualization complete!")
    print(f"  Output: {output_dir}")
    if render_both:
        print(f"  Both (cameras+3D+BEV+legend): {os.path.join(output_dir, 'both')}")
    if render_bev:
        print(f"  BEV: {os.path.join(output_dir, 'bev')}")
    if render_3d:
        print(f"  3D: {os.path.join(output_dir, '3d')}")

    # Suppress Filament cleanup messages at exit
    sys.stdout.flush()
    sys.stderr.flush()
    devnull = open(os.devnull, 'w')
    os.dup2(devnull.fileno(), 1)  # Redirect stdout to /dev/null
    os.dup2(devnull.fileno(), 2)  # Redirect stderr to /dev/null


if __name__ == '__main__':
    main()
