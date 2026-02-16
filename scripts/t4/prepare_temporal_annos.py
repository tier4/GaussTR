"""Prepare T4 annotations with temporal sweep information for PG-Occ.

For each frame, finds N past frames in the same scene to use as temporal context.
Saves augmented annotations with 'sweeps' field containing past frame info.

Usage:
    python -m scripts.t4.prepare_temporal_annos
    python -m scripts.t4.prepare_temporal_annos --num_sweeps 3
"""

import argparse
import os
import pickle
from collections import defaultdict

import numpy as np


def prepare_sweep_annos(infos, num_sweeps=7):
    """Add temporal sweep info to each annotation.

    For each frame, finds up to `num_sweeps` past frames from the same scene,
    sorted by timestamp (most recent first).

    Args:
        infos: List of annotation dicts with 'scene_token', 'timestamp', 'images'.
        num_sweeps: Number of past frames to include.

    Returns:
        Updated infos with 'sweeps' field added.
    """
    # Group frames by scene
    scene_frames = defaultdict(list)
    for idx, info in enumerate(infos):
        scene_frames[info['scene_token']].append((info['timestamp'], idx))

    # Sort each scene by timestamp
    for scene_token in scene_frames:
        scene_frames[scene_token].sort(key=lambda x: x[0])

    # For each frame, find past sweeps
    for scene_token, frames in scene_frames.items():
        for pos, (ts, idx) in enumerate(frames):
            sweeps = []
            # Collect up to num_sweeps past frames
            for sweep_pos in range(pos - 1, max(pos - 1 - num_sweeps, -1), -1):
                if sweep_pos < 0:
                    break
                sweep_ts, sweep_idx = frames[sweep_pos]
                sweep_info = infos[sweep_idx]

                sweep_data = {
                    'token': sweep_info['token'],
                    'timestamp': sweep_info['timestamp'],
                    'images': {},
                }
                for cam_name, cam_info in sweep_info['images'].items():
                    sweep_data['images'][cam_name] = {
                        'img_path': cam_info['img_path'],
                        'cam2img': cam_info['cam2img'],
                        'cam2ego': cam_info['cam2ego'],
                        'ego2global': cam_info['ego2global'],
                    }
                sweeps.append(sweep_data)

            infos[idx]['sweeps'] = sweeps

    return infos


def main():
    parser = argparse.ArgumentParser(description='Prepare T4 temporal annotations')
    parser.add_argument('--data_root', default='/mnt/nvme2/T4_processed',
                        help='Root directory for processed T4 data')
    parser.add_argument('--num_sweeps', type=int, default=7,
                        help='Number of past sweep frames (default: 7, total 8 with current)')
    args = parser.parse_args()

    for split in ['train', 'val']:
        in_file = os.path.join(args.data_root, f't4_infos_{split}.pkl')
        out_file = os.path.join(args.data_root, f't4_infos_{split}_sweep.pkl')

        if not os.path.exists(in_file):
            print(f"Skipping {split}: {in_file} not found")
            continue

        print(f"Loading {in_file}...")
        with open(in_file, 'rb') as f:
            data = pickle.load(f)  # noqa: S301 - trusted local annotation files

        if isinstance(data, dict) and 'data_list' in data:
            infos = data['data_list']
        elif isinstance(data, dict) and 'infos' in data:
            infos = data['infos']
        elif isinstance(data, list):
            infos = data
        else:
            raise ValueError(f"Unknown annotation format in {in_file}")

        print(f"  {len(infos)} frames, adding {args.num_sweeps} sweeps per frame...")
        infos = prepare_sweep_annos(infos, num_sweeps=args.num_sweeps)

        # Count sweep statistics
        sweep_counts = [len(info.get('sweeps', [])) for info in infos]
        print(f"  Sweep counts: min={min(sweep_counts)}, max={max(sweep_counts)}, "
              f"mean={np.mean(sweep_counts):.1f}")
        print(f"  Frames with full sweeps ({args.num_sweeps}): "
              f"{sum(1 for c in sweep_counts if c == args.num_sweeps)}/{len(infos)}")

        # Save in same format as input
        if isinstance(data, dict):
            if 'data_list' in data:
                data['data_list'] = infos
            else:
                data['infos'] = infos
            out_data = data
        else:
            out_data = infos

        print(f"  Saving to {out_file}...")
        with open(out_file, 'wb') as f:
            pickle.dump(out_data, f)
        print(f"  Done!")

    print("All done.")


if __name__ == '__main__':
    main()
