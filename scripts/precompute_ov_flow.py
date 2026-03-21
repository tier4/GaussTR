"""Pre-compute optical flow from DINOv3CLIP feature matching.

For each frame pair (current, past), compute dense correspondence by
finding the best-matching feature vector in the past frame for each
pixel in the current frame. The displacement = semantic optical flow.

This provides strong supervision signals for the motion head, which
can't learn from weak photometric signals within 4000 training steps.

Output format: [2, Hf, Wf] float16 .npy files (dx, dy in feature-space pixels)

Usage:
    python -m scripts.precompute_ov_flow --chunk 0 --num-chunks 8

Note: Uses pickle for loading T4 annotation files (project standard format).
"""

import os
import sys
import argparse
import pickle  # T4 annotations are stored as pickle (project standard)
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm


def compute_ov_flow(feat_cur, feat_past, search_radius=8):
    """Compute optical flow via DINOv3CLIP feature matching.

    For each pixel in feat_cur, find the best-matching pixel in feat_past
    within a local search window using cosine similarity.

    Args:
        feat_cur: [C, H, W] current-frame features
        feat_past: [C, H, W] past-frame features
        search_radius: search window radius in pixels

    Returns:
        flow: [2, H, W] optical flow (dx, dy) in feature-space pixels
        confidence: [H, W] matching confidence (cosine similarity)
    """
    C, H, W = feat_cur.shape
    device = feat_cur.device

    # Normalize features for cosine similarity
    feat_cur_norm = F.normalize(feat_cur, dim=0)  # [C, H, W]
    feat_past_norm = F.normalize(feat_past, dim=0)

    # Pad past features for border-safe search
    pad = search_radius
    feat_past_padded = F.pad(feat_past_norm, (pad, pad, pad, pad), mode='constant', value=0)

    # Initialize best matches
    best_sim = torch.full((H, W), -1.0, device=device)
    best_dy = torch.zeros(H, W, device=device)
    best_dx = torch.zeros(H, W, device=device)

    # Exhaustive local search
    for dy in range(-search_radius, search_radius + 1):
        for dx in range(-search_radius, search_radius + 1):
            past_slice = feat_past_padded[:, pad + dy:pad + dy + H, pad + dx:pad + dx + W]
            sim = (feat_cur_norm * past_slice).sum(dim=0)  # [H, W]
            better = sim > best_sim
            best_sim = torch.where(better, sim, best_sim)
            best_dy = torch.where(better, torch.tensor(float(dy), device=device), best_dy)
            best_dx = torch.where(better, torch.tensor(float(dx), device=device), best_dx)

    flow = torch.stack([best_dx, best_dy], dim=0)  # [2, H, W]
    return flow, best_sim


def _extract_chunk_name(path):
    parts = path.replace("\\", "/").split("/")
    for i, part in enumerate(parts):
        if part == "t4_datasets" and i + 1 < len(parts):
            return parts[i + 1]
    return ""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--feat-root', default='/mnt/nvme3/T4_datasets_dinov3clip')
    parser.add_argument('--output-root', default='/mnt/nvme3/T4_datasets_ov_flow')
    parser.add_argument('--anno-file', default='/mnt/nvme2/T4_processed/t4_train_temporal.pkl')
    parser.add_argument('--search-radius', type=int, default=8)
    parser.add_argument('--sweep-indices', nargs='+', type=int, default=[4, 6])
    parser.add_argument('--chunk', type=int, default=0)
    parser.add_argument('--num-chunks', type=int, default=1)
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()

    with open(args.anno_file, 'rb') as f:
        infos = pickle.load(f)  # noqa: S301 — trusted T4 annotation format

    total = len(infos)
    chunk_size = (total + args.num_chunks - 1) // args.num_chunks
    start = args.chunk * chunk_size
    end = min(start + chunk_size, total)
    infos = infos[start:end]
    print(f"Processing chunk {args.chunk}/{args.num_chunks}: samples {start}-{end}")

    device = torch.device(args.device)
    os.makedirs(args.output_root, exist_ok=True)

    cam_names = ['CAM_FRONT_WIDE', 'CAM_FRONT_LEFT_WIDE', 'CAM_FRONT_RIGHT_WIDE',
                 'CAM_BACK_LEFT_WIDE', 'CAM_BACK_RIGHT_WIDE']

    for info in tqdm(infos, desc=f"Chunk {args.chunk}"):
        sweeps = info.get('sweeps', [])
        if len(sweeps) == 0:
            continue

        for cam_name in cam_names:
            if cam_name not in info.get('images', {}):
                continue

            cur_path = info['images'][cam_name]['img_path']
            cur_basename = os.path.basename(cur_path).split('.')[0]
            cur_cam = os.path.basename(os.path.dirname(cur_path))
            chunk_name = _extract_chunk_name(cur_path)

            cur_feat_path = os.path.join(args.feat_root, chunk_name, cur_cam, cur_basename + '.npy')
            if not os.path.exists(cur_feat_path):
                continue

            cur_feat = np.load(cur_feat_path)
            if cur_feat.dtype == np.int8:
                cur_feat = cur_feat.astype(np.float32) / 127.0
            cur_feat = torch.from_numpy(cur_feat).to(device)

            for wi in args.sweep_indices:
                wi_clamped = min(wi, len(sweeps) - 1)
                sweep = sweeps[wi_clamped]

                if cam_name not in sweep.get('images', {}):
                    continue

                past_path = sweep['images'][cam_name]['img_path']
                past_basename = os.path.basename(past_path).split('.')[0]
                past_cam = os.path.basename(os.path.dirname(past_path))
                chunk_p = _extract_chunk_name(past_path)

                past_feat_path = os.path.join(args.feat_root, chunk_p, past_cam, past_basename + '.npy')
                if not os.path.exists(past_feat_path):
                    continue

                past_feat = np.load(past_feat_path)
                if past_feat.dtype == np.int8:
                    past_feat = past_feat.astype(np.float32) / 127.0
                past_feat = torch.from_numpy(past_feat).to(device)

                with torch.no_grad():
                    flow, confidence = compute_ov_flow(cur_feat, past_feat, args.search_radius)

                out_dir = os.path.join(args.output_root, chunk_name, cur_cam)
                os.makedirs(out_dir, exist_ok=True)
                out_path = os.path.join(out_dir, f"{cur_basename}_sweep{wi}.npy")
                np.save(out_path, flow.cpu().numpy().astype(np.float16))

    print(f"Done! Chunk {args.chunk}")


if __name__ == '__main__':
    main()
