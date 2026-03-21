"""Gaussian Memory Bank: cache past-frame Gaussian predictions for temporal aggregation.

SelfOccFlow Phase 3: instead of just enforcing consistency, MERGE static
Gaussians from past frames into the current frame for denser coverage.

The memory bank stores the most recent Gaussian predictions. On each forward
pass, it retrieves past Gaussians, transforms them via ego-motion to the
current ego frame, and returns them for merging with current-frame Gaussians.
"""

import torch
from collections import deque
from dataclasses import dataclass
from typing import Optional, List


@dataclass
class GaussianFrame:
    """Cached Gaussian prediction for one frame."""
    means: torch.Tensor        # [Q, 3] in ego frame
    scales: torch.Tensor       # [Q, 3]
    rotations: torch.Tensor    # [Q, 4]
    opacities: torch.Tensor    # [Q]
    ovs: Optional[torch.Tensor]  # [Q, C] OV features (PCA-projected)
    branch_probs: Optional[torch.Tensor]  # [Q, 2]
    ego2global: torch.Tensor   # [4, 4] ego frame transform
    scene_token: str           # scene identifier (don't merge across scenes)


class GaussianMemoryBank:
    """FIFO memory bank for past-frame Gaussian predictions.

    Stores up to `max_frames` past frames. On retrieval, transforms past
    Gaussians to the current ego frame and filters to static-only.

    Key design choices:
    - Detached tensors only (no gradient to past frames)
    - CPU storage to minimize GPU memory overhead
    - Scene-aware: only merge frames from the same scene
    - Static-only: only aggregate static Gaussians (p_static > threshold)
    """

    def __init__(self, max_frames: int = 2, static_threshold: float = 0.6,
                 temporal_decay: float = 0.8):
        self.max_frames = max_frames
        self.static_threshold = static_threshold
        self.temporal_decay = temporal_decay
        self.buffer: deque = deque(maxlen=max_frames)

    @torch.no_grad()
    def push(self, means, scales, rotations, opacities, ovs, branch_probs,
             ego2global, scene_token):
        """Store current frame's Gaussians in the bank (detached, on CPU)."""
        frame = GaussianFrame(
            means=means.detach().cpu(),
            scales=scales.detach().cpu(),
            rotations=rotations.detach().cpu(),
            opacities=opacities.detach().cpu(),
            ovs=ovs.detach().cpu() if ovs is not None else None,
            branch_probs=branch_probs.detach().cpu() if branch_probs is not None else None,
            ego2global=ego2global.detach().cpu(),
            scene_token=scene_token,
        )
        self.buffer.append(frame)

    @torch.no_grad()
    def retrieve(self, current_ego2global, current_scene_token, device):
        """Retrieve past static Gaussians transformed to current ego frame.

        Returns:
            means: [M, 3] aggregated past static Gaussian positions
            scales: [M, 3]
            rotations: [M, 4]
            opacities: [M] (decayed by temporal_decay^dt)
            ovs: [M, C] or None
            valid: True if any past Gaussians were retrieved
        """
        all_means = []
        all_scales = []
        all_rots = []
        all_opacities = []
        all_ovs = []

        cur_e2g = current_ego2global.float().to(device)
        cur_g2e = torch.inverse(cur_e2g)  # global → current ego

        for idx, frame in enumerate(self.buffer):
            # Skip frames from different scenes
            if frame.scene_token != current_scene_token:
                continue

            # Filter to static Gaussians
            if frame.branch_probs is not None:
                static_mask = frame.branch_probs[:, 0] > self.static_threshold
            else:
                static_mask = torch.ones(frame.means.shape[0], dtype=torch.bool)

            if static_mask.sum() == 0:
                continue

            # Transform from past ego → global → current ego
            past_e2g = frame.ego2global.float().to(device)
            transform = cur_g2e @ past_e2g  # [4, 4]

            past_means = frame.means[static_mask].to(device)  # [K, 3]
            # Homogeneous transform
            ones = torch.ones(past_means.shape[0], 1, device=device)
            past_homo = torch.cat([past_means, ones], dim=1)  # [K, 4]
            cur_means = (transform @ past_homo.T).T[:, :3]  # [K, 3]

            # Temporal decay: older frames contribute less
            dt = len(self.buffer) - idx  # distance from current
            decay = self.temporal_decay ** dt

            all_means.append(cur_means)
            all_scales.append(frame.scales[static_mask].to(device))
            all_rots.append(frame.rotations[static_mask].to(device))
            all_opacities.append(frame.opacities[static_mask].to(device) * decay)

            if frame.ovs is not None:
                all_ovs.append(frame.ovs[static_mask].to(device))

        if len(all_means) == 0:
            return None

        result = {
            'means': torch.cat(all_means, dim=0).to(device),
            'scales': torch.cat(all_scales, dim=0).to(device),
            'rotations': torch.cat(all_rots, dim=0).to(device),
            'opacities': torch.cat(all_opacities, dim=0).to(device),
        }
        if all_ovs:
            result['ovs'] = torch.cat(all_ovs, dim=0).to(device)
        else:
            result['ovs'] = None

        return result

    def clear(self):
        """Clear the memory bank (e.g. on scene change)."""
        self.buffer.clear()
