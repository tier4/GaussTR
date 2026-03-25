from dataclasses import dataclass
from typing import Optional

import torch


@dataclass
class GaussianPrediction:
    means: Optional[torch.Tensor] = None
    scales: Optional[torch.Tensor] = None
    rotations: Optional[torch.Tensor] = None
    opacities: Optional[torch.Tensor] = None
    colors: Optional[torch.Tensor] = None
    ovs: Optional[torch.Tensor] = None
    semantics: Optional[torch.Tensor] = None
    # Phase 2: static/dynamic branching (SelfOccFlow-inspired)
    branch_logits: Optional[torch.Tensor] = None   # [B, Q, 2] raw logits
    branch_probs: Optional[torch.Tensor] = None     # [B, Q, 2] softmax probs
    # Phase 4: per-query motion offsets for dynamic objects (SelfOccFlow-inspired)
    motion_offsets: Optional[torch.Tensor] = None   # [B, Q, 2*P] P=num_past_frames, XY per frame
    # Auxiliary depth prediction (direct from query features, bypasses rendering)
    aux_depth: Optional[torch.Tensor] = None        # [B, Q, 1] predicted depth per Gaussian
