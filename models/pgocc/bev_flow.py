"""BEV Similarity Flow: pseudo-label generation for dynamic object motion.

SelfOccFlow-inspired: build BEV feature grids from dynamic Gaussians,
then compute local cosine similarity matching between adjacent frames
to generate flow pseudo-labels for the motion head.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def scatter_to_bev(
    means3d: torch.Tensor,      # [Q, 3] Gaussian positions in ego frame
    features: torch.Tensor,     # [Q, C] per-Gaussian features
    weights: torch.Tensor,      # [Q] per-Gaussian weights (e.g. p_dynamic * opacity)
    pc_range: list,             # [x_min, y_min, z_min, x_max, y_max, z_max]
    bev_h: int = 100,
    bev_w: int = 100,
) -> torch.Tensor:
    """Scatter Gaussian features to a BEV grid using weighted average.

    Args:
        means3d: [Q, 3] positions
        features: [Q, C] features
        weights: [Q] importance weights
        pc_range: scene bounds
        bev_h, bev_w: BEV grid resolution

    Returns:
        bev_feat: [C, bev_h, bev_w] aggregated BEV features
    """
    device = means3d.device
    C = features.shape[1]

    # Compute BEV cell indices from XY positions
    x = means3d[:, 0]
    y = means3d[:, 1]
    x_norm = (x - pc_range[0]) / (pc_range[3] - pc_range[0])  # [0, 1]
    y_norm = (y - pc_range[1]) / (pc_range[4] - pc_range[1])  # [0, 1]

    xi = (x_norm * bev_w).long().clamp(0, bev_w - 1)
    yi = (y_norm * bev_h).long().clamp(0, bev_h - 1)

    # Valid mask: within range
    valid = (x_norm >= 0) & (x_norm < 1) & (y_norm >= 0) & (y_norm < 1) & (weights > 0.01)

    if valid.sum() == 0:
        return torch.zeros(C, bev_h, bev_w, device=device)

    xi = xi[valid]
    yi = yi[valid]
    feat = features[valid]  # [V, C]
    w = weights[valid].unsqueeze(1)  # [V, 1]

    # Flatten 2D index to 1D for scatter_add
    flat_idx = yi * bev_w + xi  # [V]

    # Weighted sum and weight sum
    weighted_feat = feat * w  # [V, C]
    bev_sum = torch.zeros(bev_h * bev_w, C, device=device)
    bev_weight = torch.zeros(bev_h * bev_w, 1, device=device)

    bev_sum.scatter_add_(0, flat_idx.unsqueeze(1).expand(-1, C), weighted_feat)
    bev_weight.scatter_add_(0, flat_idx.unsqueeze(1), w)

    # Weighted average (avoid division by zero)
    bev_feat = bev_sum / bev_weight.clamp(min=1e-6)

    # Mark empty cells as zero
    empty = (bev_weight < 1e-6).squeeze(1)
    bev_feat[empty] = 0

    return bev_feat.reshape(bev_h, bev_w, C).permute(2, 0, 1)  # [C, H, W]


def compute_bev_similarity_flow(
    bev_current: torch.Tensor,   # [C, H, W] current-frame BEV features
    bev_past: torch.Tensor,      # [C, H, W] past-frame BEV features
    search_radius: int = 5,      # max displacement in BEV cells
    pc_range: list = None,       # for converting cells to meters
    bev_h: int = 100,
    bev_w: int = 100,
) -> torch.Tensor:
    """Compute pseudo-flow by local cosine similarity matching.

    For each occupied cell in bev_current, find the best-matching cell
    in bev_past within a local search window. The displacement is the
    pseudo-flow label.

    Args:
        bev_current: [C, H, W] current BEV features
        bev_past: [C, H, W] past BEV features
        search_radius: search window radius in cells
        pc_range: scene bounds for cell-to-meter conversion

    Returns:
        flow: [2, H, W] pseudo-flow in meters (dx, dy)
        valid: [H, W] bool mask of cells with valid flow
    """
    C, H, W = bev_current.shape
    device = bev_current.device

    # Cell size in meters
    cell_x = (pc_range[3] - pc_range[0]) / bev_w
    cell_y = (pc_range[4] - pc_range[1]) / bev_h

    # Occupied cells in current frame (non-zero features)
    occupied = bev_current.norm(dim=0) > 0.1  # [H, W]

    if occupied.sum() == 0:
        return torch.zeros(2, H, W, device=device), torch.zeros(H, W, dtype=torch.bool, device=device)

    # Pad past BEV for border-safe indexing
    pad = search_radius
    bev_past_padded = F.pad(bev_past, (pad, pad, pad, pad), mode='constant', value=0)

    # Normalize features for cosine similarity
    bev_cur_norm = F.normalize(bev_current, dim=0)  # [C, H, W]
    bev_past_norm = F.normalize(bev_past_padded, dim=0)  # [C, H+2p, W+2p]

    # Build search offsets
    offsets = []
    for dy in range(-search_radius, search_radius + 1):
        for dx in range(-search_radius, search_radius + 1):
            offsets.append((dy, dx))

    # Compute cosine similarity for all offsets
    best_sim = torch.full((H, W), -1.0, device=device)
    best_dy = torch.zeros(H, W, device=device)
    best_dx = torch.zeros(H, W, device=device)

    for dy, dx in offsets:
        # Slice the padded past BEV at this offset
        past_slice = bev_past_norm[:, pad + dy:pad + dy + H, pad + dx:pad + dx + W]
        # Cosine similarity per cell
        sim = (bev_cur_norm * past_slice).sum(dim=0)  # [H, W]
        # Update best
        better = sim > best_sim
        best_sim = torch.where(better, sim, best_sim)
        best_dy = torch.where(better, torch.tensor(float(dy), device=device), best_dy)
        best_dx = torch.where(better, torch.tensor(float(dx), device=device), best_dx)

    # Convert cell offsets to meters
    flow = torch.stack([best_dx * cell_x, best_dy * cell_y], dim=0)  # [2, H, W]

    # Valid: occupied cells where the match is confident (sim > threshold)
    valid = occupied & (best_sim > 0.3)

    return flow, valid


def compute_motion_flow_loss(
    motion_offsets: torch.Tensor,   # [B, Q, 2*P] predicted motion
    means3d: torch.Tensor,          # [B, Q, 3] Gaussian positions
    branch_probs: torch.Tensor,     # [B, Q, 2] static/dynamic probs
    query_feats: torch.Tensor,      # [B, Q, C] query features (for BEV)
    t0_2_tn: torch.Tensor,          # [B, P*N, 4, 4] ego transforms
    pc_range: list,
    num_cams: int = 5,
    bev_h: int = 100,
    bev_w: int = 100,
    search_radius: int = 5,
) -> torch.Tensor:
    """Compute motion flow loss using BEV similarity pseudo-labels.

    1. Build BEV grid from dynamic Gaussians (current frame)
    2. Build BEV grid from ego-transformed dynamic Gaussians (past frame proxy)
    3. Compute similarity-based pseudo-flow
    4. Supervise motion_head predictions against pseudo-flow

    Args:
        motion_offsets: [B, Q, 2*P] predicted XY offsets per past frame
        means3d: [B, Q, 3] current Gaussian positions
        branch_probs: [B, Q, 2] static/dynamic probs
        query_feats: [B, Q, C] decoder query features
        t0_2_tn: ego transforms (used to estimate reference frame shift)
        pc_range: scene bounds
        num_cams: number of cameras

    Returns:
        Scalar motion flow loss
    """
    B = means3d.shape[0]
    device = means3d.device

    # We work with B=1 (batch_size=1 assumption)
    means = means3d[0]  # [Q, 3]
    bp = branch_probs[0]  # [Q, 2]
    feats = query_feats[0]  # [Q, C]
    p_dyn = bp[:, 1]  # [Q]

    # Dynamic weight: opacity-like weighting
    dyn_weight = p_dyn  # Use dynamic probability as weight

    # Current-frame BEV from dynamic Gaussians
    bev_current = scatter_to_bev(means, feats, dyn_weight, pc_range, bev_h, bev_w)

    # For each past frame, compute flow pseudo-labels
    num_past = motion_offsets.shape[-1] // 2
    total_loss = torch.tensor(0.0, device=device, requires_grad=True)
    valid_frames = 0

    for past_i in range(num_past):
        # Get ego transform for this past frame
        # t0_2_tn maps current cam → past cam. We need the ego translation component.
        T_slice = t0_2_tn[0, past_i * num_cams:(past_i + 1) * num_cams]  # [N, 4, 4]
        # Average ego translation across cameras (they share the same ego motion)
        avg_ego_translation = T_slice[:, :3, 3].mean(dim=0)  # [3]

        # Skip stationary frames
        if avg_ego_translation[:2].norm() < 0.3:
            continue

        # Build past-frame BEV: shift current Gaussians by ego translation
        # (This is a proxy — the actual past-frame Gaussians would come from
        #  running the decoder on past-frame features, which we don't do)
        means_past_proxy = means.clone()
        means_past_proxy[:, :2] = means_past_proxy[:, :2] - avg_ego_translation[:2]

        bev_past = scatter_to_bev(means_past_proxy, feats, dyn_weight, pc_range, bev_h, bev_w)

        # Compute similarity-based pseudo-flow
        flow_pseudo, flow_valid = compute_bev_similarity_flow(
            bev_current, bev_past, search_radius, pc_range, bev_h, bev_w)

        if flow_valid.sum() < 10:
            continue

        # Sample predicted motion at the BEV cell locations of valid Gaussians
        # First, get the predicted motion for this past frame
        pred_motion = motion_offsets[0, :, past_i * 2:(past_i + 1) * 2]  # [Q, 2]

        # Scatter predicted motion to BEV (weighted average)
        pred_bev_dx = scatter_to_bev(
            means, pred_motion[:, 0:1], dyn_weight, pc_range, bev_h, bev_w)  # [1, H, W]
        pred_bev_dy = scatter_to_bev(
            means, pred_motion[:, 1:2], dyn_weight, pc_range, bev_h, bev_w)  # [1, H, W]
        pred_bev_flow = torch.cat([pred_bev_dx, pred_bev_dy], dim=0)  # [2, H, W]

        # L1 loss on valid cells
        flow_diff = (pred_bev_flow - flow_pseudo).abs()  # [2, H, W]
        loss = (flow_diff * flow_valid.unsqueeze(0).float()).sum() / flow_valid.sum().clamp(min=1.0) / 2.0

        total_loss = total_loss + loss
        valid_frames += 1

    if valid_frames > 0:
        total_loss = total_loss / valid_frames

    return total_loss
