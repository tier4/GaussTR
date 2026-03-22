"""Flow-supervised motion head loss.

Uses pre-computed OV flow pseudo-labels (from DINOv3CLIP feature matching)
to supervise the motion head's per-query XY offset predictions.

The flow is in image-space feature coordinates. To supervise the motion head
(which predicts in ego-XY meters), we need to:
1. Project each Gaussian's 3D position to 2D camera coordinates
2. Sample the pre-computed flow at those 2D coordinates
3. Back-project the 2D flow to 3D ego-XY using depth
4. Compare with the motion head's prediction
"""

import torch
import torch.nn.functional as F


def compute_flow_supervision_loss(
    motion_offsets: torch.Tensor,     # [B, Q, 2*P] predicted XY motion
    means3d: torch.Tensor,            # [B, Q, 3] Gaussian positions in ego frame
    branch_probs: torch.Tensor,       # [B, Q, 2] static/dynamic probs
    flow_maps: torch.Tensor,          # [B, P*N, 2, Hf, Wf] pre-computed OV flow
    W2C: torch.Tensor,                # [N, 4, 4] world-to-camera transforms
    K: torch.Tensor,                  # [N, 4, 4] camera intrinsics
    pc_range: list,
    num_cams: int = 5,
    render_h: int = 180,
    render_w: int = 320,
    feat_h: int = 56,
    feat_w: int = 87,
) -> torch.Tensor:
    """Compute motion head loss using pre-computed OV flow pseudo-labels.

    For each dynamic Gaussian:
    1. Project its 3D position to each camera's 2D coordinates
    2. Sample the pre-computed flow at that 2D location
    3. Scale flow from feature-space pixels to meters using depth
    4. L1 loss between predicted motion and flow-derived motion

    Args:
        motion_offsets: [B, Q, 2*P] predicted XY offsets per past frame
        means3d: [B, Q, 3] Gaussian positions
        branch_probs: [B, Q, 2] static/dynamic
        flow_maps: [B, P*N, 2, Hf, Wf] OV flow (dx, dy in feature pixels)
        W2C: [N, 4, 4] world-to-camera
        K: [N, 4, 4] intrinsics
        pc_range: scene bounds

    Returns:
        Scalar flow supervision loss
    """
    B = means3d.shape[0]
    Q = means3d.shape[1]
    device = means3d.device

    # Work with B=1
    means = means3d[0]  # [Q, 3]
    bp = branch_probs[0]  # [Q, 2]
    p_dyn = bp[:, 1]  # [Q]

    # Only supervise dynamic Gaussians (p_dyn > 0.3)
    dyn_mask = p_dyn > 0.3
    if dyn_mask.sum() < 10:
        return torch.tensor(0.0, device=device, requires_grad=True)

    dyn_means = means[dyn_mask]  # [D, 3]
    dyn_pred = motion_offsets[0, dyn_mask]  # [D, 2*P]
    D = dyn_means.shape[0]

    # Project dynamic Gaussians to all cameras
    # means3d → homogeneous [D, 4]
    ones = torch.ones(D, 1, device=device)
    means_homo = torch.cat([dyn_means, ones], dim=1)  # [D, 4]

    num_past = flow_maps.shape[1] // num_cams
    total_loss = torch.tensor(0.0, device=device, requires_grad=True)
    valid_count = 0

    for past_i in range(num_past):
        pred_motion_xy = dyn_pred[:, past_i * 2:(past_i + 1) * 2]  # [D, 2]

        for cam_i in range(num_cams):
            # Project to camera: [D, 4] @ [4, 4]^T → [D, 4]
            cam_pts = (W2C[cam_i] @ means_homo.T).T  # [D, 4]
            z = cam_pts[:, 2].clamp(min=0.1)

            # Camera intrinsics: project to 2D pixel coords
            fx = K[cam_i, 0, 0]
            fy = K[cam_i, 1, 1]
            cx = K[cam_i, 0, 2]
            cy = K[cam_i, 1, 2]

            px = cam_pts[:, 0] / z * fx + cx  # [D] pixel x
            py = cam_pts[:, 1] / z * fy + cy  # [D] pixel y

            # Scale from render coords to feature coords
            scale_x = feat_w / render_w
            scale_y = feat_h / render_h
            fx_feat = px * scale_x
            fy_feat = py * scale_y

            # Normalize to [-1, 1] for grid_sample
            nx = (fx_feat / (feat_w - 1)) * 2 - 1
            ny = (fy_feat / (feat_h - 1)) * 2 - 1

            # Validity check: in-frame
            valid = (nx > -1) & (nx < 1) & (ny > -1) & (ny < 1) & (z > 0.5)
            if valid.sum() < 5:
                continue

            # Sample flow at projected locations
            flow_map = flow_maps[0, past_i * num_cams + cam_i]  # [2, Hf, Wf]
            grid = torch.stack([nx[valid], ny[valid]], dim=1).unsqueeze(0).unsqueeze(0)  # [1, 1, V, 2]
            sampled_flow = F.grid_sample(
                flow_map.unsqueeze(0), grid,  # [1, 2, 1, V]
                mode='bilinear', align_corners=True
            ).squeeze(0).squeeze(1)  # [2, V]
            sampled_flow = sampled_flow.T  # [V, 2] — flow in feature pixels

            # Convert flow from feature pixels to meters
            # flow_pixel_x * (meter_per_pixel_x) ≈ flow_pixel_x * z / fx * (render_w / feat_w)
            depth_at_gauss = z[valid]
            flow_meters_x = sampled_flow[:, 0] * depth_at_gauss / (fx * scale_x)
            flow_meters_y = sampled_flow[:, 1] * depth_at_gauss / (fy * scale_y)
            flow_meters = torch.stack([flow_meters_x, flow_meters_y], dim=1)  # [V, 2]

            # L1 loss between predicted and pseudo-flow
            pred_valid = pred_motion_xy[valid]  # [V, 2]
            loss = (pred_valid - flow_meters.detach()).abs().mean()

            total_loss = total_loss + loss
            valid_count += 1

    if valid_count > 0:
        total_loss = total_loss / valid_count

    return total_loss
