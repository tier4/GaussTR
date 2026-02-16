"""4D sampling for SparseBEV: projects 3D query points to multi-view multi-scale features."""

import torch
import torch.nn.functional as F

from .bbox_utils import decode_bbox
from .utils import get_rotation_matrix


def make_sample_points_from_bbox(query_bbox, offset, pc_range):
    """Generate 3D sample points from bbox center + learned offsets.

    Args:
        query_bbox: [B, Q, 10] encoded bbox (xyz normalized, wlh log, quat)
        offset: [B, Q, num_points, 3]
        pc_range: list of 6 floats

    Returns:
        sample_xyz: [B, Q, num_points, 3] in world coordinates
    """
    query_bbox = decode_bbox(query_bbox, pc_range)

    xyz = query_bbox[..., 0:3]  # [B, Q, 3]
    wlh = query_bbox[..., 3:6]  # [B, Q, 3]

    delta_xyz = offset[..., 0:3]  # [B, Q, P, 3]
    delta_xyz = wlh[:, :, None, :] * delta_xyz

    if query_bbox.shape[-1] > 6:
        quaternion_params = query_bbox[..., 6:10]
        rotation_matrices = get_rotation_matrix(quaternion_params)
        B, Q, P, _ = delta_xyz.shape
        delta_xyz_reshaped = delta_xyz.reshape(B * Q, P, 3).transpose(1, 2)
        rotation_matrices_reshaped = rotation_matrices.reshape(B * Q, 3, 3)
        rotated = torch.bmm(rotation_matrices_reshaped, delta_xyz_reshaped)
        delta_xyz = rotated.transpose(1, 2).reshape(B, Q, P, 3)

    sample_xyz = xyz[:, :, None, :] + delta_xyz
    return sample_xyz


def make_sample_points_from_3dgs(query_bbox, offset, anisotropy_info, pc_range):
    """Anisotropy-aware sampling: mu_sample = mu + R(r) * (s * mu_delta).

    Args:
        query_bbox: [B, Q, 10]
        offset: [B, Q, num_points, 3]
        anisotropy_info: dict with 'scale' [B, Q, 3] and 'rotation' [B, Q, 4]
        pc_range: list of 6 floats

    Returns:
        sample_xyz: [B, Q, num_points, 3]
    """
    if anisotropy_info is None:
        return make_sample_points_from_bbox(query_bbox, offset, pc_range)

    query_bbox = decode_bbox(query_bbox, pc_range)
    xyz = query_bbox[..., 0:3]

    xyz = xyz[:, :, None, :]
    wlh = anisotropy_info['scale'][:, :, None, :]

    delta_xyz = offset[..., 0:3]
    delta_xyz = wlh * delta_xyz

    quaternion_params = anisotropy_info['rotation']
    rotation_matrices = get_rotation_matrix(quaternion_params)

    delta_xyz_reshaped = delta_xyz.view(-1, delta_xyz.size(2), 3)
    rotation_matrices_reshaped = rotation_matrices.view(-1, 3, 3)
    rotated_delta = torch.bmm(delta_xyz_reshaped, rotation_matrices_reshaped)
    rotated_delta = rotated_delta.view(delta_xyz.size(0), delta_xyz.size(1), delta_xyz.size(2), 3)

    return xyz + rotated_delta


def msmv_sampling_pytorch(mlvl_feats, sampling_locations, scale_weights):
    """Multi-scale multi-view sampling via F.grid_sample (pure PyTorch fallback).

    Args:
        mlvl_feats: list of [B, N, H, W, C] tensors per level
        sampling_locations: [B, Q, P, 3] normalized coords (x, y, view)
        scale_weights: [B, Q, P, num_levels]

    Returns:
        [B, Q, C, P] sampled features
    """
    assert scale_weights.shape[-1] == len(mlvl_feats)

    B, _, _, _, C = mlvl_feats[0].shape
    _, Q, P, _ = sampling_locations.shape

    sampling_locations = sampling_locations * 2 - 1
    sampling_locations = sampling_locations[:, :, :, None, :]  # [B, Q, P, 1, 3]

    final = torch.zeros([B, C, Q, P], device=mlvl_feats[0].device)

    for lvl, feat in enumerate(mlvl_feats):
        feat = feat.permute(0, 4, 1, 2, 3)  # [B, C, N, H, W]
        out = F.grid_sample(
            feat, sampling_locations, mode='bilinear',
            padding_mode='zeros', align_corners=True,
        )[..., 0]  # [B, C, Q, P]
        out = out * scale_weights[..., lvl].reshape(B, 1, Q, P)
        final += out

    return final.permute(0, 2, 1, 3)  # [B, Q, C, P]


def sampling_4d(sample_points, mlvl_feats, scale_weights, ego2img, image_h, image_w,
                num_cams=5, eps=1e-5):
    """Project 3D sample points to multi-view images and sample features.

    Args:
        sample_points: [B, Q, T, G, P, 3] world-space points
        mlvl_feats: list of [BTG, N, H, W, C] feature tensors
        scale_weights: [B, Q, G, T, P, num_levels]
        ego2img: [B, T*N, 4, 4] or list
        image_h, image_w: image dimensions
        num_cams: number of cameras (default 5)

    Returns:
        [B, Q, G, T*P, C] sampled features
    """
    B, Q, T, G, P, _ = sample_points.shape
    N = num_cams

    sample_points = sample_points.reshape(B, Q, T, G * P, 3)

    ego2img = ego2img[:, :(T * N), None, None, :, :]
    ego2img = ego2img.expand(B, T * N, Q, G * P, 4, 4)
    ego2img = ego2img.reshape(B, T, N, Q, G * P, 4, 4)

    ones = torch.ones_like(sample_points[..., :1])
    sample_points = torch.cat([sample_points, ones], dim=-1)
    sample_points = sample_points[:, :, None, ..., None]
    sample_points = sample_points.expand(B, Q, N, T, G * P, 4, 1)
    sample_points = sample_points.transpose(1, 3)  # [B, T, N, Q, GP, 4, 1]

    sample_points_cam = torch.matmul(ego2img, sample_points).squeeze(-1)

    homo = sample_points_cam[..., 2:3]
    homo_nonzero = torch.maximum(homo, torch.zeros_like(homo) + eps)
    sample_points_cam = sample_points_cam[..., 0:2] / homo_nonzero

    sample_points_cam[..., 0] /= image_w
    sample_points_cam[..., 1] /= image_h

    valid_mask = (
        (homo > eps)
        & (sample_points_cam[..., 1:2] > 0.0)
        & (sample_points_cam[..., 1:2] < 1.0)
        & (sample_points_cam[..., 0:1] > 0.0)
        & (sample_points_cam[..., 0:1] < 1.0)
    ).squeeze(-1).float()

    valid_mask = valid_mask.permute(0, 1, 3, 4, 2)
    sample_points_cam = sample_points_cam.permute(0, 1, 3, 4, 2, 5)

    i_batch = torch.arange(B, dtype=torch.long, device=sample_points.device)
    i_query = torch.arange(Q, dtype=torch.long, device=sample_points.device)
    i_time = torch.arange(T, dtype=torch.long, device=sample_points.device)
    i_point = torch.arange(G * P, dtype=torch.long, device=sample_points.device)
    i_batch = i_batch.view(B, 1, 1, 1, 1).expand(B, T, Q, G * P, 1)
    i_time = i_time.view(1, T, 1, 1, 1).expand(B, T, Q, G * P, 1)
    i_query = i_query.view(1, 1, Q, 1, 1).expand(B, T, Q, G * P, 1)
    i_point = i_point.view(1, 1, 1, G * P, 1).expand(B, T, Q, G * P, 1)
    i_view = torch.argmax(valid_mask, dim=-1)[..., None]

    sample_points_cam = sample_points_cam[i_batch, i_time, i_query, i_point, i_view, :]
    valid_mask = valid_mask[i_batch, i_time, i_query, i_point, i_view]

    # Normalize view index to [0, 1]
    view_norm = N - 1 if N > 1 else 1
    sample_points_cam = torch.cat([sample_points_cam, i_view[..., None].float() / view_norm], dim=-1)
    sample_points_cam = sample_points_cam.reshape(B, T, Q, G, P, 1, 3)
    sample_points_cam = sample_points_cam.permute(0, 1, 3, 2, 4, 5, 6)
    sample_points_cam = sample_points_cam.reshape(B * T * G, Q, P, 3)

    # Don't expand by G for sampling — handle groups via channel split after
    scale_weights_flat = scale_weights.reshape(B, Q, G, T, P, -1)
    # Average scale weights across groups for sampling (groups are channel-split, not spatial)
    scale_weights_avg = scale_weights_flat.mean(dim=2)  # [B, Q, T, P, L]
    scale_weights_avg = scale_weights_avg.permute(0, 2, 1, 3, 4)  # [B, T, Q, P, L]
    scale_weights_avg = scale_weights_avg.reshape(B * T, Q, P, -1)

    # Reformat sample_points: [B, T, Q, G, P, 1, 3] -> [B*T, Q, P, 3]
    # Need to re-derive from the sample_points_cam which is [B*T*G, Q, P, 3]
    # Since all G groups sample same points, take every G-th element
    sample_points_flat = sample_points_cam.reshape(B, T, G, Q, P, 3)[:, :, 0]  # [B, T, Q, P, 3]
    sample_points_flat = sample_points_flat.reshape(B * T, Q, P, 3)

    # Reformat mlvl_feats from [B, T*N, C, H, W] to [B*T, N, H, W, C]
    reformatted_feats = []
    for feat in mlvl_feats:
        f = feat.reshape(B, T, N, *feat.shape[2:])  # [B, T, N, C, H, W]
        f = f.reshape(B * T, N, *feat.shape[2:])  # [B*T, N, C, H, W]
        f = f.permute(0, 1, 3, 4, 2)  # [B*T, N, H, W, C]
        reformatted_feats.append(f)

    final = msmv_sampling_pytorch(reformatted_feats, sample_points_flat, scale_weights_avg)
    # final: [B*T, Q, C, P]

    C = final.shape[2]
    Cg = C // G
    final = final.reshape(B, T, Q, C, P)
    # Split channels into groups: [B, T, Q, G, C//G, P]
    final = final.reshape(B, T, Q, G, Cg, P)
    # Rearrange to [B, Q, G, T*P, C//G]
    final = final.permute(0, 2, 3, 1, 5, 4)  # [B, Q, G, T, P, Cg]
    final = final.flatten(3, 4)  # [B, Q, G, T*P, Cg]

    return final
