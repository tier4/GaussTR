import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from gsplat import rasterization


def batch_splatting_render(pc, w2c, Ks, render_conf, inference=False):
    """Render Gaussians to depth and OV feature maps using gsplat.

    Args:
        pc: GaussianPrediction with batch dim 1
        w2c: World-to-camera matrices [N, 4, 4]
        Ks: Camera intrinsics [N, 3, 3] or [N, 4, 4]
        render_conf: Dict with 'render_w' and 'render_h'
        inference: If True, only render depth (skip OV)

    Returns:
        Dict with 'depth', optionally 'ov_feature', 'alphas', 'meta'
    """
    if pc.means.dim() == 2:
        # Already unbatched [Q, 3]
        means = pc.means.float()
        quats = pc.rotations.float()
        scales = pc.scales.float()
        opacities = pc.opacities.float()
    else:
        assert pc.means.shape[0] == 1, f"Expected batch=1, got {pc.means.shape}"
        means = pc.means.squeeze(0).float()
        quats = pc.rotations.squeeze(0).float()
        scales = pc.scales.squeeze(0).float()
        opacities = pc.opacities.squeeze(0).float()

    Ks = Ks[:, :3, :3]
    width, height = render_conf['render_w'], render_conf['render_h']

    if pc.ovs is not None:
        semantics = pc.ovs.squeeze(0).float() if pc.ovs.dim() > 2 else pc.ovs.float()
    else:
        semantics = torch.zeros_like(means)

    if inference:
        render_results, alphas, meta = rasterization(
            means, quats, scales, opacities, semantics, w2c, Ks,
            width, height, packed=False, sparse_grad=False, render_mode="ED",
        )
        return {"depth": render_results[..., -1:], "alphas": alphas}

    render_results, alphas, meta = rasterization(
        means, quats, scales, opacities, semantics, w2c, Ks,
        width, height, packed=False, sparse_grad=False, render_mode="RGB+ED",
    )

    if pc.ovs is not None:
        ov_features = render_results[..., :-1]
    else:
        ov_features = None

    depth = render_results[..., -1:]

    return {
        "depth": depth,
        "ov_feature": ov_features,
        "alphas": alphas,
        "meta": meta,
    }


def prepare_gs_attribute(img_metas, num_cams=5):
    """Extract camera matrices from img_metas for gsplat rendering.

    Args:
        img_metas: List of dicts with 'cam2ego' and 'render_k'
        num_cams: Number of cameras

    Returns:
        render_k: Camera intrinsics [N, 4, 4]
        C2W: Camera-to-world [N, 4, 4]
        W2C: World-to-camera [N, 4, 4]
    """
    cam2ego = img_metas[0]['cam2ego']
    render_k = img_metas[0]['render_k']

    # Ensure tensors on CUDA
    if not isinstance(cam2ego, torch.Tensor):
        cam2ego = torch.tensor(np.array(cam2ego)).float()
    if not isinstance(render_k, torch.Tensor):
        render_k = torch.tensor(np.array(render_k)).float()

    C2W = cam2ego.float().cuda()  # [N, 4, 4]
    W2C = torch.inverse(C2W)
    render_k = render_k[0:num_cams].float().cuda()

    return render_k, C2W, W2C


class SiLogLoss(nn.Module):
    def __init__(self, variance_focus=0.85):
        super().__init__()
        self.variance_focus = variance_focus

    def forward(self, depth_est, depth_gt, mask):
        d = torch.log(depth_est[mask]) - torch.log(depth_gt[mask])
        return torch.sqrt((d ** 2).mean() - self.variance_focus * (d.mean() ** 2)) * 10.0


def get_depth_loss(depth_render, depth, mask):
    """Foundation depth loss: SiLog 15% + L1 85%.

    Args:
        depth_render: Rendered depth from Gaussians
        depth: Foundation model depth (PriorDA)
        mask: Valid depth mask

    Returns:
        Scalar loss
    """
    silog_criterion = SiLogLoss(variance_focus=0.85)

    silog_loss = silog_criterion(depth_render, depth, mask.to(torch.bool))
    l1_loss = F.l1_loss(depth_render[mask], depth[mask])

    return silog_loss * 0.15 + l1_loss * 0.85
