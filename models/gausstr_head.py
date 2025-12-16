"""GaussTR Head for Gaussian parameter prediction and rendering.

Pure PyTorch implementation without MMEngine dependencies.
"""

from typing import Dict, Any, Optional, Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import (
    cam2world, rotmat_to_quat, flatten_bsn_forward,
    OCC3D_CATEGORIES
)
from .gsplat_rasterization import rasterize_gaussians

# Use CUDA-accelerated voxelizer by default
from .cuda_voxelizer import CUDAVoxelizer


class MLP(nn.Module):
    """Multi-Layer Perceptron with optional activation and output range.

    Args:
        input_dim: Input dimension.
        hidden_dim: Hidden layer dimension. Defaults to input_dim * 4.
        output_dim: Output dimension. Defaults to input_dim.
        num_layers: Number of layers. Default: 2.
        activation: Activation function name. Default: 'relu'.
        mode: Output mode ('sigmoid' or None). Default: None.
        range: Output range tuple (min, max) when mode='sigmoid'. Default: None.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: Optional[int] = None,
        output_dim: Optional[int] = None,
        num_layers: int = 2,
        activation: str = 'relu',
        mode: Optional[str] = None,
        range: Optional[Tuple[float, float]] = None
    ):
        super().__init__()
        hidden_dim = hidden_dim or input_dim * 4
        output_dim = output_dim or input_dim
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))
        self.activation = activation
        self.range = range
        self.mode = mode

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for i, layer in enumerate(self.layers):
            x = getattr(F, self.activation)(
                layer(x)) if i < self.num_layers - 1 else layer(x)

        if self.mode is not None:
            if self.mode == 'sigmoid':
                x = torch.sigmoid(x)
            if self.range is not None:
                x = self.range[0] + (self.range[1] - self.range[0]) * x
        return x


def inverse_sigmoid(x: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    """Inverse function of sigmoid."""
    x = x.clamp(min=0, max=1)
    x1 = x.clamp(min=eps)
    x2 = (1 - x).clamp(min=eps)
    return torch.log(x1 / x2)


def prompt_denoising(
    logits: torch.Tensor,
    logit_scale: float = 100,
    pd_threshold: float = 0.1
) -> torch.Tensor:
    """Apply prompt denoising to class logits."""
    probs = logits.softmax(-1)
    probs_ = F.softmax(logits * logit_scale, -1)
    max_cls_conf = probs_.flatten(1, 3).max(1).values
    mask = (max_cls_conf < pd_threshold)[:, None, None, None]
    probs = torch.where(mask, torch.zeros_like(probs), probs)
    return probs


def merge_probs(probs: torch.Tensor, categories: Tuple) -> torch.Tensor:
    """Merge probabilities for categories with multiple names."""
    merged_probs = []
    i = 0
    for cats in categories:
        p = probs[..., i:i + len(cats)]
        i += len(cats)
        if len(cats) > 1:
            p = p.max(-1, keepdim=True).values
        merged_probs.append(p)
    return torch.cat(merged_probs, dim=-1)


class SiLogLoss(nn.Module):
    """Scale-invariant logarithmic loss for depth prediction."""

    def __init__(self, lambd: float = 0.5, eps: float = 1e-8):
        super().__init__()
        self.lambd = lambd
        self.eps = eps

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        valid = target > 0
        if valid.sum() == 0:
            return torch.tensor(0.0, device=pred.device)

        pred = pred[valid]
        target = target[valid]

        diff = torch.log(pred + self.eps) - torch.log(target + self.eps)
        loss = torch.sqrt((diff ** 2).mean() - self.lambd * (diff.mean() ** 2))
        return loss


class GaussTRHead(nn.Module):
    """GaussTR prediction head.

    Predicts Gaussian parameters (position, opacity, scale, features) from
    query embeddings and renders them to 2D feature maps.

    Args:
        embed_dims: Input embedding dimensions.
        feat_dims: Feature dimensions for Gaussian features.
        reduce_dims: Reduced dimensions for PCA.
        image_shape: Input image shape (height, width).
        patch_size: Patch size used for feature extraction.
        depth_limit: Maximum depth limit.
        text_protos: Path to text prototype embeddings. Optional.
        prompt_denoising: Whether to use prompt denoising.
        num_segment_classes: Number of segmentation classes.
        voxelizer_cfg: Config for Gaussian voxelizer.
    """

    def __init__(
        self,
        embed_dims: int = 256,
        feat_dims: int = 512,
        reduce_dims: int = 128,
        image_shape: Tuple[int, int] = (432, 768),
        patch_size: int = 16,
        depth_limit: float = 51.2,
        text_protos: Optional[str] = None,
        prompt_denoising: bool = True,
        num_segment_classes: int = 17,
        voxelizer_cfg: Optional[Dict[str, Any]] = None
    ):
        super().__init__()

        self.reduce_dims = reduce_dims
        self.image_shape = image_shape
        self.patch_size = patch_size
        self.depth_limit = depth_limit
        self.use_prompt_denoising = prompt_denoising

        # Prediction heads
        self.opacity_head = MLP(embed_dims, output_dim=1, mode='sigmoid')
        self.feature_head = MLP(embed_dims, output_dim=feat_dims)
        self.scale_head = MLP(embed_dims, output_dim=3, mode='sigmoid', range=(1.0, 16.0))
        self.regress_head = MLP(embed_dims, output_dim=3)
        self.class_head = MLP(reduce_dims, output_dim=num_segment_classes)

        # Load text prototypes if provided
        if text_protos is not None:
            text_proto = torch.load(text_protos, map_location='cpu')
            # Handle both formats: (feat_dims, num_classes) or (num_classes, feat_dims)
            if text_proto.shape[0] < text_proto.shape[1]:
                text_proto = text_proto.T
            self.register_buffer('text_proto_embeds', text_proto)
        else:
            self.text_proto_embeds = None

        # Voxelizer - config passed from yaml
        if voxelizer_cfg is None:
            voxelizer_cfg = {
                'vol_range': [-40, -40, -1, 40, 40, 5.4],
                'voxel_size': 0.4
            }
        self.voxelizer = CUDAVoxelizer(**voxelizer_cfg)

        # Cache image shape tensor (avoids tensor creation every forward pass)
        self.register_buffer(
            'image_shape_tensor',
            torch.tensor(image_shape[::-1], dtype=torch.float32)  # [W, H]
        )

        # Loss
        self.silog_loss = SiLogLoss()

    def forward(
        self,
        x: torch.Tensor,
        ref_pts: torch.Tensor,
        depth: torch.Tensor,
        cam2img: torch.Tensor,
        cam2ego: torch.Tensor,
        mode: str = 'tensor',
        feats: Optional[torch.Tensor] = None,
        img_aug_mat: Optional[torch.Tensor] = None,
        sem_segs: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """Forward pass.

        Args:
            x: Query embeddings [B, N, Q, embed_dims].
            ref_pts: Reference points [B, N, Q, 2].
            depth: Depth maps [B, N, 1, H, W].
            cam2img: Camera intrinsics [B, N, 4, 4].
            cam2ego: Camera extrinsics [B, N, 4, 4].
            mode: 'tensor', 'loss', or 'predict'.
            feats: Target features for supervision [B, N, C, H, W].
            img_aug_mat: Image augmentation matrix [B, N, 4, 4].
            sem_segs: Semantic segmentation labels [B, N, H, W].

        Returns:
            If mode='loss': Dictionary of losses.
            If mode='predict': Occupancy predictions.
            If mode='tensor': Gaussian parameters.
        """
        bs, n = cam2img.shape[:2]
        x = x.reshape((bs, n) + tuple(x.shape[1:]))

        # Predict Gaussian position deltas
        deltas = self.regress_head(x)
        ref_pts = (
            deltas[..., :2] +
            inverse_sigmoid(ref_pts.reshape(tuple(x.shape[:-1]) + (-1,)))).sigmoid()

        # Sample depth at reference points
        # depth shape: [B, N, H, W] - same as original implementation
        # Squeeze if [B, N, 1, H, W] to match original 4D format
        depth = depth.clamp(max=self.depth_limit)
        if depth.dim() == 5:
            depth = depth.squeeze(2)  # [B, N, 1, H, W] -> [B, N, H, W]
        # Add channel dim temporarily for grid_sample (original: depth[:, :n, None])
        sample_depth = flatten_bsn_forward(
            F.grid_sample, depth[:, :n, None],
            ref_pts.unsqueeze(2) * 2 - 1,
            mode='bilinear', align_corners=False)
        sample_depth = sample_depth[:, :, 0, 0, :, None]

        # Compute 3D points from 2D reference points + depth
        points = torch.cat([
            ref_pts * self.image_shape_tensor,
            sample_depth * (1 + deltas[..., 2:3])
        ], -1)
        means3d = cam2world(points, cam2img, cam2ego, img_aug_mat)

        # Predict Gaussian parameters
        opacities = self.opacity_head(x).float()
        features = self.feature_head(x).float()
        scales = self.scale_head(x) * self.scale_transform(
            sample_depth, cam2img[..., 0, 0]).clamp(1e-6)

        # Compute rotations from camera extrinsics
        rotations = flatten_bsn_forward(rotmat_to_quat, cam2ego[..., :3, :3])
        rotations = rotations.unsqueeze(2).expand(-1, -1, x.size(2), -1)

        # Inference mode: voxelize and predict occupancy
        if mode == 'predict':
            if self.text_proto_embeds is not None:
                features = features @ self.text_proto_embeds

            density, grid_feats = self.voxelizer(
                means3d=means3d.flatten(1, 2),
                opacities=opacities.flatten(1, 2),
                features=features.flatten(1, 2).softmax(-1),
                scales=scales.flatten(1, 2),
                rotations=rotations.flatten(1, 2))

            if self.use_prompt_denoising:
                probs = prompt_denoising(grid_feats)
            else:
                probs = grid_feats.softmax(-1)

            probs = merge_probs(probs, OCC3D_CATEGORIES)
            preds = probs.argmax(-1)
            preds += (preds > 10) * 1 + 1  # skip two classes of "others"
            preds = torch.where(density.squeeze(-1) > 4e-2, preds, 17)

            return preds

        # Training mode: render and compute losses
        tgt_feats = feats.flatten(-2).mT

        # PCA for dimensionality reduction (GPU, FP32)
        # Disable autocast - pca_lowrank doesn't support FP16
        with torch.amp.autocast('cuda', enabled=False):
            u, s, v = torch.pca_lowrank(
                tgt_feats.flatten(0, 2).float(), q=self.reduce_dims, niter=4)
        tgt_feats = tgt_feats.float() @ v
        features = features.float() @ v

        # Render Gaussians
        rendered = rasterize_gaussians(
            means3d.flatten(1, 2),
            features.flatten(1, 2),
            opacities.squeeze(-1).flatten(1, 2),
            scales.flatten(1, 2),
            rotations.flatten(1, 2),
            cam2img,
            cam2ego,
            img_aug_mats=img_aug_mat,
            image_size=(900, 1600),
            near_plane=0.1,
            far_plane=100,
            render_mode='RGB+D',
            channel_chunk=32).flatten(0, 1)

        rendered_depth = rendered[:, -1]
        rendered = rendered[:, :-1]

        losses = {}

        # Depth loss - depth is [B, N, H, W], flatten to [B*N, H, W] to match rendered_depth
        depth = torch.where(depth < self.depth_limit, depth,
                            1e-3).flatten(0, 1)
        losses['loss_depth'] = self.depth_loss(rendered_depth, depth)
        losses['mae_depth'] = self.depth_loss(
            rendered_depth, depth, criterion='l1')

        # Feature loss
        bsn, c, h, w = rendered.shape
        feat_h = self.image_shape[0] // self.patch_size
        feat_w = self.image_shape[1] // self.patch_size
        tgt_feats = tgt_feats.mT.reshape(bsn, c, feat_h, feat_w)
        tgt_feats = F.interpolate(
            tgt_feats, size=(h, w), mode='bilinear', align_corners=False)
        rendered = rendered.flatten(2).mT
        tgt_feats = tgt_feats.flatten(2).mT.flatten(0, 1)
        losses['loss_cosine'] = F.cosine_embedding_loss(
            rendered.flatten(0, 1), tgt_feats,
            torch.ones_like(tgt_feats[:, 0])) * 5

        # Segmentation loss
        if sem_segs is not None:
            losses['loss_ce'] = F.cross_entropy(
                self.class_head(rendered).mT,
                sem_segs.flatten(0, 1).flatten(1).long(),
                ignore_index=0)

        return losses

    def depth_loss(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        criterion: str = 'silog_l1'
    ) -> torch.Tensor:
        """Compute depth loss."""
        loss = 0
        if 'silog' in criterion:
            loss += self.silog_loss(pred, target)
        if 'l1' in criterion:
            target = target.flatten()
            pred = pred.flatten()[target != 0]
            l1_loss = F.l1_loss(pred, target[target != 0])
            if loss != 0:
                l1_loss *= 0.2
            loss += l1_loss
        return loss

    def scale_transform(
        self,
        depth: torch.Tensor,
        focal: torch.Tensor,
        multiplier: float = 7.5
    ) -> torch.Tensor:
        """Transform scale based on depth and focal length."""
        return depth * multiplier / focal.reshape(tuple(depth.shape[:2]) + (1, 1))
