"""GaussTR Head for Gaussian parameter prediction and rendering.

Pure PyTorch implementation without MMEngine dependencies.
"""

from typing import Dict, Any, Optional, Tuple, List
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

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
        image_shape: Input image shape (height, width) for features.
        render_image_size: Original image size (height, width) for Gaussian rasterization.
        patch_size: Patch size used for feature extraction.
        depth_limit: Maximum depth limit.
        text_protos: Path to text prototype embeddings (21-class). Optional.
        text_protos_sam3: Path to merged text embeddings aligned with SAM3 (17-class). Optional.
        prompt_denoising: Whether to use prompt denoising.
        num_segment_classes: Number of segmentation classes.
        voxelizer_cfg: Config for Gaussian voxelizer.
        text_loss_weight: Weight for text-guided contrastive loss. Default: 3.0.
        text_loss_temp: Temperature for text contrastive loss. Default: 0.1.
        cosine_loss_weight: Weight for visual feature cosine loss. Default: 2.0.
    """

    def __init__(
        self,
        embed_dims: int = 256,
        feat_dims: int = 512,
        reduce_dims: int = 128,
        image_shape: Tuple[int, int] = (432, 768),
        render_image_size: Tuple[int, int] = (900, 1600),
        patch_size: int = 16,
        depth_limit: float = 51.2,
        text_protos: Optional[str] = None,
        text_protos_sam3: Optional[str] = None,
        prompt_denoising: bool = True,
        num_segment_classes: int = 17,
        voxelizer_cfg: Optional[Dict[str, Any]] = None,
        text_loss_weight: float = 3.0,
        text_loss_temp: float = 0.1,
        cosine_loss_weight: float = 2.0,
        depth_loss_weight: float = 1.0,
        position_loss_weight: float = 1.0,
        edge_loss_weight: float = 0.5,
        pca_path: Optional[str] = None,
        density_thresh: float = 1e-3
    ):
        super().__init__()

        self.reduce_dims = reduce_dims
        self.density_thresh = density_thresh
        self.image_shape = image_shape
        self.render_image_size = render_image_size
        self.patch_size = patch_size
        self.depth_limit = depth_limit
        self.use_prompt_denoising = prompt_denoising

        # Loss weights and temperature
        self.text_loss_weight = text_loss_weight
        self.text_loss_temp = text_loss_temp
        self.cosine_loss_weight = cosine_loss_weight
        self.depth_loss_weight = depth_loss_weight
        self.position_loss_weight = position_loss_weight
        self.edge_loss_weight = edge_loss_weight

        # Sobel kernels for edge detection (registered as buffers for device compatibility)
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        self.register_buffer('sobel_x', sobel_x)
        self.register_buffer('sobel_y', sobel_y)

        # Prediction heads
        self.opacity_head = MLP(embed_dims, output_dim=1, mode='sigmoid')
        self.feature_head = MLP(embed_dims, output_dim=feat_dims)
        self.scale_head = MLP(embed_dims, output_dim=3, mode='sigmoid', range=(1.0, 16.0))
        self.regress_head = MLP(embed_dims, output_dim=3)
        self.class_head = MLP(reduce_dims, output_dim=num_segment_classes)

        # Load text prototypes for inference (21-class)
        if text_protos is not None:
            text_proto = torch.load(text_protos, map_location='cpu', weights_only=True)
            # Handle both formats: (feat_dims, num_classes) or (num_classes, feat_dims)
            if text_proto.shape[0] < text_proto.shape[1]:
                text_proto = text_proto.T
            self.register_buffer('text_proto_embeds', text_proto)
        else:
            self.text_proto_embeds = None

        # Load merged text prototypes for SAM3-aligned training (17-class)
        if text_protos_sam3 is not None:
            text_proto_sam3 = torch.load(text_protos_sam3, map_location='cpu', weights_only=True)
            # Handle both formats: (feat_dims, num_classes) or (num_classes, feat_dims)
            if text_proto_sam3.shape[0] < text_proto_sam3.shape[1]:
                text_proto_sam3 = text_proto_sam3.T
            self.register_buffer('text_proto_embeds_sam3', text_proto_sam3)
        else:
            self.text_proto_embeds_sam3 = None

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

        # PCA for dimensionality reduction
        # Option 1: Load pre-computed PCA (recommended for single-view datasets like T4)
        # Option 2: Compute per-batch PCA with EMA smoothing (works for multi-view)
        self.pca_ema_momentum = 0.1
        self.register_buffer('pca_v', torch.zeros(feat_dims, reduce_dims))
        self.register_buffer('pca_initialized', torch.tensor(0, dtype=torch.long))

        if pca_path is not None and pca_path:
            pca_data = torch.load(pca_path, map_location='cpu', weights_only=True)
            pca_v = pca_data['v']  # [feat_dims, reduce_dims]
            # Verify dimensions match
            if pca_v.shape[0] != feat_dims or pca_v.shape[1] != reduce_dims:
                raise ValueError(
                    f"PCA dimensions mismatch: expected ({feat_dims}, {reduce_dims}), "
                    f"got {pca_v.shape}. Re-run precompute_pca.py with --reduce_dims {reduce_dims}"
                )
            self.pca_v.copy_(pca_v)
            self.pca_initialized.fill_(1)
            self.use_fixed_pca = True
            print(f"[GaussTRHead] Loaded pre-computed PCA from {pca_path} "
                  f"(variance explained: {pca_data.get('variance_explained', 0)*100:.1f}%)")
        else:
            self.use_fixed_pca = False

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
        layer_idx: Optional[int] = None,
        debug_step: bool = False,
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

        # Reshape ref_pts to [B, N, Q, 2] before any modification
        ref_pts_reshaped = ref_pts.reshape(tuple(x.shape[:-1]) + (-1,))

        # Predict Gaussian position deltas (no constraints on z-delta)
        deltas = self.regress_head(x)
        ref_pts = (deltas[..., :2] + inverse_sigmoid(ref_pts_reshaped)).sigmoid()

        # Sample depth at reference points
        # depth shape: [B, N, H, W] - same as original implementation
        # Squeeze if [B, N, 1, H, W] to match original 4D format
        if depth.dim() == 5:
            depth = depth.squeeze(2)  # [B, N, 1, H, W] -> [B, N, H, W]

        # Clamp depth for geometry/sampling (avoids extreme Gaussian positions)
        depth = depth.clamp(max=self.depth_limit)

        # Add channel dim temporarily for grid_sample (original: depth[:, :n, None])
        sample_depth = flatten_bsn_forward(
            F.grid_sample, depth[:, :n, None],
            ref_pts.unsqueeze(2) * 2 - 1,
            mode='bilinear', align_corners=False)
        sample_depth = sample_depth[:, :, 0, 0, :, None]

        # Compute 3D points from 2D reference points + depth
        # Clamp adjusted depth to [0.1, depth_limit] to prevent log(0) in SiLogLoss
        adjusted_depth = (sample_depth * (1 + deltas[..., 2:3])).clamp(min=0.1, max=self.depth_limit)
        points = torch.cat([
            ref_pts * self.image_shape_tensor,
            adjusted_depth
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
            preds = torch.where(density.squeeze(-1) > self.density_thresh, preds, 17)

            return preds

        # Training mode: render and compute losses
        tgt_feats = feats.flatten(-2).mT.float()  # [B*N, H*W, C]

        # PCA for dimensionality reduction
        with torch.amp.autocast('cuda', enabled=False):
            if self.use_fixed_pca:
                # Use pre-computed fixed PCA (recommended for single-view datasets)
                v = self.pca_v
            else:
                # Compute per-batch PCA with EMA smoothing (for multi-view datasets)
                u, s, v = torch.pca_lowrank(
                    tgt_feats.flatten(0, 2), q=self.reduce_dims, niter=4)

                # EMA update for PCA to stabilize training
                if self.training:
                    if self.pca_initialized.item() == 0:
                        self.pca_v.copy_(v)
                        self.pca_initialized.fill_(1)
                    else:
                        # Align signs to handle PCA sign ambiguity
                        sign = torch.sign((self.pca_v * v).sum(dim=0, keepdim=True))
                        sign = torch.where(sign == 0, torch.ones_like(sign), sign)
                        v_aligned = v * sign
                        # EMA: 90% old + 10% new
                        self.pca_v.mul_(1 - self.pca_ema_momentum).add_(
                            v_aligned * self.pca_ema_momentum)

                    # Sync pca_v across GPUs in distributed training to prevent divergence
                    if dist.is_initialized():
                        dist.all_reduce(self.pca_v, op=dist.ReduceOp.AVG)

                # Use EMA-smoothed PCA
                v = self.pca_v

        tgt_feats = tgt_feats @ v
        features = features @ v

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
            image_size=self.render_image_size,
            near_plane=0.1,
            far_plane=100,
            render_mode='RGB+ED',
            channel_chunk=32).flatten(0, 1)

        rendered_depth = rendered[:, -1]
        rendered = rendered[:, :-1]

        losses = {}

        # Add monitoring metrics (detached, won't affect gradients)
        losses['dz_mean'] = deltas[..., 2].detach().mean()
        losses['dz_max'] = deltas[..., 2].detach().abs().max()
        losses['opacity'] = opacities.detach().mean()
        losses['rd_mean'] = rendered_depth.detach().mean()

        # Position loss: directly supervise Gaussian depth positions
        # This prevents opacity cheating when using RGB+ED mode
        # adjusted_depth: [B, N, Q, 1] - Gaussian's actual depth = sample_depth * (1 + delta_z)
        # sample_depth: [B, N, Q, 1] - GT depth at reference points
        valid_position = sample_depth > 0
        if valid_position.sum() > 0:
            losses['loss_position'] = F.l1_loss(
                adjusted_depth[valid_position],
                sample_depth[valid_position]
            ) * self.position_loss_weight
        else:
            losses['loss_position'] = torch.tensor(0.0, device=adjusted_depth.device)

        # Depth loss - depth is [B, N, H, W], flatten to [B*N, H, W] to match rendered_depth
        depth_for_loss = depth.flatten(0, 1)

        # Resize depth to match rendered_depth shape if needed
        if depth_for_loss.shape[-2:] != rendered_depth.shape[-2:]:
            depth_for_loss = F.interpolate(
                depth_for_loss.unsqueeze(1),
                size=rendered_depth.shape[-2:], mode='nearest'
            ).squeeze(1)

        # Create sky mask to exclude sky pixels (PriorDA has sky depth but it's not reliable)
        # SAM3 class 17 = sky
        sky_mask = None
        if sem_segs is not None:
            sky_mask = F.interpolate(
                sem_segs.flatten(0, 1).unsqueeze(1).float(),
                size=rendered_depth.shape[-2:], mode='nearest'
            ).squeeze(1).long()
            sky_mask = (sky_mask != 17)  # True for non-sky pixels

        # Compute depth loss with sky mask (edge-aware enabled for training loss)
        losses['loss_depth'] = self.depth_loss(rendered_depth, depth_for_loss, sky_mask=sky_mask, edge_aware=True)
        # MAE for monitoring only (no edge-aware, just raw L1)
        losses['mae_depth'] = self.depth_loss(rendered_depth, depth_for_loss, criterion='l1', sky_mask=sky_mask, edge_aware=False)

        # Feature loss
        bsn, c, h, w = rendered.shape
        feat_h = feats.shape[-2]
        feat_w = feats.shape[-1]
        tgt_feats = tgt_feats.mT.reshape(bsn, c, feat_h, feat_w)
        tgt_feats = F.interpolate(
            tgt_feats, size=(h, w), mode='bilinear', align_corners=False)

        # Keep rendered in 2D form for text loss before flattening
        rendered_2d = rendered  # [B*N, C, H, W]

        rendered = rendered.flatten(2).mT
        tgt_feats = tgt_feats.flatten(2).mT.flatten(0, 1)

        # Cosine loss for feature alignment (original weight: 5)
        losses['loss_cosine'] = F.cosine_embedding_loss(
            rendered.flatten(0, 1), tgt_feats,
            torch.ones_like(tgt_feats[:, 0])) * 5

        # Segmentation loss
        if sem_segs is not None:
            # Resize sem_segs to match rendered size
            sem_segs_resized = F.interpolate(
                sem_segs.flatten(0, 1).unsqueeze(1).float(),
                size=(h, w), mode='nearest'
            ).squeeze(1).long()
            # Map sky (class 17) to 0, which will be ignored by ignore_index=0
            sem_segs_clamped = torch.where(
                sem_segs_resized == 17,
                torch.zeros_like(sem_segs_resized),
                sem_segs_resized
            )
            losses['loss_ce'] = F.cross_entropy(
                self.class_head(rendered).mT,
                sem_segs_clamped.flatten(1).long(),
                ignore_index=0)

            # Text-guided contrastive loss (additional, uses SAM3-aligned text embeddings)
            if self.text_proto_embeds_sam3 is not None:
                # Project text embeddings through same PCA as features
                text_embeds_pca = self.text_proto_embeds_sam3.T @ v  # [17, reduce_dims]
                text_embeds_pca = text_embeds_pca.T  # [reduce_dims, 17]
                losses['loss_text'] = self.text_contrastive_loss(
                    rendered_2d, sem_segs_resized, text_embeds_pca
                ) * self.text_loss_weight

        return losses

    def compute_depth_edges(self, depth: torch.Tensor) -> torch.Tensor:
        """Compute depth edge weights using Sobel operators.

        Args:
            depth: Depth tensor [B, H, W] or [B, 1, H, W]

        Returns:
            Edge weights [B, H, W] normalized to [0, 1], higher at depth discontinuities
        """
        # Ensure 4D input for conv2d
        if depth.dim() == 3:
            depth = depth.unsqueeze(1)  # [B, 1, H, W]

        # Normalize depth to [0, 1] for stable gradient computation
        depth_min = depth.amin(dim=(2, 3), keepdim=True)
        depth_max = depth.amax(dim=(2, 3), keepdim=True)
        depth_norm = (depth - depth_min) / (depth_max - depth_min + 1e-6)

        # Compute gradients using Sobel operators
        grad_x = F.conv2d(depth_norm, self.sobel_x, padding=1)
        grad_y = F.conv2d(depth_norm, self.sobel_y, padding=1)

        # Edge magnitude
        edge_mag = torch.sqrt(grad_x ** 2 + grad_y ** 2 + 1e-6)

        # Normalize edge magnitude to [0, 1] per sample
        edge_min = edge_mag.amin(dim=(2, 3), keepdim=True)
        edge_max = edge_mag.amax(dim=(2, 3), keepdim=True)
        edge_weights = (edge_mag - edge_min) / (edge_max - edge_min + 1e-6)

        return edge_weights.squeeze(1)  # [B, H, W]

    def depth_loss(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        criterion: str = 'silog_l1',
        sky_mask: Optional[torch.Tensor] = None,
        edge_aware: bool = True
    ) -> torch.Tensor:
        """Compute depth loss with optional sky mask and edge-aware weighting.

        Args:
            pred: Predicted depth [B, H, W]
            target: Target depth [B, H, W]
            criterion: Loss criterion ('silog', 'l1', or 'silog_l1')
            sky_mask: Boolean mask [B, H, W], True for valid (non-sky) pixels
            edge_aware: Whether to add edge-aware loss component

        Returns:
            Combined depth loss
        """
        # Compute edge-aware loss BEFORE applying sky mask (need 2D structure)
        edge_loss = torch.tensor(0.0, device=pred.device)
        if edge_aware and self.edge_loss_weight > 0:
            # Compute edge weights from target depth
            edge_weights = self.compute_depth_edges(target)  # [B, H, W]

            # Compute per-pixel L1 error
            pixel_error = (pred - target).abs()

            # Apply sky mask if provided
            if sky_mask is not None:
                edge_weights = edge_weights * sky_mask.float()
                pixel_error = pixel_error * sky_mask.float()
                valid_count = sky_mask.float().sum().clamp(min=1)
            else:
                valid_mask = target > 0
                edge_weights = edge_weights * valid_mask.float()
                pixel_error = pixel_error * valid_mask.float()
                valid_count = valid_mask.float().sum().clamp(min=1)

            # Edge-weighted loss: higher weight at depth discontinuities
            # Add base weight of 1.0 so all pixels contribute, edges contribute more
            weighted_error = pixel_error * (1.0 + edge_weights)
            edge_loss = weighted_error.sum() / valid_count

        # Apply sky mask for standard losses
        if sky_mask is not None:
            pred = pred[sky_mask]
            target = target[sky_mask]

        loss = 0
        if 'silog' in criterion:
            loss += self.silog_loss(pred, target)
        if 'l1' in criterion:
            target_flat = target.flatten()
            valid = target_flat > 0
            if valid.sum() > 0:
                l1_loss = F.l1_loss(pred.flatten()[valid], target_flat[valid])
                if loss != 0:
                    l1_loss *= 0.2
                loss += l1_loss

        # Add edge-aware component
        if edge_aware and self.edge_loss_weight > 0:
            loss = loss + self.edge_loss_weight * edge_loss

        return loss

    def scale_transform(
        self,
        depth: torch.Tensor,
        focal: torch.Tensor,
        multiplier: float = 7.5
    ) -> torch.Tensor:
        """Transform scale based on depth and focal length."""
        return depth * multiplier / focal.reshape(tuple(depth.shape[:2]) + (1, 1))

    def text_contrastive_loss(
        self,
        pred_feats: torch.Tensor,
        sam3_labels: torch.Tensor,
        text_embeds: torch.Tensor
    ) -> torch.Tensor:
        """Compute pure contrastive loss using cross-entropy over text embeddings.

        Args:
            pred_feats: Predicted features [B, C, H, W] in PCA-reduced space
            sam3_labels: SAM3 class labels [B, H, W] with values 0-17
            text_embeds: Text embeddings [C, 17] aligned with SAM3 classes 1-17

        Returns:
            Contrastive loss. Class 0 (background) and class 12 (other_flat) are ignored.
        """
        B, C, H, W = pred_feats.shape

        # Normalize features and text embeddings
        # Note: eps=1e-6 prevents NaN for zero-norm embeddings (e.g., class 12 "other_flat")
        pred_norm = F.normalize(pred_feats, dim=1, eps=1e-6)  # [B, C, H, W]
        text_norm = F.normalize(text_embeds, dim=0, eps=1e-6)  # [C, 17]

        # Compute cosine similarity to all 17 classes
        # [B, C, H, W] @ [C, 17] -> [B, 17, H, W]
        cos_sim = torch.einsum('bchw,ck->bkhw', pred_norm, text_norm)

        # Scale by temperature to get logits
        logits = cos_sim / self.text_loss_temp  # [B, 17, H, W]

        # Valid mask: exclude background (0), other_flat (12, no text embed), sky (17, trivial)
        valid_mask = (sam3_labels >= 1) & (sam3_labels <= 16) & (sam3_labels != 12)

        # Shift SAM3 labels to 0-indexed: SAM3 1-17 -> 0-16
        sam3_shifted = (sam3_labels - 1).clamp(min=0)  # [B, H, W], values 0-16

        # Cross-entropy contrastive loss
        ce_loss = F.cross_entropy(
            logits,
            sam3_shifted.long(),
            reduction='none'
        )  # [B, H, W]

        # Apply valid mask and compute mean
        num_valid = valid_mask.float().sum().clamp(min=1)
        loss = (ce_loss * valid_mask.float()).sum() / num_valid

        return loss

    # === COMMENTED OUT: Previous method with cosine + contrastive split ===
    # def text_contrastive_loss_v2(
    #     self,
    #     pred_feats: torch.Tensor,
    #     sam3_labels: torch.Tensor,
    #     text_embeds: torch.Tensor
    # ) -> torch.Tensor:
    #     """Compute text-guided loss with cosine and contrastive components.
    #
    #     For pixels where CLIP prediction matches SAM3 label: cosine loss (reinforce)
    #     For pixels where CLIP prediction differs from SAM3 label: contrastive loss (correct)
    #     """
    #     B, C, H, W = pred_feats.shape
    #     pred_norm = F.normalize(pred_feats, dim=1, eps=1e-6)
    #     text_norm = F.normalize(text_embeds, dim=0, eps=1e-6)
    #     cos_sim = torch.einsum('bchw,ck->bkhw', pred_norm, text_norm)
    #     clip_pred = cos_sim.argmax(dim=1)
    #     valid_mask = (sam3_labels >= 1) & (sam3_labels <= 16) & (sam3_labels != 12)
    #     sam3_shifted = (sam3_labels - 1).clamp(min=0)
    #     match_mask = valid_mask & (clip_pred == sam3_shifted)
    #     nonmatch_mask = valid_mask & (clip_pred != sam3_shifted)
    #     target_embeds = text_norm[:, sam3_shifted].permute(1, 0, 2, 3)
    #     cos_sim_target = (pred_norm * target_embeds).sum(dim=1)
    #     cosine_loss = 1.0 - cos_sim_target
    #     logits = cos_sim / self.text_loss_temp
    #     ce_loss = F.cross_entropy(logits, sam3_shifted.long(), reduction='none') * 0.5
    #     num_match = match_mask.float().sum().clamp(min=1)
    #     num_nonmatch = nonmatch_mask.float().sum().clamp(min=1)
    #     loss_match = (cosine_loss * match_mask.float()).sum() / num_match
    #     loss_nonmatch = (ce_loss * nonmatch_mask.float()).sum() / num_nonmatch
    #     return loss_match + loss_nonmatch
