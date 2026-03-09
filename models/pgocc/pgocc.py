"""PG-Occ Lightning Module — Progressive Gaussian Occupancy prediction.

Combines ResNet50+FPN backbone with SparseGaussiansDecoder for self-supervised
3D occupancy prediction from multi-view cameras using Gaussian splatting.
"""

import json
import math
import os

import cv2
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50

from .fpn import FPN
from .gaussian_prediction import GaussianPrediction
from .sparse_gaussians_decoder import SparseGaussiansDecoder
from .render import batch_splatting_render, prepare_gs_attribute, get_depth_loss, get_gt_loss
from .loss_utils import BackprojectDepth, Project3D, calc_time_warping_loss
from .utils import GridMask, GpuPhotoMetricDistortion, pad_multiple, OCC3D_CATEGORIES


class PGOccLightning(pl.LightningModule):
    """PG-Occ implemented as PyTorch Lightning module.

    Progressive Gaussian occupancy prediction with multi-frame temporal
    supervision and open-vocabulary features.
    """

    def __init__(
        self,
        # Backbone
        frozen_stages: int = 1,
        norm_eval: bool = True,
        backbone_pretrained: str = "",
        # Architecture
        embed_dims: int = 256,
        num_queries: list = None,
        num_frames: int = 8,
        num_points: int = 4,
        num_groups: int = 4,
        num_levels: int = 4,
        num_cams: int = 5,
        # Range
        pc_range: list = None,
        occ_size: list = None,
        voxel_size: float = 0.4,
        # Rendering
        render_h: int = 180,
        render_w: int = 320,
        use_ov: bool = True,
        ov_dim: int = 768,
        # Losses
        loss_weights: dict = None,
        warp_warmup_epochs: float = 2,
        # Masking
        ego_car_mask_dir: str = "",
        ego_car_mask_map: dict = None,
        sam3_class_config: str = "",
        # Evaluation
        density_threshold: float = 0.04,
        text_protos: str = "",
        text_protos_sam3: str = "",
        text_loss_temp: float = 0.07,
        # Voxelizer
        vol_range: list = None,
        filter_gaussians: bool = True,
        opacity_thresh: float = 0.6,
        sigma_factor: float = 3.0,
        # Optimizer
        learning_rate: float = 2e-4,
        weight_decay: float = 0.01,
        backbone_lr_mult: float = 0.1,
        sampling_offset_lr_mult: float = 0.1,
        gradient_clip_val: float = 350.0,
        warmup_iters: int = 500,
        warmup_factor: float = 1.0 / 3.0,
        lr_schedule: str = "cosine",
        min_lr_ratio: float = 0.001,
        # Augmentation
        use_grid_mask: bool = False,
        img_color_aug: bool = True,
        to_rgb: bool = True,
        size_divisor: int = 32,
        mean: list = None,
        std: list = None,
        # Camera names (ordered)
        camera_names: list = None,
        **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters()

        # Defaults
        if num_queries is None:
            num_queries = [4000, 1000, 1000]
        if pc_range is None:
            pc_range = [-40, -40, -1, 40, 40, 5.4]
        if occ_size is None:
            occ_size = [200, 200, 16]
        if loss_weights is None:
            loss_weights = dict(depth_warping=10.0, ov_mse=10.0, ov_cos=1.0, depth_foundation=0.5)
        if vol_range is None:
            vol_range = [-40.4, -40.4, -1.4, 39.6, 39.6, 5.0]
        if mean is None:
            mean = [123.675, 116.28, 103.53]
        if std is None:
            std = [58.395, 57.12, 57.375]
        if camera_names is None:
            camera_names = [
                "CAM_FRONT_WIDE", "CAM_FRONT_RIGHT_WIDE", "CAM_FRONT_LEFT_WIDE",
                "CAM_BACK_LEFT_WIDE", "CAM_BACK_RIGHT_WIDE",
            ]

        self.pc_range = pc_range
        self.occ_size = occ_size
        self.num_cams = num_cams
        self.num_frames = num_frames
        self.loss_weights = loss_weights
        self.warp_warmup_epochs = warp_warmup_epochs
        self.density_threshold = density_threshold
        self.render_conf = dict(render_h=render_h, render_w=render_w)
        self.img_color_aug = img_color_aug
        self.size_divisor = size_divisor
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.backbone_lr_mult = backbone_lr_mult
        self.sampling_offset_lr_mult = sampling_offset_lr_mult
        self.warmup_iters = warmup_iters
        self.warmup_factor = warmup_factor
        self.min_lr_ratio = min_lr_ratio
        self.to_rgb = to_rgb
        self.norm_eval = norm_eval
        self.training_iter = 0

        # Image normalization
        self.register_buffer('img_mean', torch.tensor(mean).view(1, 3, 1, 1))
        self.register_buffer('img_std', torch.tensor(std).view(1, 3, 1, 1))

        # === EMA PCA for stable dimensionality reduction (ported from GaussTR) ===
        reduce_dims = 128
        self.reduce_dims = reduce_dims
        self.pca_ema_momentum = 0.1
        self.register_buffer('pca_v', torch.zeros(ov_dim, reduce_dims))
        self.register_buffer('pca_initialized', torch.tensor(0, dtype=torch.long))

        # === Classification head MLP (ported from GaussTR) ===
        # Learnable MLP absorbs classification gradient, preventing sem_ce from
        # pulling OV features away from DINOv3CLIP targets.
        # Input: PCA-reduced rendered features (128-dim) → 16 SAM3 classes
        self.class_head = nn.Sequential(
            nn.Linear(reduce_dims, reduce_dims * 4),
            nn.ReLU(),
            nn.Linear(reduce_dims * 4, 16),
        )

        # === Ego car masks ===
        # Pre-computed binary masks for cameras where the ego car bonnet is visible.
        # Loaded once at init, resized to render resolution, registered as buffers.
        self._load_ego_car_masks(
            ego_car_mask_dir, ego_car_mask_map or {}, camera_names,
            render_h, render_w, num_cams)

        # === SAM3 semantic class config ===
        # Load class IDs for sky and dynamic objects from config file.
        self._load_sam3_class_config(sam3_class_config)

        # Augmentation
        if use_grid_mask:
            self.grid_mask = GridMask(ratio=0.5, prob=0.7)
        else:
            self.grid_mask = None
        self.color_aug = GpuPhotoMetricDistortion()

        # === Backbone: ResNet50 + FPN ===
        self._build_backbone(frozen_stages, backbone_pretrained)

        self.neck = FPN(
            in_channels=[256, 512, 1024, 2048],
            out_channels=embed_dims,
            num_outs=num_levels,
        )

        # === Decoder ===
        self.decoder = SparseGaussiansDecoder(
            embed_dims=embed_dims,
            num_queries=num_queries,
            num_frames=num_frames,
            num_points=num_points,
            num_groups=num_groups,
            num_levels=num_levels,
            num_cams=num_cams,
            pc_range=pc_range,
            ov_dim=ov_dim if use_ov else 0,
            render_conf=self.render_conf,
        )
        self.decoder.init_weights()

        # === Projection modules for temporal warping loss ===
        self.backproject_depth = BackprojectDepth(num_cams, render_h, render_w)
        self.project_3d = Project3D(num_cams, render_h, render_w)

        # === Text prompt embeddings for OV evaluation ===
        if text_protos and os.path.exists(text_protos):
            embeds = torch.load(text_protos, map_location='cpu')
            self.register_buffer('text_proto_embeds', embeds)
        else:
            self.register_buffer('text_proto_embeds', None)

        # === SAM3-aligned text embeddings for semantic contrastive loss ===
        if text_protos_sam3 and os.path.exists(text_protos_sam3):
            sam3_embeds = torch.load(text_protos_sam3, map_location='cpu')
            # Ensure shape is [embed_dim, num_classes]
            if sam3_embeds.shape[0] < sam3_embeds.shape[1]:
                sam3_embeds = sam3_embeds.T
            self.register_buffer('text_proto_embeds_sam3', sam3_embeds)
        else:
            self.register_buffer('text_proto_embeds_sam3', None)
        self.text_loss_temp = text_loss_temp

        # === Voxelizer for evaluation ===
        self.voxelizer = None
        self._voxelizer_cfg = dict(
            vol_range=vol_range, voxel_size=voxel_size,
            filter_gaussians=filter_gaussians,
            opacity_thresh=opacity_thresh, sigma_factor=sigma_factor,
        )

    def _load_ego_car_masks(self, mask_dir, mask_map, camera_names, render_h, render_w, num_cams):
        """Load ego car masks and register as a buffer [N, 1, Rh, Rw] bool."""
        # Build per-camera mask: True = valid pixel, False = ego car
        masks = []
        for i in range(num_cams):
            cam_name = camera_names[i] if i < len(camera_names) else f"cam_{i}"
            mask_file = mask_map.get(cam_name, "")
            if mask_file and mask_dir and os.path.exists(os.path.join(mask_dir, mask_file)):
                mask_path = os.path.join(mask_dir, mask_file)
                raw = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                # Mask PNGs: 255 = ego car, 0 = background → invert to True = valid
                resized = cv2.resize(raw, (render_w, render_h), interpolation=cv2.INTER_NEAREST)
                mask_tensor = torch.from_numpy(resized).bool()
                masks.append(~mask_tensor)  # True = valid, False = ego car
            else:
                # No ego car visible in this camera → all valid
                masks.append(torch.ones(render_h, render_w, dtype=torch.bool))

        # [N, 1, Rh, Rw] bool
        ego_mask = torch.stack(masks).unsqueeze(1)
        self.register_buffer('ego_car_mask', ego_mask)

    def _load_sam3_class_config(self, config_path):
        """Load SAM3 class IDs for sky and dynamic object classes."""
        if config_path and os.path.exists(config_path):
            with open(config_path) as f:
                cfg = json.load(f)
            sky_ids = []
            dynamic_ids = []
            for cat_key, cat_val in cfg['categories'].items():
                for cls in cat_val['classes']:
                    if cls['name'] == 'sky':
                        sky_ids.append(cls['id'])
                    elif cat_key == 'dynamic':
                        dynamic_ids.append(cls['id'])
            self._sam3_sky_ids = sky_ids or [16]
            self._sam3_dynamic_ids = dynamic_ids or [2, 3, 4, 5, 6, 7, 9, 10]
        else:
            # Hardcoded fallback matching T4 SAM3 classes
            self._sam3_sky_ids = [16]
            self._sam3_dynamic_ids = [2, 3, 4, 5, 6, 7, 9, 10]

    def _build_sam3_masks(self, sam3_mask):
        """Build per-loss masks from SAM3 semantic labels.

        Args:
            sam3_mask: [N, 1, Rh, Rw] int64 SAM3 class labels (pre-resized in dataloader)

        Returns:
            sky_mask: [N, 1, Rh, Rw] bool — True = not sky
            dynamic_mask: [N, 1, Rh, Rw] bool — True = not dynamic object
        """
        # SAM3 is already at render resolution (resized in PackPGOccInputs)
        # Sky mask: True where NOT sky
        sky_mask = torch.ones_like(sam3_mask, dtype=torch.bool)
        for sid in self._sam3_sky_ids:
            sky_mask &= (sam3_mask != sid)

        # Dynamic mask: True where NOT dynamic object
        dynamic_mask = torch.ones_like(sam3_mask, dtype=torch.bool)
        dynamic_ids = torch.tensor(self._sam3_dynamic_ids, device=sam3_mask.device)
        for did in dynamic_ids:
            dynamic_mask &= (sam3_mask != did)

        return sky_mask, dynamic_mask

    def _text_contrastive_loss(self, pred_feats, sam3_labels, text_embeds, valid_row=0):
        """Text contrastive loss: cosine similarity to SAM3-aligned text prototypes.

        Adapted from GaussTR's text_contrastive_loss (gausstr_head.py:544-591).

        Args:
            pred_feats: [N, C, H, W] features in full OV space (reconstructed from PCA)
            sam3_labels: [N, H, W] SAM3 class labels (0-17)
            text_embeds: [C, 17] SAM3-aligned text prototype embeddings
            valid_row: Number of top rows to exclude (blind region)

        Returns:
            Scalar contrastive loss
        """
        N, C, H, W = pred_feats.shape

        pred_norm = F.normalize(pred_feats, dim=1, eps=1e-6)
        text_norm = F.normalize(text_embeds, dim=0, eps=1e-6)

        # Cosine similarity: [N, C, H, W] @ [C, 17] -> [N, 17, H, W]
        logits = torch.einsum('nchw,ck->nkhw', pred_norm, text_norm) / self.text_loss_temp

        # Valid mask: exclude background (0), other_flat (12), sky (17)
        valid_mask = (sam3_labels >= 1) & (sam3_labels <= 16) & (sam3_labels != 12)
        if valid_row > 0:
            valid_mask[:, :valid_row, :] = False
        # Apply ego mask
        valid_mask = valid_mask & self.ego_car_mask.squeeze(1).to(valid_mask.device)

        # Shift labels: SAM3 1-17 -> 0-16
        targets = (sam3_labels - 1).clamp(min=0)

        ce_loss = F.cross_entropy(logits, targets.long(), reduction='none')
        num_valid = valid_mask.float().sum().clamp(min=1)
        return (ce_loss * valid_mask.float()).sum() / num_valid

    def _build_backbone(self, frozen_stages, pretrained_path):
        """Build ResNet50 backbone with optional COCO pretrained weights."""
        backbone = resnet50(weights=None)

        # Load pretrained weights (local checkpoint path)
        if pretrained_path and os.path.exists(pretrained_path):
            ckpt = torch.load(pretrained_path, map_location='cpu')
            state_dict = ckpt.get('state_dict', ckpt) if isinstance(ckpt, dict) else ckpt

            # Strip common prefixes from mmcv/mmdet checkpoints.
            prefixes = (
                'img_backbone.',
                'module.img_backbone.',
                'model.img_backbone.',
                'backbone.',
                'module.backbone.',
                'model.backbone.',
            )
            cleaned = {}
            for k, v in state_dict.items():
                new_k = k
                for prefix in prefixes:
                    if new_k.startswith(prefix):
                        new_k = new_k[len(prefix):]
                        break
                if new_k.startswith('module.'):
                    new_k = new_k[len('module.'):]
                cleaned[new_k] = v

            missing, unexpected = backbone.load_state_dict(cleaned, strict=False)
            print(
                f"Loaded backbone weights from {pretrained_path} "
                f"(missing={len(missing)}, unexpected={len(unexpected)})"
            )
        else:
            # Fall back to ImageNet pretrained
            from torchvision.models import ResNet50_Weights
            backbone = resnet50(weights=ResNet50_Weights.DEFAULT)
            print("Using ImageNet pretrained ResNet50 backbone")

        # Extract feature stages (discard avgpool + fc)
        self.backbone_stem = nn.Sequential(
            backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool)
        self.backbone_layers = nn.ModuleList([
            backbone.layer1, backbone.layer2, backbone.layer3, backbone.layer4])

        # Freeze stages: stem is always frozen when frozen_stages >= 0
        if frozen_stages >= 0:
            for param in self.backbone_stem.parameters():
                param.requires_grad = False

        # Freeze layer1..layerN based on frozen_stages (1-indexed)
        for i in range(min(frozen_stages, len(self.backbone_layers))):
            layer = self.backbone_layers[i]
            for param in layer.parameters():
                param.requires_grad = False

        self._frozen_stages = frozen_stages

    def _get_voxelizer(self):
        """Lazy-init voxelizer (avoids CUDA init at __init__ time)."""
        if self.voxelizer is None:
            from models.cuda_voxelizer import CUDAVoxelizer
            self.voxelizer = CUDAVoxelizer(**self._voxelizer_cfg).to(self.device)
        return self.voxelizer

    def extract_feat(self, img, img_metas):
        """Extract multi-scale features from images.

        Args:
            img: [B, T*N, 3, H, W] raw images (uint8 range [0, 255] or float).
            img_metas: List of dicts with camera metadata.

        Returns:
            List of [B, T*N, C, Hi, Wi] feature tensors per FPN level.
        """
        B, TN, C, H, W = img.shape
        img = img.reshape(B * TN, C, H, W).float()

        # Color augmentation (training only)
        if self.training and self.img_color_aug:
            img = self.color_aug(img)

        # Match original PG-Occ normalization path: BGR -> RGB before norm.
        if self.to_rgb:
            img = img[:, [2, 1, 0], :, :]

        # Normalize
        img = (img - self.img_mean) / self.img_std

        # Update img_metas with shape info
        for b in range(B):
            img_shape = (img.shape[2], img.shape[3], img.shape[1])
            img_metas[b]['img_shape'] = [img_shape for _ in range(TN)]
            img_metas[b]['ori_shape'] = [img_shape for _ in range(TN)]

        # Pad to size_divisor
        if self.size_divisor > 1:
            img = pad_multiple(img, img_metas, size_divisor=self.size_divisor)

        input_shape = img.shape[-2:]
        for img_meta in img_metas:
            img_meta.update(input_shape=input_shape)

        # Grid mask augmentation
        if self.training and self.grid_mask is not None:
            img = self.grid_mask(img)

        # Backbone forward
        x = self.backbone_stem(img)
        feats = []
        for layer in self.backbone_layers:
            x = layer(x)
            feats.append(x)

        # FPN
        fpn_feats = self.neck(feats)

        # Reshape: [B*TN, C, H, W] -> [B, TN, C, H, W]
        out = []
        for feat in fpn_feats:
            _, Cf, Hf, Wf = feat.shape
            out.append(feat.reshape(B, TN, Cf, Hf, Wf))

        return out

    def forward(self, batch):
        """Forward pass for training or inference."""
        img = batch['img']
        img_metas = batch['img_metas']
        depth = batch.get('depth')

        mlvl_feats = self.extract_feat(img, img_metas)
        gau_preds = self.decoder(mlvl_feats, img_metas=img_metas, depth=depth)

        return gau_preds

    def training_step(self, batch, batch_idx):
        """Training step with multi-level Gaussian losses."""
        self.training_iter += 1
        gau_preds = self.forward(batch)
        img_metas = batch['img_metas']

        # Prepare camera attributes for rendering
        K, C2W, W2C = prepare_gs_attribute(img_metas, num_cams=self.num_cams)

        # EMA PCA reduction of OV target features (ported from GaussTR)
        ov_tgt_feature = batch['text_vision'].clone().detach().permute(0, 1, 3, 4, 2)
        ov_tgt_feature_pca = ov_tgt_feature.flatten(2, 3)

        with torch.amp.autocast('cuda', enabled=False):
            _, _, pca_v_batch = torch.pca_lowrank(
                ov_tgt_feature_pca.flatten(0, 2).double(), q=self.reduce_dims, niter=4)

            if self.pca_initialized.item() == 0:
                self.pca_v.copy_(pca_v_batch.float())
                self.pca_initialized.fill_(1)
            else:
                # Align signs to handle PCA sign ambiguity
                sign = torch.sign((self.pca_v.double() * pca_v_batch).sum(dim=0, keepdim=True))
                sign = torch.where(sign == 0, torch.ones_like(sign), sign)
                v_aligned = (pca_v_batch * sign).float()
                self.pca_v.mul_(1 - self.pca_ema_momentum).add_(
                    v_aligned * self.pca_ema_momentum)

            # Sync across GPUs in DDP
            if torch.distributed.is_initialized():
                torch.distributed.all_reduce(self.pca_v, op=torch.distributed.ReduceOp.AVG)

        pca_v = self.pca_v  # [ov_dim, reduce_dims], stable EMA projection
        ov_tgt_feature = ov_tgt_feature @ pca_v.to(ov_tgt_feature)

        # Interpolate OV target to render resolution
        ov_tgt_feature = ov_tgt_feature.permute(0, 1, 4, 2, 3)
        B, N_cam, D, H_ov, W_ov = ov_tgt_feature.shape
        ov_tgt_feature = ov_tgt_feature.reshape(B * N_cam, D, H_ov, W_ov)
        ov_tgt_feature = F.interpolate(
            ov_tgt_feature,
            size=(self.render_conf['render_h'], self.render_conf['render_w']),
            mode='bilinear', align_corners=False)
        ov_tgt_feature = ov_tgt_feature.reshape(
            B, N_cam, D, self.render_conf['render_h'], self.render_conf['render_w'])
        ov_tgt_feature = ov_tgt_feature.permute(0, 1, 3, 4, 2)  # [B, N, Rh, Rw, D]

        # Compute losses per Gaussian level
        total_loss = torch.tensor(0.0, device=self.device)
        loss_dict = {}

        # Blind region mask: render rows with no backbone feature coverage.
        # For T4 (1860x2880 → 256x704), the top ~43% of render rows are blind.
        valid_row = img_metas[0].get('backbone_valid_row', 0)

        # === Build per-loss pixel masks ===
        ego_mask = self.ego_car_mask.to(self.device)  # [N, 1, Rh, Rw] bool

        # SAM3 semantic masks (sky + dynamic object exclusion)
        sam3_mask = batch.get('sam3_mask')
        sky_mask = None
        if sam3_mask is not None:
            sam3_mask = sam3_mask.squeeze(0).to(self.device)  # [N, 1, H, W]
            sky_mask, dynamic_mask = self._build_sam3_masks(sam3_mask)
            # Warp mask: exclude ego car + sky + dynamic objects
            warp_pixel_mask = ego_mask & sky_mask & dynamic_mask
            # Depth mask: exclude ego car + sky (dynamic objects have valid depth)
            depth_pixel_mask = ego_mask & sky_mask
            # OV mask: exclude ego car only (sky is a valid semantic class)
            ov_pixel_mask = ego_mask
        else:
            warp_pixel_mask = ego_mask
            depth_pixel_mask = ego_mask
            ov_pixel_mask = ego_mask

        # Smooth per-step warp warmup (replaces epoch-based jumps)
        if self.warp_warmup_epochs > 0:
            total_steps = self.trainer.estimated_stepping_batches
            max_epochs = self.trainer.max_epochs
            steps_per_epoch = max(1, total_steps // max(1, max_epochs))
            warp_warmup_steps = self.warp_warmup_epochs * steps_per_epoch
            warp_factor = min(1.0, self.training_iter / max(1, warp_warmup_steps))
        else:
            warp_factor = 1.0

        for i, gaussian in enumerate(gau_preds):
            # Apply PCA to predicted OV features
            if gaussian.ovs is not None:
                gaussian.ovs = gaussian.ovs @ pca_v.to(gaussian.ovs)

            # Render depth + OV features from Gaussians (blended — all Gaussians)
            render_results = batch_splatting_render(
                gaussian, W2C, K, render_conf=self.render_conf)

            render_depth_ed = render_results['depth'].permute(0, 3, 1, 2)
            render_alphas = render_results['alphas'].permute(0, 3, 1, 2)  # [N, 1, H, W]
            render_depth_ed = render_depth_ed.clamp(min=0.1, max=80.0)
            alpha_mask = (render_alphas > 0.1).detach()

            # === Phase 2: Branch-aware static/dynamic rendering ===
            if gaussian.branch_probs is not None and sam3_mask is not None:
                p_static = gaussian.branch_probs[..., 0]   # [B, Q]
                p_dynamic = gaussian.branch_probs[..., 1]   # [B, Q]

                # Static layer: opacity * p_static — for warp loss
                gaussian_static = GaussianPrediction(
                    means=gaussian.means, scales=gaussian.scales,
                    rotations=gaussian.rotations,
                    opacities=gaussian.opacities * p_static,
                    ovs=None, colors=None,
                )
                static_results = batch_splatting_render(
                    gaussian_static, W2C, K, render_conf=self.render_conf,
                    inference=True)
                static_depth = static_results['depth'].permute(0, 3, 1, 2).clamp(0.1, 80.0)
                static_alpha = static_results['alphas'].permute(0, 3, 1, 2)

                # Dynamic layer: opacity * p_dynamic — for branch supervision
                gaussian_dynamic = GaussianPrediction(
                    means=gaussian.means, scales=gaussian.scales,
                    rotations=gaussian.rotations,
                    opacities=gaussian.opacities * p_dynamic,
                    ovs=None, colors=None,
                )
                dynamic_results = batch_splatting_render(
                    gaussian_dynamic, W2C, K, render_conf=self.render_conf,
                    inference=True)
                dynamic_alpha = dynamic_results['alphas'].permute(0, 3, 1, 2)

                # Warp uses STATIC depth (dynamic objects excluded at render level)
                warp_depth = static_depth
                warp_alpha_mask = (static_alpha > 0.1).detach()
            else:
                warp_depth = render_depth_ed
                warp_alpha_mask = alpha_mask
                static_alpha = None
                dynamic_alpha = None

            # Temporal depth warping loss (on STATIC depth only)
            warp_full_mask = warp_pixel_mask & warp_alpha_mask
            do_warp_diag = (i == 0 and batch_idx % 50 == 0)
            loss_warp_result = calc_time_warping_loss(
                warp_depth[0:self.num_cams],
                batch['t0_2_x_geo'], batch['render_gt'],
                self.backproject_depth, self.project_3d, K,
                num_cams=self.num_cams, valid_row=valid_row,
                pixel_mask=warp_full_mask,
                return_diagnostics=do_warp_diag)
            if do_warp_diag:
                loss_warp, warp_diag = loss_warp_result
                # Always populate diagnostic keys (even for stationary samples)
                # to ensure all DDP ranks call the same number of sync_dist allreduces.
                loss_dict['warp_identity_loss'] = warp_diag['identity_loss']
                loss_dict['warp_reproj_loss'] = warp_diag['warp_reproj_loss']
                loss_dict['warp_wins_frac'] = warp_diag['warp_wins_frac']
                loss_dict['warp_oob_frac'] = warp_diag.get('oob_frac', 0.0)
                if not warp_diag.get('skipped', False):
                    # Control: compute warp with foundation depth
                    with torch.no_grad():
                        foundation_depth = batch['depth'].clone().squeeze(0).to(self.device)
                        fd_resized = F.interpolate(
                            foundation_depth, size=(self.render_conf['render_h'],
                                                    self.render_conf['render_w']),
                            mode='bilinear', align_corners=False)
                        _, fd_diag = calc_time_warping_loss(
                            fd_resized[0:self.num_cams],
                            batch['t0_2_x_geo'], batch['render_gt'],
                            self.backproject_depth, self.project_3d, K,
                            num_cams=self.num_cams, valid_row=valid_row,
                            pixel_mask=warp_pixel_mask,
                            return_diagnostics=True)
                    loss_dict['fd_warp_identity_loss'] = fd_diag['identity_loss']
                    loss_dict['fd_warp_reproj_loss'] = fd_diag['warp_reproj_loss']
                    loss_dict['fd_warp_wins_frac'] = fd_diag['warp_wins_frac']
                else:
                    loss_dict['fd_warp_identity_loss'] = 0.0
                    loss_dict['fd_warp_reproj_loss'] = 0.0
                    loss_dict['fd_warp_wins_frac'] = 0.0
            else:
                loss_warp = loss_warp_result
            loss_dict[f'warp_{i}'] = loss_warp.item()
            total_loss = total_loss + loss_warp * self.loss_weights['depth_warping'] * warp_factor

            # OV feature losses (masked to backbone-visible region + ego mask)
            if gaussian.ovs is not None:
                ov_feature = render_results['ov_feature'].unsqueeze(0)  # [1, N, Rh, Rw, D]

                # Build OV spatial mask: ego car + valid_row (no alpha_mask — supervise low-alpha regions)
                ov_mask_spatial = ov_pixel_mask.float()  # [N, 1, Rh, Rw]
                if valid_row > 0:
                    ov_mask_spatial[:, :, :valid_row, :] = 0.0

                # Apply mask to OV features: [1, N, Rh, Rw, 1]
                ov_mask_5d = ov_mask_spatial.unsqueeze(0).permute(0, 1, 3, 4, 2)  # [1, N, Rh, Rw, 1]

                # MSE loss (masked)
                ov_diff_sq = (ov_feature - ov_tgt_feature) ** 2
                ov_diff_masked = ov_diff_sq * ov_mask_5d
                ov_count = ov_mask_5d.sum().clamp(min=1.0) * D
                loss_ov_mse = ov_diff_masked.sum() / ov_count
                loss_dict[f'ov_mse_{i}'] = loss_ov_mse.item()
                total_loss = total_loss + loss_ov_mse * self.loss_weights['ov_mse']

                # Cosine similarity loss (masked)
                ov_normed = ov_feature / (ov_feature.norm(dim=-1, keepdim=True) + 1e-8)
                tgt_normed = ov_tgt_feature / (ov_tgt_feature.norm(dim=-1, keepdim=True) + 1e-8)
                cos_sim = F.cosine_similarity(
                    ov_normed.reshape(-1, D), tgt_normed.reshape(-1, D))
                cos_mask_flat = ov_mask_5d.squeeze(-1).reshape(-1)
                cos_count = cos_mask_flat.sum().clamp(min=1.0)
                loss_ov_cos = 1.0 - (cos_sim * cos_mask_flat).sum() / cos_count
                loss_dict[f'ov_cos_{i}'] = loss_ov_cos.item()
                total_loss = total_loss + loss_ov_cos * self.loss_weights['ov_cos']

            # SAM3 semantic classification loss via class_head MLP (ported from GaussTR)
            # The MLP absorbs classification gradient — features stay aligned with DINOv3CLIP.
            if (sam3_mask is not None and gaussian.ovs is not None
                    and 'sem_ce' in self.loss_weights):
                # ov_feature: [1, N, Rh, Rw, D_pca] — rendered PCA-projected features
                ov_rendered = ov_feature.squeeze(0)  # [N, Rh, Rw, D_pca]
                # class_head MLP: [N, Rh, Rw, D_pca] → [N, Rh, Rw, 16] → [N, 16, Rh, Rw]
                class_logits = self.class_head(ov_rendered).permute(0, 3, 1, 2)

                # SAM3 labels: values 0-17 (0=bg, 16=sky, 17=sky variant)
                sam3_labels = sam3_mask.squeeze(1)  # [N, Rh, Rw]
                targets = (sam3_labels - 1).clamp(min=0)  # shift 1-16 to 0-15

                # Valid: exclude background (0), sky (>=16)
                valid = (sam3_labels >= 1) & (sam3_labels <= 15)
                if valid_row > 0:
                    valid[:, :valid_row, :] = False
                valid = valid & ego_mask.squeeze(1)
                targets = torch.where(valid, targets, torch.zeros_like(targets))

                if valid.sum() > 0:
                    ce = F.cross_entropy(class_logits, targets.long(), reduction='none')
                    loss_sem_ce = (ce * valid.float()).sum() / valid.float().sum()
                    loss_dict[f'sem_ce_{i}'] = loss_sem_ce.item()
                    total_loss = total_loss + loss_sem_ce * self.loss_weights['sem_ce']

            # Text contrastive loss (gentle text-prototype alignment alongside MLP-based sem_ce)
            if (sam3_mask is not None and gaussian.ovs is not None
                    and 'sem_text' in self.loss_weights
                    and self.text_proto_embeds_sam3 is not None):
                ov_rendered = ov_feature.squeeze(0)  # [N, Rh, Rw, D_pca]
                # Project PCA features to full OV space for text similarity
                # pca_v: [ov_dim, reduce_dims] → transpose to [reduce_dims, ov_dim]
                ov_full = ov_rendered @ pca_v.to(ov_rendered).T  # [N, Rh, Rw, ov_dim]
                ov_full = ov_full.permute(0, 3, 1, 2)  # [N, ov_dim, Rh, Rw]

                sam3_labels = sam3_mask.squeeze(1)  # [N, Rh, Rw]
                loss_text = self._text_contrastive_loss(
                    ov_full, sam3_labels, self.text_proto_embeds_sam3, valid_row)
                loss_dict[f'sem_text_{i}'] = loss_text.item()
                total_loss = total_loss + loss_text * self.loss_weights['sem_text']

            # Foundation depth loss (masked: ego car + sky + valid_row)
            depth_tgt = batch['depth'].clone().squeeze(0)  # [N, 1, Hd, Wd]
            mask = (depth_tgt > 0.1) & (depth_tgt < 51.2)
            if valid_row > 0:
                mask[:, :, :valid_row, :] = False
            # Apply ego + sky + alpha mask (alpha=0 → depth_ed is 0.1 clamp → SiLog noise)
            mask = mask & depth_pixel_mask & alpha_mask
            mask.detach_()
            loss_depth = get_depth_loss(render_depth_ed, depth_tgt, mask)
            loss_dict[f'depth_{i}'] = loss_depth.item()
            total_loss = total_loss + loss_depth * self.loss_weights['depth_foundation']

            # Sparse LiDAR GT depth loss (GT pre-downsampled to render resolution)
            if 'gt_depth' in batch and 'depth_gt' in self.loss_weights:
                gt_depth = batch['gt_depth'].squeeze(0).to(self.device)  # [N, 1, Rh, Rw]
                gt_mask = (gt_depth > 0.1) & (gt_depth < 80.0)
                # Mask blind region
                if valid_row > 0:
                    gt_mask[:, :, :valid_row, :] = False
                # Apply sky + ego masks (already at render resolution)
                if sky_mask is not None:
                    gt_mask = gt_mask & sky_mask
                gt_mask = gt_mask & ego_mask & alpha_mask
                gt_mask.detach_()
                loss_gt = get_gt_loss(render_depth_ed, gt_depth, gt_mask)
                loss_dict[f'depth_gt_{i}'] = loss_gt.item()
                total_loss = total_loss + loss_gt * self.loss_weights['depth_gt']

        # === Phase 2: Branch supervision losses (SelfOccFlow-inspired) ===
        # Encourage static alpha to cover static pixels, dynamic alpha to cover dynamic pixels.
        if (sam3_mask is not None
                and static_alpha is not None
                and dynamic_alpha is not None
                and 'branch_static' in self.loss_weights):
            # Build branch supervision mask: ego + valid_row + sky
            branch_valid = ego_mask.clone()
            if valid_row > 0:
                branch_valid[:, :, :valid_row, :] = False
            if sky_mask is not None:
                branch_valid = branch_valid & sky_mask

            # static_mask_pixels: True where SAM3 says static (dynamic_mask = True = NOT dynamic)
            static_pixels = dynamic_mask & branch_valid  # [N, 1, Rh, Rw]
            # dynamic_mask_pixels: True where SAM3 says dynamic
            dynamic_pixels = ~dynamic_mask & branch_valid  # [N, 1, Rh, Rw]

            # L_branch_static: BCE(alpha_static, 1) on static pixels
            if static_pixels.sum() > 10:
                sa = static_alpha[static_pixels].float().clamp(1e-6, 1 - 1e-6)
                loss_branch_s = F.binary_cross_entropy_with_logits(
                    torch.logit(sa), torch.ones_like(sa))
                loss_dict['branch_static'] = loss_branch_s.item()
                total_loss = total_loss + loss_branch_s * self.loss_weights['branch_static']

            # L_branch_dynamic: BCE(alpha_dynamic, 1) on dynamic pixels
            if dynamic_pixels.sum() > 10 and 'branch_dynamic' in self.loss_weights:
                da = dynamic_alpha[dynamic_pixels].float().clamp(1e-6, 1 - 1e-6)
                loss_branch_d = F.binary_cross_entropy_with_logits(
                    torch.logit(da), torch.ones_like(da))
                loss_dict['branch_dynamic'] = loss_branch_d.item()
                total_loss = total_loss + loss_branch_d * self.loss_weights['branch_dynamic']

            # L_branch_sep: penalize leakage (static alpha on dynamic, dynamic alpha on static)
            if 'branch_sep' in self.loss_weights:
                leak_s2d = (static_alpha[dynamic_pixels]).mean() if dynamic_pixels.sum() > 0 else torch.tensor(0.0)
                leak_d2s = (dynamic_alpha[static_pixels]).mean() if static_pixels.sum() > 0 else torch.tensor(0.0)
                loss_sep = leak_s2d + leak_d2s
                loss_dict['branch_sep'] = loss_sep.item()
                total_loss = total_loss + loss_sep * self.loss_weights['branch_sep']

            # Log branch stats periodically
            if batch_idx % 100 == 0:
                self.log('train/static_alpha_on_static',
                         static_alpha[static_pixels].mean().item() if static_pixels.sum() > 0 else 0.0,
                         sync_dist=True)
                self.log('train/dynamic_alpha_on_dynamic',
                         dynamic_alpha[dynamic_pixels].mean().item() if dynamic_pixels.sum() > 0 else 0.0,
                         sync_dist=True)
                # Log branch probability distribution
                if gau_preds and gau_preds[-1].branch_probs is not None:
                    bp = gau_preds[-1].branch_probs
                    self.log('train/p_static_mean', bp[..., 0].mean().item(), sync_dist=True)
                    self.log('train/p_dynamic_mean', bp[..., 1].mean().item(), sync_dist=True)

        # === Dynamic coverage losses (Phase 1: SelfOccFlow-inspired) ===
        # Encourage Gaussians to cover dynamic object regions instead of ignoring them.
        if (sam3_mask is not None
                and 'dyn_cov' in self.loss_weights
                and gau_preds):
            # dynamic_mask is True=NOT dynamic; invert for IS dynamic
            dyn_pixel_mask = ~dynamic_mask & ego_mask  # [N, 1, Rh, Rw]
            if valid_row > 0:
                dyn_pixel_mask[:, :, :valid_row, :] = False
            # Exclude sky from dynamic mask (safety)
            if sky_mask is not None:
                dyn_pixel_mask = dyn_pixel_mask & sky_mask

            dyn_count = dyn_pixel_mask.sum().clamp(min=1.0)

            # Use renders from finest Gaussian level (last in loop)
            # render_alphas / render_depth_ed are already from the last iteration

            # L_dyn_cov: BCE encouraging alpha→1 on dynamic pixels
            # Use logit-space BCE (autocast-safe under bf16-mixed)
            dyn_alpha = render_alphas[dyn_pixel_mask].float().clamp(1e-6, 1 - 1e-6)
            dyn_logits = torch.logit(dyn_alpha)
            loss_dyn_cov = F.binary_cross_entropy_with_logits(
                dyn_logits, torch.ones_like(dyn_logits))
            loss_dict['dyn_cov'] = loss_dyn_cov.item()
            total_loss = total_loss + loss_dyn_cov * self.loss_weights['dyn_cov']

            # L_dyn_depth: L1 on dynamic pixels where foundation depth is valid
            if 'dyn_depth' in self.loss_weights:
                dyn_depth_mask = dyn_pixel_mask & (depth_tgt > 0.1) & (depth_tgt < 51.2)
                if dyn_depth_mask.sum() > 10:
                    loss_dyn_depth = F.l1_loss(
                        render_depth_ed[dyn_depth_mask],
                        depth_tgt[dyn_depth_mask])
                    loss_dict['dyn_depth'] = loss_dyn_depth.item()
                    total_loss = total_loss + loss_dyn_depth * self.loss_weights['dyn_depth']

            # Log dynamic coverage stats periodically
            if batch_idx % 100 == 0:
                dyn_alpha_mean = render_alphas[dyn_pixel_mask].mean().item() if dyn_pixel_mask.sum() > 0 else 0.0
                self.log('train/dyn_alpha_coverage', dyn_alpha_mean, sync_dist=True)
                self.log('train/dyn_pixel_ratio', dyn_pixel_mask.float().mean().item(), sync_dist=True)

        # Log losses
        self.log('train_loss', total_loss, prog_bar=True, sync_dist=True)
        self.log('train/warp_factor', warp_factor, sync_dist=True)
        for k, v in loss_dict.items():
            self.log(f'train/{k}', v, sync_dist=True)

        # Log mask coverage stats (periodically)
        if batch_idx % 100 == 0:
            warp_coverage = warp_pixel_mask.float().mean().item()
            self.log('train/warp_mask_coverage', warp_coverage, sync_dist=True)

        return total_loss

    def validation_step(self, batch, batch_idx):
        """Validation step: predict occupancy grid."""
        gau_preds = self.forward(batch)
        img_metas = batch['img_metas']

        occ_preds = self.merge_occ_pred(gau_preds, img_metas)
        return {'occ_preds': occ_preds}

    def merge_occ_pred(self, gau_preds, img_metas):
        """Convert Gaussian predictions to occupancy grid via voxelization.

        Merges ALL progressive stages (coarse+medium+fine) for maximum Gaussian
        capacity, then voxelizes the combined set.

        Args:
            gau_preds: List of GaussianPrediction per decoder layer.
            img_metas: Image metadata.

        Returns:
            occ_preds: [B, X, Y, Z] integer class predictions.
        """
        voxelizer = self._get_voxelizer()

        # Merge all progressive stages for maximum capacity
        all_means = torch.cat([g.means for g in gau_preds], dim=1)
        all_opacities = torch.cat([g.opacities for g in gau_preds], dim=1)
        all_ovs = torch.cat([g.ovs for g in gau_preds], dim=1)
        all_rotations = torch.cat([g.rotations for g in gau_preds], dim=1)
        all_scales = torch.cat([g.scales for g in gau_preds], dim=1)

        # Compute class similarity via text prototype embeddings
        class_sim = torch.einsum('bnd,dm->bnm', all_ovs, self.text_proto_embeds)

        # Clamp scales to avoid singular covariance matrices in voxelizer
        scales = all_scales.clamp(min=1e-4)

        # Voxelize
        density, grid_feats = voxelizer(
            means3d=all_means,
            opacities=all_opacities,
            features=class_sim,
            rotations=all_rotations,
            scales=scales,
        )

        # Semantic predictions
        probs = grid_feats.softmax(-1)
        probs = self._merge_probs(probs, OCC3D_CATEGORIES)
        preds = probs.argmax(-1)
        # Adjust class indices for nuScenes convention (skip index 11)
        preds += (preds > 10) * 1 + 1
        # Apply density threshold: low density -> class 17 (empty)
        preds = torch.where(density.squeeze(-1) > self.density_threshold, preds, 17)

        return preds.cpu().numpy()

    @staticmethod
    def _merge_probs(probs, categories):
        """Merge probabilities for categories with multiple sub-classes."""
        merged = []
        idx = 0
        for cats in categories:
            p = probs[..., idx:idx + len(cats)]
            idx += len(cats)
            if len(cats) > 1:
                p = p.max(-1, keepdim=True).values
            merged.append(p)
        return torch.cat(merged, dim=-1)

    def configure_optimizers(self):
        """Configure AdamW with backbone lr multiplier and cosine schedule."""
        # Parameters that should not have weight decay:
        # biases, LayerNorm/layer_norm weights, and learnable embeddings.
        no_decay_keywords = {'bias', 'LayerNorm', 'layer_norm', 'query_embeds'}

        def _needs_decay(name):
            return not any(kw in name for kw in no_decay_keywords)

        # Separate backbone / sampling_offset / other parameters,
        # each split into decay and no-decay groups.
        backbone_decay, backbone_no_decay = [], []
        sampling_decay, sampling_no_decay = [], []
        other_decay, other_no_decay = [], []

        backbone_modules = [self.backbone_stem, self.backbone_layers]
        backbone_param_ids = set()
        for mod in backbone_modules:
            for name, p in mod.named_parameters():
                if p.requires_grad:
                    backbone_param_ids.add(id(p))
                    if _needs_decay(name):
                        backbone_decay.append(p)
                    else:
                        backbone_no_decay.append(p)

        for name, p in self.named_parameters():
            if not p.requires_grad or id(p) in backbone_param_ids:
                continue
            if 'sampling_offset' in name:
                if _needs_decay(name):
                    sampling_decay.append(p)
                else:
                    sampling_no_decay.append(p)
            else:
                if _needs_decay(name):
                    other_decay.append(p)
                else:
                    other_no_decay.append(p)

        param_groups = []
        if other_decay:
            param_groups.append({'params': other_decay, 'lr': self.learning_rate})
        if other_no_decay:
            param_groups.append({'params': other_no_decay, 'lr': self.learning_rate,
                                 'weight_decay': 0.0})
        if sampling_decay:
            param_groups.append({
                'params': sampling_decay,
                'lr': self.learning_rate * self.sampling_offset_lr_mult,
            })
        if sampling_no_decay:
            param_groups.append({
                'params': sampling_no_decay,
                'lr': self.learning_rate * self.sampling_offset_lr_mult,
                'weight_decay': 0.0,
            })
        if backbone_decay:
            param_groups.append({
                'params': backbone_decay,
                'lr': self.learning_rate * self.backbone_lr_mult,
            })
        if backbone_no_decay:
            param_groups.append({
                'params': backbone_no_decay,
                'lr': self.learning_rate * self.backbone_lr_mult,
                'weight_decay': 0.0,
            })

        optimizer = torch.optim.AdamW(
            param_groups,
            weight_decay=self.weight_decay,
        )

        # Cosine annealing with linear warmup
        total_steps = self.trainer.estimated_stepping_batches
        warmup_steps = self.warmup_iters

        def lr_lambda(step):
            if step < warmup_steps:
                alpha = step / max(warmup_steps, 1)
                return self.warmup_factor + (1.0 - self.warmup_factor) * alpha
            progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
            return max(self.min_lr_ratio, 0.5 * (1.0 + math.cos(progress * math.pi)))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'step',
                'frequency': 1,
            },
        }

    def train(self, mode=True):
        """Override to keep frozen backbone stages in eval mode."""
        super().train(mode)
        if mode:
            if self.norm_eval:
                for m in self.modules():
                    if isinstance(m, nn.BatchNorm2d):
                        m.eval()
            # Keep frozen stages in eval mode
            if self._frozen_stages >= 0:
                self.backbone_stem.eval()
                for param in self.backbone_stem.parameters():
                    param.requires_grad = False
            for i in range(min(self._frozen_stages, len(self.backbone_layers))):
                layer = self.backbone_layers[i]
                layer.eval()
                for param in layer.parameters():
                    param.requires_grad = False
        return self
