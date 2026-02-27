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
        warp_warmup_epochs: int = 2,
        # Masking
        ego_car_mask_dir: str = "",
        ego_car_mask_map: dict = None,
        sam3_class_config: str = "",
        # Evaluation
        density_threshold: float = 0.04,
        text_protos: str = "",
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
            sam3_mask: [N, 1, H_orig, W_orig] int64 SAM3 class labels

        Returns:
            sky_mask: [N, 1, Rh, Rw] bool — True = not sky
            dynamic_mask: [N, 1, Rh, Rw] bool — True = not dynamic object
        """
        rh, rw = self.render_conf['render_h'], self.render_conf['render_w']
        # Resize to render resolution using nearest-neighbor
        sam3_render = F.interpolate(
            sam3_mask.float(), size=(rh, rw), mode='nearest').long()  # [N, 1, Rh, Rw]

        # Sky mask: True where NOT sky
        sky_mask = torch.ones_like(sam3_render, dtype=torch.bool)
        for sid in self._sam3_sky_ids:
            sky_mask &= (sam3_render != sid)

        # Dynamic mask: True where NOT dynamic object
        dynamic_mask = torch.ones_like(sam3_render, dtype=torch.bool)
        dynamic_ids = torch.tensor(self._sam3_dynamic_ids, device=sam3_render.device)
        for did in dynamic_ids:
            dynamic_mask &= (sam3_render != did)

        return sky_mask, dynamic_mask

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

        # PCA reduction of OV target features
        ov_tgt_feature = batch['text_vision'].clone().detach().permute(0, 1, 3, 4, 2)
        ov_tgt_feature_pca = ov_tgt_feature.flatten(2, 3)

        pca_u, pca_s, pca_v = torch.pca_lowrank(
            ov_tgt_feature_pca.flatten(0, 2).double(), q=128, niter=4)

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

            # Render depth + OV features from Gaussians
            render_results = batch_splatting_render(
                gaussian, W2C, K, render_conf=self.render_conf)

            render_depths = render_results['depth'].permute(0, 3, 1, 2)
            render_depths = render_depths.clamp(min=0.1, max=80.0)

            # Temporal depth warping loss (with smooth warmup + pixel masking)
            loss_warp = calc_time_warping_loss(
                render_depths[0:self.num_cams],
                batch['t0_2_x_geo'], batch['render_gt'],
                self.backproject_depth, self.project_3d, K,
                num_cams=self.num_cams, valid_row=valid_row,
                pixel_mask=warp_pixel_mask)
            loss_dict[f'warp_{i}'] = loss_warp.item()
            total_loss = total_loss + loss_warp * self.loss_weights['depth_warping'] * warp_factor

            # OV feature losses (masked to backbone-visible region + ego mask)
            if gaussian.ovs is not None:
                ov_feature = render_results['ov_feature'].unsqueeze(0)  # [1, N, Rh, Rw, D]

                # Build OV spatial mask: ego car + valid_row
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

            # Foundation depth loss (masked: ego car + sky + valid_row)
            depth_tgt = batch['depth'].clone().squeeze(0)  # [N, 1, Hd, Wd]
            mask = (depth_tgt > 0.1) & (depth_tgt < 51.2)
            if valid_row > 0:
                mask[:, :, :valid_row, :] = False
            # Apply ego + sky mask to foundation depth
            mask = mask & depth_pixel_mask
            mask.detach_()
            loss_depth = get_depth_loss(render_depths, depth_tgt, mask)
            loss_dict[f'depth_{i}'] = loss_depth.item()
            total_loss = total_loss + loss_depth * self.loss_weights['depth_foundation']

            # Sparse LiDAR GT depth loss
            if 'gt_depth' in batch and 'depth_gt' in self.loss_weights:
                gt_depth = batch['gt_depth'].squeeze(0).to(self.device)  # [N, 1, H, W]
                gt_mask = (gt_depth > 0.1) & (gt_depth < 80.0)
                # Apply ego + sky mask at original resolution
                if sam3_mask is not None:
                    gt_sky_mask = torch.ones_like(sam3_mask, dtype=torch.bool)
                    for sid in self._sam3_sky_ids:
                        gt_sky_mask &= (sam3_mask != sid)
                    gt_mask = gt_mask & gt_sky_mask
                # Ego mask at original resolution
                ego_orig = F.interpolate(
                    ego_mask.float(), size=gt_depth.shape[-2:], mode='nearest').bool()
                gt_mask = gt_mask & ego_orig
                gt_mask.detach_()
                loss_gt = get_gt_loss(render_depths, gt_depth, gt_mask)
                loss_dict[f'depth_gt_{i}'] = loss_gt.item()
                total_loss = total_loss + loss_gt * self.loss_weights['depth_gt']

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

        Args:
            gau_preds: List of GaussianPrediction per decoder layer.
            img_metas: Image metadata.

        Returns:
            occ_preds: [B, X, Y, Z] integer class predictions.
        """
        voxelizer = self._get_voxelizer()

        # Use finest level (last decoder output)
        gaussian = gau_preds[-1]

        # Compute class similarity via text prototype embeddings
        class_sim = torch.einsum('bnd,dm->bnm', gaussian.ovs, self.text_proto_embeds)

        # Clamp scales to avoid singular covariance matrices in voxelizer
        scales = gaussian.scales.clamp(min=1e-4)

        # Voxelize
        density, grid_feats = voxelizer(
            means3d=gaussian.means,
            opacities=gaussian.opacities,
            features=class_sim,
            rotations=gaussian.rotations,
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
