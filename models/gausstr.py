"""GaussTR Lightning Module - Main model class.

PyTorch Lightning implementation of GaussTR for training and inference.
"""

from collections.abc import Iterable
from typing import Dict, Any, Optional, List, Tuple

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn

from .vitdet_fpn import ViTDetFPN
from .gausstr_decoder import GaussTRDecoder
from .gausstr_head import GaussTRHead
from .utils import flatten_multi_scale_feats


class GaussTRLightning(pl.LightningModule):
    """GaussTR implemented as PyTorch Lightning module.

    A foundation model-aligned Gaussian Transformer for self-supervised
    3D spatial understanding.

    Args:
        config: GaussTRConfig dataclass with model configuration.
    """

    def __init__(
        self,
        # Model architecture
        num_queries: int = 300,
        embed_dims: int = 256,
        feat_dims: int = 512,
        # Neck config
        neck_in_channels: int = 512,
        neck_out_channels: int = 256,
        neck_norm_type: str = "LN2d",
        # Decoder config
        decoder_num_layers: int = 3,
        decoder_embed_dims: int = 256,
        decoder_num_heads: int = 8,
        decoder_ffn_channels: int = 2048,
        decoder_num_levels: int = 4,
        # Head config
        head_reduce_dims: int = 128,
        head_image_shape: Tuple[int, int] = (432, 768),
        head_render_image_size: Tuple[int, int] = (900, 1600),
        head_patch_size: int = 16,
        head_depth_limit: float = 51.2,
        head_text_protos: Optional[str] = "ckpts/text_proto_embeds_clip.pth",
        head_text_protos_sam3: Optional[str] = None,  # SAM3-aligned text embeddings for training
        head_prompt_denoising: bool = True,
        head_num_segment_classes: int = 17,
        head_text_loss_weight: float = 3.0,  # Weight for text contrastive loss
        head_text_loss_temp: float = 0.1,  # Temperature for text contrastive loss
        head_cosine_loss_weight: float = 2.0,  # Weight for visual cosine loss
        head_depth_loss_weight: float = 1.0,  # Weight for depth loss
        head_position_loss_weight: float = 1.0,  # Weight for position loss (prevents opacity cheating)
        head_depth_warmup_iters: int = 0,  # Warmup steps to ramp depth loss weight
        head_pca_path: Optional[str] = None,  # Pre-computed PCA for single-view datasets
        # Voxelizer config (defaults match original GaussTR)
        vol_range: List[float] = None,
        voxel_size: float = 0.4,
        filter_gaussians: bool = False,
        opacity_thresh: float = 0.0,
        sigma_factor: float = 3.0,
        # Training config
        optimizer: str = "adamw",  # "adamw", "lion", or "adam8bit"
        learning_rate: float = 2e-4,
        weight_decay: float = 5e-3,
        betas: Tuple[float, float] = (0.9, 0.999),  # Adam betas
        warmup_iters: int = 200,
        warmup_factor: float = 1e-3,
        lr_schedule: str = "cosine",  # "cosine", "onecycle", or "multistep"
        min_lr_ratio: float = 0.01,  # Final LR = learning_rate * min_lr_ratio
        lr_milestones: List[int] = None,
        lr_gamma: float = 0.1,
        steps_per_epoch: int = None,  # Auto-calculated from dataloader if not specified
        gradient_clip_val: float = 35.0,
        # Data preprocessor config
        mean: List[float] = None,
        std: List[float] = None,
        # Performance optimizations
        torch_compile: bool = False,
        compile_mode: str = "reduce-overhead",
    ):
        super().__init__()
        self.save_hyperparameters()

        # Set defaults
        if vol_range is None:
            vol_range = [-40, -40, -1, 40, 40, 5.4]
        if lr_milestones is None:
            lr_milestones = [16]
        if mean is None:
            mean = [123.675, 116.28, 103.53]
        if std is None:
            std = [58.395, 57.12, 57.375]

        self.num_queries = num_queries
        self.embed_dims = embed_dims
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.warmup_iters = warmup_iters
        self.warmup_factor = warmup_factor
        self.lr_milestones = lr_milestones
        self.lr_gamma = lr_gamma
        self._steps_per_epoch = steps_per_epoch  # Will be auto-calculated if None
        self.gradient_clip_val = gradient_clip_val

        # Data preprocessing
        self.register_buffer('mean', torch.tensor(mean).view(1, 1, 3, 1, 1))
        self.register_buffer('std', torch.tensor(std).view(1, 1, 3, 1, 1))

        # Build model components
        self.neck = ViTDetFPN(
            in_channels=neck_in_channels,
            out_channels=neck_out_channels,
            norm_type=neck_norm_type
        )

        # Decoder config
        decoder_layer_cfg = {
            'self_attn_cfg': {
                'embed_dims': decoder_embed_dims,
                'num_heads': decoder_num_heads,
                'dropout': 0.0
            },
            'cross_attn_cfg': {
                'embed_dims': decoder_embed_dims,
                'num_levels': decoder_num_levels
            },
            'ffn_cfg': {
                'embed_dims': decoder_embed_dims,
                'feedforward_channels': decoder_ffn_channels
            }
        }

        self.decoder = GaussTRDecoder(
            num_layers=decoder_num_layers,
            layer_cfg=decoder_layer_cfg,
            return_intermediate=True
        )

        # Query embeddings
        self.query_embeds = nn.Embedding(num_queries, embed_dims)

        # Gaussian heads for each decoder layer
        head_cfg = {
            'embed_dims': embed_dims,
            'feat_dims': feat_dims,
            'reduce_dims': head_reduce_dims,
            'image_shape': head_image_shape,
            'render_image_size': head_render_image_size,
            'patch_size': head_patch_size,
            'depth_limit': head_depth_limit,
            'text_protos': head_text_protos,
            'text_protos_sam3': head_text_protos_sam3,
            'prompt_denoising': head_prompt_denoising,
            'num_segment_classes': head_num_segment_classes,
            'text_loss_weight': head_text_loss_weight,
            'text_loss_temp': head_text_loss_temp,
            'cosine_loss_weight': head_cosine_loss_weight,
            'depth_loss_weight': head_depth_loss_weight,
            'position_loss_weight': head_position_loss_weight,
            'pca_path': head_pca_path,
            'voxelizer_cfg': {
                'vol_range': vol_range,
                'voxel_size': voxel_size,
                'filter_gaussians': filter_gaussians,
                'opacity_thresh': opacity_thresh,
                'sigma_factor': sigma_factor,
            }
        }
        self.gauss_heads = nn.ModuleList([
            GaussTRHead(**head_cfg) for _ in range(decoder_num_layers)
        ])

        # Apply torch.compile() for performance optimization (PyTorch 2.0+)
        if torch_compile and hasattr(torch, 'compile'):
            print(f"Applying torch.compile() with mode='{compile_mode}' to neck and decoder")
            self.neck = torch.compile(self.neck, mode=compile_mode)
            self.decoder = torch.compile(self.decoder, mode=compile_mode)

    def _forward_features(
        self,
        feats: torch.Tensor,
        batch_size: int,
        sem_segs: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """Shared feature extraction through neck and decoder.

        Args:
            feats: Pre-extracted features [B, N, C, H, W].
            batch_size: Batch size.
            sem_segs: Semantic segmentation masks [B, N, H, W] for sky filtering.

        Returns:
            Dictionary with hidden_states and references from decoder.
        """
        # Use pre-extracted features
        x = feats.flatten(0, 1)

        # Multi-scale features
        multi_scale_feats = self.neck(x)

        # Prepare decoder inputs
        decoder_inputs = self.pre_transformer(multi_scale_feats)
        feat_flatten = flatten_multi_scale_feats(multi_scale_feats)[0]
        decoder_inputs.update(self.pre_decoder(feat_flatten, batch_size, sem_segs=sem_segs))

        # Forward through decoder
        return self.forward_decoder(
            reg_branches=[h.regress_head for h in self.gauss_heads],
            **decoder_inputs
        )

    def forward(
        self,
        images: torch.Tensor,
        feats: torch.Tensor,
        depth: torch.Tensor,
        cam2img: torch.Tensor,
        cam2ego: torch.Tensor,
        img_aug_mat: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """Forward pass for inference.

        Args:
            images: Multi-view images [B, N, 3, H, W].
            feats: Pre-extracted features [B, N, C, H, W].
            depth: Depth maps [B, N, 1, H, W].
            cam2img: Camera intrinsics [B, N, 4, 4].
            cam2ego: Camera extrinsics [B, N, 4, 4].
            img_aug_mat: Image augmentation matrix [B, N, 4, 4]. Optional.

        Returns:
            Occupancy predictions.
        """
        bs, n = images.shape[:2]

        # Forward through neck and decoder
        decoder_outputs = self._forward_features(feats, bs)
        query = decoder_outputs['hidden_states']
        reference_points = decoder_outputs['references']

        # Ensure depth has channel dimension [B, N, 1, H, W]
        if depth.dim() == 4:
            depth = depth.unsqueeze(2)

        # Use last layer for prediction
        result = self.gauss_heads[-1](
            query[-1], reference_points[-1],
            depth=depth,
            cam2img=cam2img,
            cam2ego=cam2ego,
            img_aug_mat=img_aug_mat,
            mode='predict',
            **kwargs
        )

        return result

    def training_step(
        self,
        batch: Dict[str, torch.Tensor],
        batch_idx: int
    ) -> torch.Tensor:
        """Training step.

        Args:
            batch: Dictionary containing:
                - images: [B, N, 3, H, W]
                - feats: [B, N, C, H, W]
                - depth: [B, N, 1, H, W]
                - cam2img: [B, N, 4, 4]
                - cam2ego: [B, N, 4, 4]
                - img_aug_mat: [B, N, 4, 4] (optional)
                - sem_segs: [B, N, H, W] (optional)
            batch_idx: Batch index.

        Returns:
            Total loss.
        """
        images = batch['images']
        feats = batch['feats']
        depth = batch['depth']
        cam2img = batch['cam2img']
        cam2ego = batch['cam2ego']
        img_aug_mat = batch.get('img_aug_mat')
        sem_segs = batch.get('sem_seg')

        bs, n = images.shape[:2]

        # Forward through neck and decoder (pass sem_segs for sky-aware reference point init)
        decoder_outputs = self._forward_features(feats, bs, sem_segs=sem_segs)
        query = decoder_outputs['hidden_states']
        reference_points = decoder_outputs['references']

        # Compute losses at each decoder layer
        losses = {}
        total_loss = 0.0

        for i, gauss_head in enumerate(self.gauss_heads):
            layer_losses = gauss_head(
                query[i], reference_points[i],
                depth=depth,
                cam2img=cam2img,
                cam2ego=cam2ego,
                feats=feats,
                img_aug_mat=img_aug_mat,
                sem_segs=sem_segs,
                layer_idx=i,
                debug_step=(self.global_step == 0 and batch_idx == 0),
                mode='loss'
            )
            for k, v in layer_losses.items():
                losses[f'{k}/{i}'] = v
                # Only add actual losses to total, not monitoring metrics
                if k.startswith('loss_'):
                    total_loss += v

        # Log losses
        self.log_dict(losses, prog_bar=True, sync_dist=True, batch_size=bs)
        self.log('train_loss', total_loss, prog_bar=True, sync_dist=True, on_step=True, on_epoch=True, batch_size=bs)

        return total_loss

    @torch.no_grad()
    def validation_step(
        self,
        batch: Dict[str, torch.Tensor],
        batch_idx: int
    ) -> Dict[str, torch.Tensor]:
        """Validation step.

        Args:
            batch: Same as training_step plus gt_occ for evaluation.
            batch_idx: Batch index.

        Returns:
            Dictionary with predictions and ground truth.
        """
        preds = self.forward(
            images=batch['images'],
            feats=batch['feats'],
            depth=batch['depth'],
            cam2img=batch['cam2img'],
            cam2ego=batch['cam2ego'],
            img_aug_mat=batch.get('img_aug_mat')
        )

        # Update metric if ground truth available
        gt_occ = batch.get('gt_occ')
        mask = batch.get('mask_camera')

        if gt_occ is not None:
            # Lazy initialization of metric (separate from test metric)
            if not hasattr(self, 'val_occ_metric'):
                from evaluation import OccupancyIoU
                self.val_occ_metric = OccupancyIoU(
                    num_classes=18,
                    use_camera_mask=True,
                    ignore_index=17
                ).to(self.device)

            self.val_occ_metric.update(preds, gt_occ, mask)

        return {
            'preds': preds,
            'gt_occ': gt_occ,
            'mask': mask
        }

    def on_validation_epoch_end(self):
        """Compute and log final metrics at end of validation epoch."""
        if hasattr(self, 'val_occ_metric'):
            results = self.val_occ_metric.compute()
            self.log('val/miou', results['miou'], sync_dist=True)
            self.log('val/occ_iou', results['occ_iou'], sync_dist=True)

            # Log per-class IoU
            for i, class_name in enumerate(self.val_occ_metric.class_names):
                if i != self.val_occ_metric.ignore_index:
                    self.log(f'val/iou_{class_name}', results[f'iou_{class_name}'], sync_dist=True)

            # Print formatted table only on rank 0
            if self.global_rank == 0:
                print(self.val_occ_metric.get_table_str())
            self.val_occ_metric.reset()

    @torch.no_grad()
    def test_step(
        self,
        batch: Dict[str, torch.Tensor],
        batch_idx: int
    ) -> Dict[str, torch.Tensor]:
        """Test step - computes predictions and updates metrics.

        Args:
            batch: Dictionary containing input data and ground truth.
            batch_idx: Batch index.

        Returns:
            Dictionary with predictions.
        """
        preds = self.forward(
            images=batch['images'],
            feats=batch['feats'],
            depth=batch['depth'],
            cam2img=batch['cam2img'],
            cam2ego=batch['cam2ego'],
            img_aug_mat=batch.get('img_aug_mat')
        )

        # Update metric if ground truth available
        gt_occ = batch.get('gt_occ')
        mask = batch.get('mask_camera')

        if gt_occ is not None:
            # Lazy initialization of metric
            if not hasattr(self, 'occ_metric'):
                from evaluation import OccupancyIoU
                self.occ_metric = OccupancyIoU(
                    num_classes=18,
                    use_camera_mask=True,
                    ignore_index=17
                ).to(self.device)

            self.occ_metric.update(preds, gt_occ, mask)

        return {'preds': preds}

    def on_test_epoch_end(self):
        """Compute and log final metrics at end of test epoch."""
        if hasattr(self, 'occ_metric'):
            results = self.occ_metric.compute()
            self.log('test/miou', results['miou'], sync_dist=True)
            self.log('test/occ_iou', results['occ_iou'], sync_dist=True)

            # Log per-class IoU
            for i, class_name in enumerate(self.occ_metric.class_names):
                if i != self.occ_metric.ignore_index:
                    self.log(f'test/iou_{class_name}', results[f'iou_{class_name}'], sync_dist=True)

            # Print formatted table with recall/precision on rank 0
            if self.global_rank == 0:
                hist = self.occ_metric.hist
                EPS = 1e-6

                # Per-class metrics
                tp = torch.diag(hist)
                fp = hist.sum(dim=0) - tp
                fn = hist.sum(dim=1) - tp
                iou = tp / (tp + fp + fn + EPS)
                recall = tp / (tp + fn + EPS)
                precision = tp / (tp + fp + EPS)

                # Occ metrics (official formula)
                free_idx = self.occ_metric.ignore_index
                tp_occ = hist[:free_idx, :free_idx].sum()
                total_occ = hist.sum() - hist[free_idx, free_idx]
                occ_iou = tp_occ / (total_occ + EPS)
                gt_occ = hist[:free_idx, :].sum()
                pred_occ = hist[:, :free_idx].sum()
                occ_recall = tp_occ / (gt_occ + EPS)
                occ_precision = tp_occ / (pred_occ + EPS)

                print("=" * 80)
                print("Test Results")
                print("=" * 80)
                print(f"  mIoU:       {results['miou']:.4f}")
                print(f"  Occ IoU:    {occ_iou:.4f}")
                print(f"  Occ Recall: {occ_recall:.4f}")
                print(f"  Occ Prec:   {occ_precision:.4f}")
                print("-" * 80)
                print(f"{'Class':<25} | {'IoU':<8} | {'Recall':<8} | {'Precision':<8}")
                print("-" * 80)
                for i, name in enumerate(self.occ_metric.class_names[:-1]):  # Exclude 'free'
                    print(f"{name:<25} | {iou[i]:.4f}   | {recall[i]:.4f}   | {precision[i]:.4f}")
                print("=" * 80)

            self.occ_metric.reset()

    def configure_optimizers(self):
        """Configure optimizers and schedulers."""
        optimizer_type = getattr(self.hparams, 'optimizer', 'adamw').lower()
        betas = getattr(self.hparams, 'betas', (0.9, 0.999))

        if optimizer_type == 'lion':
            # Lion optimizer (Google 2023) - faster convergence, lower memory
            # Recommended: lr 3-10x lower than AdamW, weight_decay 3-10x higher
            try:
                from lion_pytorch import Lion
                optimizer = Lion(
                    self.parameters(),
                    lr=self.learning_rate,
                    weight_decay=self.weight_decay,
                    betas=(betas[0], 0.99),  # Lion uses different beta2
                )
                print(f"Using Lion optimizer (lr={self.learning_rate}, wd={self.weight_decay})")
            except ImportError:
                print("lion-pytorch not installed, falling back to AdamW. Install: pip install lion-pytorch")
                optimizer = torch.optim.AdamW(
                    self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay, betas=betas
                )
        elif optimizer_type == 'adam8bit':
            # 8-bit Adam for memory efficiency (~50% less optimizer memory)
            try:
                import bitsandbytes as bnb
                optimizer = bnb.optim.AdamW8bit(
                    self.parameters(),
                    lr=self.learning_rate,
                    weight_decay=self.weight_decay,
                    betas=betas,
                )
                print(f"Using 8-bit AdamW (lr={self.learning_rate})")
            except ImportError:
                print("bitsandbytes not installed, falling back to AdamW. Install: pip install bitsandbytes")
                optimizer = torch.optim.AdamW(
                    self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay, betas=betas
                )
        else:
            # Default: AdamW
            optimizer = torch.optim.AdamW(
                self.parameters(),
                lr=self.learning_rate,
                weight_decay=self.weight_decay,
                betas=betas,
            )

        # Auto-calculate steps_per_epoch from trainer if not specified
        if self._steps_per_epoch is not None:
            steps_per_epoch = self._steps_per_epoch
        else:
            # Get from trainer's estimated stepping batches
            steps_per_epoch = self.trainer.estimated_stepping_batches // self.trainer.max_epochs
            print(f"Auto-calculated steps_per_epoch: {steps_per_epoch}")

        total_steps = steps_per_epoch * self.trainer.max_epochs
        lr_schedule = getattr(self.hparams, 'lr_schedule', 'cosine')
        min_lr_ratio = getattr(self.hparams, 'min_lr_ratio', 0.01)  # Configurable min LR

        if lr_schedule == 'cosine':
            # Cosine Annealing with Warmup
            # - Linear warmup for warmup_iters steps
            # - Cosine decay to min_lr after warmup
            import math

            def lr_lambda(step):
                if step < self.warmup_iters:
                    # Linear warmup
                    return self.warmup_factor + (1 - self.warmup_factor) * step / self.warmup_iters
                else:
                    # Cosine annealing
                    progress = (step - self.warmup_iters) / max(1, total_steps - self.warmup_iters)
                    return min_lr_ratio + 0.5 * (1 - min_lr_ratio) * (1 + math.cos(math.pi * progress))

        elif lr_schedule == 'onecycle':
            # OneCycleLR - fast convergence, good for limited epochs
            # Ramps up to max_lr then down, with momentum annealing
            import math
            pct_start = self.warmup_iters / total_steps  # Warmup portion

            def lr_lambda(step):
                pct = step / total_steps
                if pct < pct_start:
                    # Warmup phase: ramp up
                    return self.warmup_factor + (1 - self.warmup_factor) * (pct / pct_start)
                else:
                    # Annealing phase: cosine down to min_lr
                    progress = (pct - pct_start) / (1 - pct_start)
                    return min_lr_ratio + 0.5 * (1 - min_lr_ratio) * (1 + math.cos(math.pi * progress))
        else:
            # MultiStepLR scheduler (original MMEngine config)
            # - Linear warmup for warmup_iters steps
            # - Step decay at milestones
            def lr_lambda(step):
                if step < self.warmup_iters:
                    # Linear warmup
                    return self.warmup_factor + (1 - self.warmup_factor) * step / self.warmup_iters
                else:
                    # MultiStepLR decay (convert epoch milestones to steps)
                    decay = 1.0
                    for milestone in self.lr_milestones:
                        if step >= milestone * steps_per_epoch:
                            decay *= self.lr_gamma
                    return decay

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'step'
            }
        }

    def pre_transformer(
        self,
        mlvl_feats: List[torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """Prepare inputs for transformer decoder.

        Args:
            mlvl_feats: Multi-level features from FPN.

        Returns:
            Dictionary of decoder inputs.
        """
        batch_size = mlvl_feats[0].size(0)

        feat_flatten = []
        spatial_shapes = []

        for lvl, feat in enumerate(mlvl_feats):
            bs, c, h, w = feat.shape
            spatial_shape = torch.tensor([h, w], device=feat.device)
            # [bs, c, h, w] -> [bs, h*w, c]
            feat = feat.view(bs, c, -1).permute(0, 2, 1)
            feat_flatten.append(feat)
            spatial_shapes.append(spatial_shape)

        # Concatenate all levels
        feat_flatten = torch.cat(feat_flatten, dim=1)
        spatial_shapes = torch.stack(spatial_shapes)

        level_start_index = torch.cat((
            spatial_shapes.new_zeros((1,)),
            spatial_shapes.prod(1).cumsum(0)[:-1]
        ))

        valid_ratios = feat_flatten.new_ones(batch_size, len(mlvl_feats), 2)

        return {
            'memory_mask': None,
            'spatial_shapes': spatial_shapes,
            'level_start_index': level_start_index,
            'valid_ratios': valid_ratios
        }

    def pre_decoder(
        self,
        memory: torch.Tensor,
        batch_size: int,
        sem_segs: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """Prepare query embeddings and reference points.

        Args:
            memory: Encoder memory [B*N, L, C].
            batch_size: Batch size.
            sem_segs: Semantic segmentation masks [B, N, H, W] for sky filtering.
                      If provided, reference points will be sampled from non-sky regions.

        Returns:
            Dictionary with query and reference points.
        """
        bs = memory.size(0)
        c = memory.size(-1)

        query = self.query_embeds.weight.unsqueeze(0).expand(bs, -1, -1)
        num_queries = query.size(1)

        # Sample reference points, avoiding sky regions if sem_segs provided
        if sem_segs is not None:
            # sem_segs: [B, N, H, W] -> flatten to [B*N, H, W]
            sem_segs_flat = sem_segs.flatten(0, 1)
            h, w = sem_segs_flat.shape[-2:]

            reference_points = []
            for i in range(bs):
                # Create mask of valid (non-sky) pixels
                # Sky class is 17 in SAM3
                valid_mask = (sem_segs_flat[i] != 17)
                valid_indices = valid_mask.nonzero(as_tuple=False)  # [num_valid, 2] (row, col)

                if len(valid_indices) >= num_queries:
                    # Sample from valid regions
                    perm = torch.randperm(len(valid_indices), device=memory.device)[:num_queries]
                    sampled_indices = valid_indices[perm]  # [num_queries, 2]
                    # Convert to normalized coordinates [0, 1]
                    # Reference points are (x, y) = (col/w, row/h)
                    ref_pts = torch.stack([
                        sampled_indices[:, 1].float() / w,  # x = col / width
                        sampled_indices[:, 0].float() / h,  # y = row / height
                    ], dim=-1)
                else:
                    # Not enough valid pixels, fall back to random sampling
                    # but bias toward valid regions
                    ref_pts = torch.rand((num_queries, 2), device=memory.device)
                    if len(valid_indices) > 0:
                        # Replace first len(valid_indices) points with valid ones
                        perm = torch.randperm(len(valid_indices), device=memory.device)
                        sampled_indices = valid_indices[perm]
                        ref_pts[:len(valid_indices)] = torch.stack([
                            sampled_indices[:, 1].float() / w,
                            sampled_indices[:, 0].float() / h,
                        ], dim=-1)

                reference_points.append(ref_pts)

            reference_points = torch.stack(reference_points)  # [B*N, num_queries, 2]
        else:
            # Fallback to uniform random sampling
            reference_points = torch.rand((bs, num_queries, 2), device=query.device)

        return {
            'query': query,
            'memory': memory,
            'reference_points': reference_points
        }

    def forward_decoder(
        self,
        query: torch.Tensor,
        memory: torch.Tensor,
        memory_mask: Optional[torch.Tensor],
        reference_points: torch.Tensor,
        spatial_shapes: torch.Tensor,
        level_start_index: torch.Tensor,
        valid_ratios: torch.Tensor,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """Forward through decoder.

        Args:
            query: Query embeddings [B, num_queries, embed_dims].
            memory: Encoder memory [B, num_feat_points, embed_dims].
            memory_mask: Memory padding mask.
            reference_points: Initial reference points [B, num_queries, 2].
            spatial_shapes: Spatial shapes of each level.
            level_start_index: Start index for each level.
            valid_ratios: Valid ratios for each level.

        Returns:
            Dictionary with hidden states and references.
        """
        inter_states, references = self.decoder(
            query=query,
            value=memory,
            key_padding_mask=memory_mask,
            reference_points=reference_points,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            valid_ratios=valid_ratios,
            **kwargs
        )

        return {
            'hidden_states': inter_states,
            'references': list(references)
        }

    @classmethod
    def from_mmengine_checkpoint(
        cls,
        checkpoint_path: str,
        config: Optional[Dict] = None,
        strict: bool = False
    ) -> "GaussTRLightning":
        """Load model from MMEngine checkpoint.

        Args:
            checkpoint_path: Path to MMEngine checkpoint.
            config: Model configuration. If None, uses default FeatUp config.
            strict: Whether to strictly enforce state dict matching.

        Returns:
            GaussTRLightning model with loaded weights.
        """
        # Load checkpoint (weights_only=False needed for MMEngine checkpoints)
        ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        state_dict = ckpt.get('state_dict', ckpt)

        # Remove 'model.' prefix if present
        state_dict = {
            k.replace('model.', '').replace('gauss_heads.', 'gauss_heads.'):
            v for k, v in state_dict.items()
        }

        # Create model with default config if not provided
        if config is None:
            model = cls()
        else:
            model = cls(**config)

        # Load state dict
        missing, unexpected = model.load_state_dict(state_dict, strict=strict)
        if missing:
            print(f"Missing keys: {missing}")
        if unexpected:
            print(f"Unexpected keys: {unexpected}")

        return model
