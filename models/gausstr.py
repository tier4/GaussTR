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
        head_patch_size: int = 16,
        head_depth_limit: float = 51.2,
        head_text_protos: Optional[str] = "ckpts/text_proto_embeds_clip.pth",
        head_prompt_denoising: bool = True,
        head_num_segment_classes: int = 17,
        # Voxelizer config
        vol_range: List[float] = None,
        voxel_size: float = 0.4,
        # Training config
        learning_rate: float = 2e-4,
        weight_decay: float = 5e-3,
        warmup_iters: int = 200,
        warmup_factor: float = 1e-3,
        lr_milestones: List[int] = None,
        lr_gamma: float = 0.1,
        steps_per_epoch: int = None,  # Auto-calculated from dataloader if not specified
        gradient_clip_val: float = 35.0,
        # Data preprocessor config
        mean: List[float] = None,
        std: List[float] = None,
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
            'patch_size': head_patch_size,
            'depth_limit': head_depth_limit,
            'text_protos': head_text_protos,
            'prompt_denoising': head_prompt_denoising,
            'num_segment_classes': head_num_segment_classes,
            'voxelizer_cfg': {
                'vol_range': vol_range,
                'voxel_size': voxel_size
            }
        }
        self.gauss_heads = nn.ModuleList([
            GaussTRHead(**head_cfg) for _ in range(decoder_num_layers)
        ])

    def forward(
        self,
        images: torch.Tensor,
        feats: torch.Tensor,
        depth: torch.Tensor,
        cam2img: torch.Tensor,
        cam2ego: torch.Tensor,
        img_aug_mat: Optional[torch.Tensor] = None,
        _profile: bool = False,
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
            _profile: Enable detailed profiling.

        Returns:
            Occupancy predictions.
        """
        import time

        if _profile:
            torch.cuda.synchronize()
            t0 = time.time()

        bs, n = images.shape[:2]

        # Use pre-extracted features
        x = feats.flatten(0, 1)

        # Multi-scale features
        if _profile:
            torch.cuda.synchronize()
            t1 = time.time()
        multi_scale_feats = self.neck(x)
        if _profile:
            torch.cuda.synchronize()
            t2 = time.time()
            print(f"    [Forward] Neck: {(t2-t1)*1000:.1f}ms")

        # Prepare decoder inputs
        decoder_inputs = self.pre_transformer(multi_scale_feats)
        feat_flatten = flatten_multi_scale_feats(multi_scale_feats)[0]
        decoder_inputs.update(self.pre_decoder(feat_flatten, bs))

        if _profile:
            torch.cuda.synchronize()
            t3 = time.time()
            print(f"    [Forward] Pre-decoder: {(t3-t2)*1000:.1f}ms")

        # Forward through decoder
        decoder_outputs = self.forward_decoder(
            reg_branches=[h.regress_head for h in self.gauss_heads],
            **decoder_inputs
        )

        if _profile:
            torch.cuda.synchronize()
            t4 = time.time()
            print(f"    [Forward] Decoder: {(t4-t3)*1000:.1f}ms")

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

        if _profile:
            torch.cuda.synchronize()
            t5 = time.time()
            print(f"    [Forward] Head: {(t5-t4)*1000:.1f}ms")
            print(f"    [Forward] TOTAL: {(t5-t0)*1000:.1f}ms")

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

        # Use pre-extracted features
        x = feats.flatten(0, 1)

        # Multi-scale features
        multi_scale_feats = self.neck(x)

        # Prepare decoder inputs
        decoder_inputs = self.pre_transformer(multi_scale_feats)
        feat_flatten = flatten_multi_scale_feats(multi_scale_feats)[0]
        decoder_inputs.update(self.pre_decoder(feat_flatten, bs))

        # Forward through decoder
        decoder_outputs = self.forward_decoder(
            reg_branches=[h.regress_head for h in self.gauss_heads],
            **decoder_inputs
        )

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
                mode='loss'
            )
            for k, v in layer_losses.items():
                losses[f'{k}/{i}'] = v
                total_loss += v

        # Log losses
        self.log_dict(losses, prog_bar=True, sync_dist=True)
        self.log('train_loss', total_loss, prog_bar=True, sync_dist=True)

        return total_loss

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
        # Profiling - enable for first 5 batches on rank 0
        _do_profile = batch_idx < 5 and self.global_rank == 0
        if _do_profile:
            import time
            # Enable detailed profiling in head
            self.gauss_heads[-1]._profile_enabled = True
            torch.cuda.synchronize()
            t0 = time.time()
            print(f"\n=== Batch {batch_idx} Profiling ===")
        else:
            self.gauss_heads[-1]._profile_enabled = False

        preds = self.forward(
            images=batch['images'],
            feats=batch['feats'],
            depth=batch['depth'],
            cam2img=batch['cam2img'],
            cam2ego=batch['cam2ego'],
            img_aug_mat=batch.get('img_aug_mat'),
            _profile=_do_profile
        )

        if _do_profile:
            torch.cuda.synchronize()
            t1 = time.time()

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

        if _do_profile:
            torch.cuda.synchronize()
            t2 = time.time()
            print(f"[Batch {batch_idx}] Metric: {(t2-t1)*1000:.1f}ms, TOTAL: {(t2-t0)*1000:.1f}ms")

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

            # Print formatted table only on rank 0 to avoid duplicate output in multi-GPU testing
            if self.global_rank == 0:
                print(self.occ_metric.get_table_str())
            self.occ_metric.reset()

    def configure_optimizers(self):
        """Configure optimizers and schedulers."""
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )

        # Auto-calculate steps_per_epoch from trainer if not specified
        if self._steps_per_epoch is not None:
            steps_per_epoch = self._steps_per_epoch
        else:
            # Get from trainer's estimated stepping batches
            steps_per_epoch = self.trainer.estimated_stepping_batches // self.trainer.max_epochs
            print(f"Auto-calculated steps_per_epoch: {steps_per_epoch}")

        # Warmup + MultiStepLR scheduler (matches original MMEngine config)
        # - LinearLR warmup: start_factor=1e-3, end=200 steps
        # - MultiStepLR decay: milestones=[16] epochs, gamma=0.1
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
        batch_size: int
    ) -> Dict[str, torch.Tensor]:
        """Prepare query embeddings and reference points.

        Args:
            memory: Encoder memory [B*N, L, C].
            batch_size: Batch size.

        Returns:
            Dictionary with query and reference points.
        """
        bs = memory.size(0)
        c = memory.size(-1)

        query = self.query_embeds.weight.unsqueeze(0).expand(bs, -1, -1)
        reference_points = torch.rand((bs, query.size(1), 2), device=query.device)

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
