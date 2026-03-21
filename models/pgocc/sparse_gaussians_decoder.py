"""SparseGaussiansDecoder: progressive 3-layer Gaussian densification decoder."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .sparsebev_transformer import SparseBEVSelfAttention, SparseBEVSampling, AdaptiveMixing, FFN
from .bbox_utils import encode_bbox
from .gaussian_prediction import GaussianPrediction
from .loss_utils import BackprojectDepth
from ..pgocc.pointops import farthest_point_sampling
from .render import prepare_gs_attribute, batch_splatting_render


def index2point(coords, pc_range, voxel_size):
    coords = coords * voxel_size
    coords = coords + torch.tensor(pc_range[:3], device=coords.device)
    return coords


def point2bbox(coords, box_size, quaternion_params=None):
    if isinstance(box_size, (float, int)):
        wlh = torch.ones_like(coords.float()) * box_size
    elif box_size.dim() == 3:
        wlh = box_size
    else:
        raise ValueError(f"Invalid box_size dimension: {box_size.dim()}")
    if quaternion_params is not None:
        return torch.cat([coords, wlh, quaternion_params], dim=-1)
    else:
        return torch.cat([coords, wlh], dim=-1)


class SparseGaussiansDecoder(nn.Module):
    """Progressive 3-layer decoder: coarse (4000) → medium (1000) → fine (1000)."""

    def __init__(self,
                 embed_dims=256,
                 num_frames=8,
                 num_points=4,
                 num_groups=4,
                 num_levels=4,
                 num_cams=5,
                 pc_range=None,
                 render_conf=None,
                 num_queries=None,
                 ov_dim=768,
                 restrict_xyz=True,
                 use_anisotropy_encoding=True,
                 scale_range=(0.0, 2.0),
                 use_hard_mask=True,
                 num_motion_frames=2):
        super().__init__()
        self.scale_range = scale_range
        self.use_hard_mask = use_hard_mask
        self.num_motion_frames = num_motion_frames

        self.embed_dims = embed_dims
        self.num_frames = num_frames
        self.num_cams = num_cams
        self.pc_range = pc_range or [-40, -40, -1, 40, 40, 5.4]
        self.render_conf = render_conf or {}
        self.num_queries = num_queries or [4000, 1000, 1000]
        self.use_anisotropy_encoding = use_anisotropy_encoding

        total_queries = sum(self.num_queries)
        self.query_embeds = nn.Embedding(total_queries, embed_dims)

        h = self.render_conf.get('render_h', 180)
        w = self.render_conf.get('render_w', 320)
        self.backproject_depth = BackprojectDepth(num_cams, h, w)

        if restrict_xyz:
            unit_xyz = [4.0, 4.0, 0.32]
            self.unit_sigmoid = [
                8 * unit_xyz[i] / (self.pc_range[i + 3] - self.pc_range[i])
                for i in range(3)
            ]

        self.decoder_layers = nn.ModuleList()
        self.gau_pred_heads = nn.ModuleList()
        self.ov_heads = nn.ModuleList()

        self.layers_scales = ['coarse', 'medium', 'fine']
        for i in range(len(self.layers_scales)):
            self.decoder_layers.append(SparseGaussiansDecoderLayer(
                embed_dims=embed_dims,
                num_frames=num_frames,
                num_points=num_points,
                num_groups=num_groups,
                num_levels=num_levels,
                num_cams=num_cams,
                pc_range=self.pc_range,
                self_attn=True,
                past_queries=sum(self.num_queries[:i]),
                use_anisotropy_encoding=use_anisotropy_encoding,
            ))

            self.gau_pred_heads.append(nn.Sequential(
                nn.Linear(embed_dims, embed_dims * 4),
                nn.ReLU(inplace=True),
                nn.Linear(embed_dims * 4, 11),
            ))

            if self.render_conf.get('use_ov', True):
                self.ov_heads.append(nn.ModuleList([
                    nn.Sequential(
                        nn.Linear(embed_dims, embed_dims * 4),
                        nn.ReLU(inplace=True),
                    ),
                    nn.Sequential(
                        nn.Linear(embed_dims * 4, embed_dims * 4),
                        nn.ReLU(inplace=True),
                    ),
                    nn.Sequential(
                        nn.Linear(embed_dims * 4, ov_dim),
                    ),
                ]))

        # Phase 2: static/dynamic branch head (SelfOccFlow-inspired)
        # Predicts per-Gaussian routing: p_static + p_dynamic = 1
        self.branch_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(embed_dims, embed_dims),
                nn.ReLU(inplace=True),
                nn.Linear(embed_dims, 2),
            )
            for _ in range(len(self.layers_scales))
        ])
        # Freeze branch_heads when not supervised — random params waste optimizer capacity
        if not use_hard_mask:
            for p in self.branch_heads.parameters():
                p.requires_grad = False

        # Phase 4: motion head — predicts per-query XY offsets to past frames
        # Only on finest layer. Output: [B, Q, 2*num_motion_frames]
        # Zero-init output layer so motion starts at zero (no motion = static)
        self.motion_head = nn.Sequential(
            nn.Linear(embed_dims, embed_dims),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dims, 2 * num_motion_frames),
        )
        # Zero-init: start with no predicted motion (safe initialization)
        nn.init.zeros_(self.motion_head[-1].weight)
        nn.init.zeros_(self.motion_head[-1].bias)

    @torch.no_grad()
    def init_weights(self):
        for layer in self.decoder_layers:
            layer.init_weights()

    def query_2_gaussian(self, gau_pred, scale_range=None):
        """Map 11D predictions to Gaussian parameters.

        Args:
            gau_pred: [B, Q, 11] raw predictions
            scale_range: (min, max) scale range. Default: self.scale_range
        """
        if scale_range is None:
            scale_range = self.scale_range
        gau_pred = torch.nan_to_num(gau_pred, nan=0.0)

        gau_xyz_sigmoid = 2 * torch.sigmoid(gau_pred[..., 0:3]) - 1
        gau_xyz_delta = torch.stack([
            gau_xyz_sigmoid[..., 0] * self.unit_sigmoid[0],
            gau_xyz_sigmoid[..., 1] * self.unit_sigmoid[1],
            gau_xyz_sigmoid[..., 2] * self.unit_sigmoid[2],
        ], dim=-1)
        gau_xyz_delta = torch.nan_to_num(gau_xyz_delta, nan=0.0)

        gau_rots = F.normalize(gau_pred[..., 3:7], dim=-1)
        gau_rots = torch.nan_to_num(gau_rots, nan=0.0)

        gau_scales = torch.sigmoid(gau_pred[..., 7:10]) * (scale_range[1] - scale_range[0]) + scale_range[0]
        gau_scales = torch.nan_to_num(gau_scales, nan=0.0)

        gau_opacities = torch.sigmoid(gau_pred[..., 10:11]).squeeze(-1)
        gau_opacities = torch.nan_to_num(gau_opacities, nan=0.5)

        return dict(
            delta_xyz=gau_xyz_delta,
            gau_rots=gau_rots,
            gau_scales=gau_scales,
            gau_opacities=gau_opacities,
        )

    def forward(self, mlvl_feats, img_metas, depth=None):
        B = len(img_metas)
        gau_preds = []
        N = self.num_cams

        query_feat = self.query_embeds.weight.unsqueeze(0)

        # --- Depth-guided query initialization via FPS ---
        if depth is not None:
            depth = depth[0]  # [N, 1, H, W]
            depth = depth.clamp(max=51.2)
            mask = ((depth > 0) & (depth < 80)).squeeze(1)

            # Exclude blind region (no backbone features) from query initialization.
            # Without this, ~43% of queries are placed where they never get gradient.
            valid_row = img_metas[0].get('backbone_valid_row', 0)
            if valid_row > 0:
                mask[:, :valid_row, :] = False

            render_k, cam2ego, W2C = prepare_gs_attribute(img_metas, num_cams=N)
            inv_k = torch.inverse(render_k)

            # Move backproject_depth buffers to correct device
            self.backproject_depth = self.backproject_depth.to(depth.device)
            cam_points = self.backproject_depth(depth, inv_k)

            ego_points = cam2ego @ cam_points
            ego_points = ego_points[..., 0:3, :]
            ego_points = ego_points.reshape(
                N, 3, self.render_conf['render_h'], self.render_conf['render_w']
            ).permute(0, 2, 3, 1)

            roi_mask = (
                (ego_points[..., 0] > self.pc_range[0]) & (ego_points[..., 0] < self.pc_range[3]) &
                (ego_points[..., 1] > self.pc_range[1]) & (ego_points[..., 1] < self.pc_range[4]) &
                (ego_points[..., 2] > self.pc_range[2]) & (ego_points[..., 2] < self.pc_range[5])
            )

            ego_points_coarse = ego_points[mask]
            if ego_points_coarse.numel() == 0:
                ego_points_coarse = torch.randn(1000, 3, device=ego_points.device) * 10.0

            ego_points_coarse = torch.nan_to_num(ego_points_coarse, nan=0.0)

            total_queries = sum(self.num_queries)
            try:
                selected_ego_points = farthest_point_sampling(
                    ego_points_coarse.float(),
                    torch.tensor([ego_points_coarse.shape[0]], device=ego_points_coarse.device, dtype=torch.int),
                    torch.tensor([total_queries], device=ego_points_coarse.device, dtype=torch.int),
                )
                selected_ego_points_coarse = selected_ego_points[:self.num_queries[0]]
                query_coord = ego_points_coarse[selected_ego_points_coarse].reshape(B, -1, 3)
            except Exception:
                indices = torch.randperm(ego_points_coarse.shape[0], device=ego_points_coarse.device)[:self.num_queries[0]]
                query_coord = ego_points_coarse[indices].reshape(B, -1, 3)

        # --- Progressive decoder layers ---
        prev_branch_probs = None  # passed from layer i to layer i+1 for temporal masking (only when use_hard_mask=True)
        for i, layer in enumerate(self.decoder_layers):
            # Refine depth residual mask for progressive densification
            if i != 0:
                if 'res_depths_mask' not in locals():
                    res_depths_mask = torch.ones_like(mask, dtype=torch.bool)
                mask = mask & roi_mask & res_depths_mask

            # Medium layer: sample new query coords
            if i == 1:
                if mask.sum().item() >= self.num_queries[1]:
                    ego_points_medium = torch.nan_to_num(ego_points[mask], nan=0.0)
                    try:
                        sel = farthest_point_sampling(
                            ego_points_medium.float(),
                            torch.tensor([ego_points_medium.shape[0]], device=ego_points_medium.device, dtype=torch.int),
                            torch.tensor([self.num_queries[1]], device=ego_points_medium.device, dtype=torch.int),
                        )
                        query_coord_medium = ego_points_medium[sel].reshape(B, -1, 3)
                    except Exception:
                        indices = torch.randperm(ego_points_medium.shape[0], device=ego_points_medium.device)[:self.num_queries[1]]
                        query_coord_medium = ego_points_medium[indices].reshape(B, -1, 3)
                else:
                    sel = selected_ego_points[self.num_queries[0]:self.num_queries[0] + self.num_queries[1]]
                    query_coord_medium = ego_points_coarse[sel].reshape(B, -1, 3)

            # Fine layer: sample new query coords
            elif i == 2:
                mask = mask & res_depths_mask & roi_mask
                if mask.sum().item() >= self.num_queries[2]:
                    ego_points_fine = ego_points[mask]
                    sel = farthest_point_sampling(
                        ego_points_fine.float(),
                        torch.tensor([ego_points_fine.shape[0]], device=ego_points_fine.device, dtype=torch.int),
                        torch.tensor([self.num_queries[2]], device=ego_points_fine.device, dtype=torch.int),
                    )
                    query_coord_fine = ego_points_fine[sel].reshape(B, -1, 3)
                else:
                    start = self.num_queries[0] + self.num_queries[1]
                    sel = selected_ego_points[start:start + self.num_queries[2]]
                    query_coord_fine = ego_points_coarse[sel].reshape(B, -1, 3)
            else:
                query_coord_medium = None
                query_coord_fine = None

            # --- Build query bbox (encode coords + box size) ---
            q0 = self.num_queries[0]
            q1 = self.num_queries[1]

            query_bbox = point2bbox(query_coord[:, :q0], box_size=1.6)
            query_bbox = encode_bbox(query_bbox, pc_range=self.pc_range)

            if i == 1:
                query_bbox_medium = encode_bbox(
                    point2bbox(query_coord_medium, box_size=0.8), pc_range=self.pc_range)
                query_bbox = torch.cat([query_bbox, query_bbox_medium], dim=1)
                query_coord = torch.cat([query_coord, query_coord_medium], dim=1)

            if i == 2:
                query_coord = torch.cat([query_coord, query_coord_fine], dim=1)
                qc_med = query_coord[:, q0:q0 + q1]
                query_bbox_medium = encode_bbox(
                    point2bbox(qc_med, box_size=0.8), pc_range=self.pc_range)
                query_bbox_fine = encode_bbox(
                    point2bbox(query_coord_fine, box_size=0.4), pc_range=self.pc_range)
                query_bbox = torch.cat([query_bbox, query_bbox_medium, query_bbox_fine], dim=1)

            # Anisotropy disabled to match original runtime behavior
            anisotropy_info = None

            query_feat_part = query_feat[:, :query_bbox.size(1)]
            query_feat_part = layer(query_feat_part, query_bbox, mlvl_feats, anisotropy_info, img_metas,
                                    prev_branch_probs=prev_branch_probs if self.use_hard_mask else None)

            gau_pred = self.gau_pred_heads[i](query_feat_part)
            all_scales, all_rots, all_opacities = [], [], []

            if i == 0:
                gaussian = self.query_2_gaussian(gau_pred)
                query_coord = gaussian['delta_xyz'] + query_coord
                all_scales.append(gaussian['gau_scales'])
                all_rots.append(gaussian['gau_rots'])
                all_opacities.append(gaussian['gau_opacities'])
            elif i == 1:
                gaussian = self.query_2_gaussian(gau_pred[:, :q0])
                query_coord[:, :q0] = (
                    gaussian['delta_xyz'] + query_coord[:, :q0]
                )
                all_scales.append(gaussian['gau_scales'])
                all_rots.append(gaussian['gau_rots'])
                all_opacities.append(gaussian['gau_opacities'])

                gaussian_medium = self.query_2_gaussian(gau_pred[:, q0:q0 + q1])
                query_coord[:, q0:q0 + q1] = (
                    gaussian_medium['delta_xyz'] / 2
                    + query_coord[:, q0:q0 + q1]
                )
                all_scales.append(gaussian_medium['gau_scales'])
                all_rots.append(gaussian_medium['gau_rots'])
                all_opacities.append(gaussian_medium['gau_opacities'])
            elif i == 2:
                gaussian = self.query_2_gaussian(gau_pred[:, :q0])
                query_coord[:, :q0] = (
                    gaussian['delta_xyz'] + query_coord[:, :q0]
                )
                all_scales.append(gaussian['gau_scales'])
                all_rots.append(gaussian['gau_rots'])
                all_opacities.append(gaussian['gau_opacities'])

                gaussian_medium = self.query_2_gaussian(gau_pred[:, q0:q0 + q1])
                query_coord[:, q0:q0 + q1] = (
                    gaussian_medium['delta_xyz'] / 2
                    + query_coord[:, q0:q0 + q1]
                )
                all_scales.append(gaussian_medium['gau_scales'])
                all_rots.append(gaussian_medium['gau_rots'])
                all_opacities.append(gaussian_medium['gau_opacities'])

                gaussian_fine = self.query_2_gaussian(gau_pred[:, q0 + q1:])
                query_coord[:, q0 + q1:] = (
                    gaussian_fine['delta_xyz'] / 4
                    + query_coord[:, q0 + q1:]
                )
                all_scales.append(gaussian_fine['gau_scales'])
                all_rots.append(gaussian_fine['gau_rots'])
                all_opacities.append(gaussian_fine['gau_opacities'])

            merged_scales = torch.cat(all_scales, dim=1)
            merged_rots = torch.cat(all_rots, dim=1)
            merged_opacities = torch.cat(all_opacities, dim=1)

            # OV feature prediction
            if self.render_conf.get('use_ov', True) and len(self.ov_heads) > i:
                ov_query_feat = query_feat_part
                for ov_layer in self.ov_heads[i]:
                    ov_query_feat = ov_layer(ov_query_feat)
            else:
                ov_query_feat = None

            # Phase 2: predict static/dynamic branch routing
            # Detach features — branch classification must NOT corrupt shared features
            b_logits = self.branch_heads[i](query_feat_part.detach())  # [B, Q, 2]
            b_probs = torch.softmax(b_logits, dim=-1)  # [B, Q, 2]
            prev_branch_probs = b_probs  # feed to next layer for temporal masking

            # Phase 4: motion head (finest layer only)
            # Detach to prevent motion gradients from corrupting shared decoder features
            motion_offsets = None
            if i == len(self.layers_scales) - 1:
                motion_offsets = self.motion_head(query_feat_part.detach())  # [B, Q, 2*P]

            pred_gaussians = GaussianPrediction(
                means=query_coord,
                scales=merged_scales,
                rotations=merged_rots,
                opacities=merged_opacities,
                ovs=ov_query_feat,
                colors=None,
                branch_logits=b_logits,
                branch_probs=b_probs,
                motion_offsets=motion_offsets,
            )
            gau_preds.append(pred_gaussians)

            # --- Render depth for residual mask (progressive densification) ---
            with torch.no_grad():
                render_results = batch_splatting_render(
                    pred_gaussians, W2C, render_k, render_conf=self.render_conf, inference=True)
                render_depths = render_results['depth']
                render_depths = torch.nan_to_num(render_depths, nan=0.0)
                render_depths = render_depths.permute(0, 3, 1, 2)
                try:
                    depth_diff = render_depths - depth
                    depth_diff = torch.nan_to_num(depth_diff, nan=0.0)
                    res_depths_mask = (depth_diff > 0.2).squeeze(1)
                    if res_depths_mask.shape != mask.shape:
                        res_depths_mask = torch.ones_like(mask, dtype=torch.bool)
                except Exception:
                    res_depths_mask = torch.ones_like(mask, dtype=torch.bool)

        return gau_preds


class SparseGaussiansDecoderLayer(nn.Module):
    """Single decoder layer: position encoding → self-attn → sampling → mixing → FFN."""

    def __init__(self, embed_dims=256, num_frames=8, num_points=4, num_groups=4,
                 num_levels=4, num_cams=5, pc_range=None, self_attn=True,
                 past_queries=None, use_anisotropy_encoding=True):
        super().__init__()

        self.position_encoder = nn.Sequential(
            nn.Linear(3, embed_dims),
            nn.LayerNorm(embed_dims),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dims, embed_dims),
            nn.LayerNorm(embed_dims),
            nn.ReLU(inplace=True),
        )

        if self_attn:
            self.self_attn = SparseBEVSelfAttention(
                embed_dims, num_heads=8, dropout=0.1,
                pc_range=pc_range, scale_adaptive=True, past_queries=past_queries)
            self.norm1 = nn.LayerNorm(embed_dims)
        else:
            self.self_attn = None

        self.num_frames = num_frames
        self.num_points = num_points

        self.sampling = SparseBEVSampling(
            embed_dims=embed_dims,
            num_frames=num_frames,
            num_groups=num_groups,
            num_points=num_points,
            num_levels=num_levels,
            num_cams=num_cams,
            pc_range=pc_range,
            use_anisotropy_encoding=use_anisotropy_encoding,
        )

        # Temporal gate — DISABLED at runtime but kept for checkpoint compat.
        # Drifts from identity during training and corrupts temporal features.
        self.temporal_gate = nn.Sequential(
            nn.Linear(embed_dims, embed_dims // 4),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dims // 4, num_frames),
        )
        for p in self.temporal_gate.parameters():
            p.requires_grad = False

        self.mixing = AdaptiveMixing(
            in_dim=embed_dims,
            in_points=num_points * num_frames,
            n_groups=num_groups,
            out_points=num_points * num_frames * num_groups,
        )

        self.ffn = FFN(embed_dims, feedforward_channels=embed_dims * 2, ffn_drop=0.1)

        self.norm2 = nn.LayerNorm(embed_dims)
        self.norm3 = nn.LayerNorm(embed_dims)

    @torch.no_grad()
    def init_weights(self):
        if self.self_attn is not None:
            self.self_attn.init_weights()
        self.sampling.init_weights()
        self.mixing.init_weights()
        self.ffn.init_weights()
        # Zero-init temporal gate → uniform weights at start (no behavior change)
        nn.init.zeros_(self.temporal_gate[-1].weight)
        nn.init.zeros_(self.temporal_gate[-1].bias)

    def forward(self, query_feat, query_3dgs, mlvl_feats, anisotropy_info, img_metas,
                prev_branch_probs=None):
        query_pos = self.position_encoder(query_3dgs[..., :3])
        query_feat = query_feat + query_pos
        if self.self_attn is not None:
            query_feat = self.norm1(self.self_attn(query_3dgs, query_feat))

        sampled_feat = self.sampling(query_3dgs, query_feat, mlvl_feats, anisotropy_info, img_metas)
        # sampled_feat: [B, Q, G, T*P, C]

        B, Q, G, TP, C = sampled_feat.shape
        T, P = self.num_frames, self.num_points

        # Hard branch-conditioned temporal masking (SelfOccFlow-inspired):
        # Dynamic queries: replace past-frame features with current-frame copies
        #   → AdaptiveMixing sees consistent features from frame 0 only
        # Static queries: keep all temporal frames for multi-frame consistency
        # prev_branch_probs comes from the PREVIOUS decoder layer's branch_head.
        if prev_branch_probs is not None and T > 1:
            sampled_feat = sampled_feat.view(B, Q, G, T, P, C)

            # prev_branch_probs may cover fewer queries than current layer
            n_prev = min(prev_branch_probs.size(1), Q)
            dynamic_prob = torch.zeros(B, Q, device=sampled_feat.device)
            dynamic_prob[:, :n_prev] = prev_branch_probs[:, :n_prev, 1].detach()

            current_frame = sampled_feat[:, :, :, 0:1, :, :]  # [B, Q, G, 1, P, C]
            current_expanded = current_frame.expand_as(sampled_feat)

            # Blend: past frames interpolate toward current frame based on dynamic_prob
            dp = dynamic_prob[:, :, None, None, None, None]  # [B, Q, 1, 1, 1, 1]
            past_mask = torch.zeros(1, 1, 1, T, 1, 1, device=sampled_feat.device)
            past_mask[:, :, :, 1:, :, :] = 1.0  # 1.0 for past frames only

            sampled_feat = sampled_feat * (1 - past_mask * dp) + current_expanded * (past_mask * dp)
            sampled_feat = sampled_feat.reshape(B, Q, G, TP, C)
            self._last_dynamic_prob_mean = dynamic_prob.mean().detach()
        else:
            self._last_dynamic_prob_mean = None

        # Temporal gate DISABLED: it drifts from identity during training and
        # disrupts converged temporal features. See fixv36a trend analysis.
        # Module kept in __init__ for checkpoint compatibility only.
        self._last_gate = None

        query_feat = self.norm2(self.mixing(sampled_feat, query_feat))
        query_feat = self.norm3(self.ffn(query_feat))
        return query_feat
