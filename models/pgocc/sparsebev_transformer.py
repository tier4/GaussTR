"""SparseBEV transformer components: self-attention, sampling, adaptive mixing."""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as cp

from .bbox_utils import decode_bbox
from .sparsebev_sampling import sampling_4d, make_sample_points_from_bbox, make_sample_points_from_3dgs


class FFN(nn.Module):
    """Feed-Forward Network (standalone replacement for mmcv FFN)."""

    def __init__(self, embed_dims, feedforward_channels, ffn_drop=0.0):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(embed_dims, feedforward_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(ffn_drop),
            nn.Linear(feedforward_channels, embed_dims),
            nn.Dropout(ffn_drop),
        )

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x, identity=None):
        out = self.layers(x)
        if identity is None:
            identity = x
        return identity + out


class SparseBEVSelfAttention(nn.Module):
    """Scale-adaptive self-attention with distance-based tau."""

    def __init__(self, embed_dims=256, num_heads=8, dropout=0.1, pc_range=None,
                 scale_adaptive=True, past_queries=None):
        super().__init__()
        self.pc_range = pc_range or []
        self.past_queries = past_queries or 0

        self.attention = nn.MultiheadAttention(embed_dims, num_heads, dropout=dropout, batch_first=True)

        if scale_adaptive:
            self.gen_tau = nn.Linear(embed_dims, num_heads)
        else:
            self.gen_tau = None

    @torch.no_grad()
    def init_weights(self):
        if self.gen_tau is not None:
            nn.init.zeros_(self.gen_tau.weight)
            nn.init.uniform_(self.gen_tau.bias, 0.0, 2.0)

    def inner_forward(self, query_bbox, query_feat, pre_attn_mask=None):
        if self.gen_tau is not None:
            dist = self._calc_bbox_dists(query_bbox)
            tau = self.gen_tau(query_feat)
            tau = tau.permute(0, 2, 1)  # [B, H, Q]
            attn_mask = dist[:, None, :, :] * tau[..., None]  # [B, H, Q, Q]

            if pre_attn_mask is not None:
                attn_mask[:, :, pre_attn_mask] = float('-inf')
            attn_mask = attn_mask.flatten(0, 1)  # [B*H, Q, Q]

            # Block past queries from attending to future queries
            past_queries = self.past_queries
            if past_queries > 0:
                attn_mask[:, :past_queries, past_queries:] = float('-inf')
        else:
            attn_mask = None

        out, _ = self.attention(query_feat, query_feat, query_feat, attn_mask=attn_mask)
        # mmcv MultiheadAttention returns identity + out (proj_drop=0, dropout_layer=0)
        return query_feat + out

    def forward(self, query_bbox, query_feat, pre_attn_mask=None):
        if self.training and query_feat.requires_grad:
            return cp(self.inner_forward, query_bbox, query_feat, pre_attn_mask, use_reentrant=False)
        else:
            return self.inner_forward(query_bbox, query_feat, pre_attn_mask)

    @torch.no_grad()
    def _calc_bbox_dists(self, bboxes):
        centers = decode_bbox(bboxes, self.pc_range)[..., :2]  # [B, Q, 2]
        dist = []
        for b in range(centers.shape[0]):
            dist_b = torch.norm(centers[b].reshape(-1, 1, 2) - centers[b].reshape(1, -1, 2), dim=-1)
            dist.append(dist_b[None, ...])
        dist = torch.cat(dist, dim=0)
        return -dist


class SparseBEVSampling(nn.Module):
    """Multi-scale deformable sampling with learned offsets."""

    def __init__(self, embed_dims=256, num_frames=4, num_groups=4, num_points=8,
                 num_levels=4, num_cams=5, pc_range=None, use_anisotropy_encoding=True):
        super().__init__()
        self.num_frames = num_frames
        self.num_points = num_points
        self.num_groups = num_groups
        self.num_levels = num_levels
        self.num_cams = num_cams
        self.pc_range = pc_range or []
        self.use_anisotropy_encoding = use_anisotropy_encoding
        self.sampling_offset = nn.Linear(embed_dims, num_groups * num_points * 3)
        self.scale_weights = nn.Linear(embed_dims, num_groups * num_points * num_levels)

    def init_weights(self):
        nn.init.zeros_(self.sampling_offset.weight)
        bias = self.sampling_offset.bias.data.view(self.num_groups * self.num_points, 3)
        nn.init.uniform_(bias[:, 0:3], -0.5, 0.5)

    def inner_forward(self, query_bbox, query_feat, mlvl_feats, anisotropy_info, img_metas):
        B, Q = query_bbox.shape[:2]
        image_h, image_w, _ = img_metas[0]['img_shape'][0]

        sampling_offset = self.sampling_offset(query_feat)
        sampling_offset = sampling_offset.view(B, Q, self.num_groups * self.num_points, 3)

        # Use bbox-based sampling (anisotropy disabled in original code)
        sampling_points = make_sample_points_from_bbox(query_bbox, sampling_offset, self.pc_range)
        sampling_points = sampling_points.reshape(B, Q, 1, self.num_groups, self.num_points, 3)
        sampling_points = sampling_points.expand(B, Q, self.num_frames, self.num_groups, self.num_points, 3)

        scale_weights = self.scale_weights(query_feat).view(
            B, Q, self.num_groups, 1, self.num_points, self.num_levels)
        scale_weights = torch.softmax(scale_weights, dim=-1)
        scale_weights = scale_weights.expand(
            B, Q, self.num_groups, self.num_frames, self.num_points, self.num_levels)

        ego2img = img_metas[0]['ego2img']
        if ego2img.dim() == 3:
            ego2img = ego2img.unsqueeze(0)  # [T*N, 4, 4] -> [1, T*N, 4, 4]

        sampled_feats = sampling_4d(
            sampling_points, mlvl_feats, scale_weights,
            ego2img, image_h, image_w, num_cams=self.num_cams,
        )
        return sampled_feats

    def forward(self, query_bbox, query_feat, mlvl_feats, anisotropy_info, img_metas):
        if self.training and query_feat.requires_grad:
            return cp(self.inner_forward, query_bbox, query_feat, mlvl_feats,
                      anisotropy_info, img_metas, use_reentrant=False)
        else:
            return self.inner_forward(query_bbox, query_feat, mlvl_feats, anisotropy_info, img_metas)


class AdaptiveMixing(nn.Module):
    """Adaptive channel + point mixing with parameter generators."""

    def __init__(self, in_dim, in_points, n_groups=1, query_dim=None, out_dim=None, out_points=None):
        super().__init__()
        out_dim = out_dim or in_dim
        out_points = out_points or in_points
        query_dim = query_dim or in_dim

        self.query_dim = query_dim
        self.in_dim = in_dim
        self.in_points = in_points
        self.n_groups = n_groups
        self.out_dim = out_dim
        self.out_points = out_points

        self.eff_in_dim = in_dim // n_groups
        self.eff_out_dim = out_dim // n_groups

        self.m_parameters = self.eff_in_dim * self.eff_out_dim
        self.s_parameters = self.in_points * self.out_points
        self.total_parameters = self.m_parameters + self.s_parameters

        self.parameter_generator = nn.Linear(query_dim, n_groups * self.total_parameters)
        self.out_proj = nn.Linear(self.eff_out_dim * self.out_points * n_groups, query_dim)
        self.act = nn.ReLU(inplace=True)

    @torch.no_grad()
    def init_weights(self):
        nn.init.zeros_(self.parameter_generator.weight)

    def inner_forward(self, x, query):
        B, Q, G, P, C = x.shape

        params = self.parameter_generator(query)
        params = params.reshape(B * Q, G, -1)
        out = x.reshape(B * Q, G, P, C)

        M, S = params.split([self.m_parameters, self.s_parameters], 2)
        M = M.reshape(B * Q, G, self.eff_in_dim, self.eff_out_dim)
        S = S.reshape(B * Q, G, self.out_points, self.in_points)

        # Adaptive channel mixing
        out = torch.matmul(out, M)
        out = F.layer_norm(out, [out.size(-2), out.size(-1)])
        out = self.act(out)

        # Adaptive point mixing
        out = torch.matmul(S, out)
        out = F.layer_norm(out, [out.size(-2), out.size(-1)])
        out = self.act(out)

        out = out.reshape(B, Q, -1)
        out = self.out_proj(out)
        out = query + out
        return out

    def forward(self, x, query):
        if self.training and x.requires_grad:
            return cp(self.inner_forward, x, query, use_reentrant=False)
        else:
            return self.inner_forward(x, query)
