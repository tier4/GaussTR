import os
from functools import reduce

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from numpy import random
from torch.cuda.amp import autocast


class GridMask(nn.Module):
    def __init__(self, ratio=0.5, prob=0.7):
        super().__init__()
        self.ratio = ratio
        self.prob = prob

    def forward(self, x):
        if np.random.rand() > self.prob or not self.training:
            return x

        n, c, h, w = x.size()
        x = x.view(-1, h, w)
        hh = int(1.5 * h)
        ww = int(1.5 * w)

        d = np.random.randint(2, h)
        l = min(max(int(d * self.ratio + 0.5), 1), d - 1)
        mask = np.ones((hh, ww), np.uint8)
        st_h = np.random.randint(d)
        st_w = np.random.randint(d)

        for i in range(hh // d):
            s = d * i + st_h
            t = min(s + l, hh)
            mask[s:t, :] = 0

        for i in range(ww // d):
            s = d * i + st_w
            t = min(s + l, ww)
            mask[:, s:t] = 0

        mask = mask[(hh - h) // 2:(hh - h) // 2 + h, (ww - w) // 2:(ww - w) // 2 + w]
        mask = torch.tensor(mask, dtype=x.dtype, device=x.device)
        mask = 1 - mask
        mask = mask.expand_as(x)
        x = x * mask

        return x.view(n, c, h, w)


def inverse_sigmoid(x, eps=1e-5):
    x = x.clamp(min=0, max=1)
    x1 = x.clamp(min=eps)
    x2 = (1 - x).clamp(min=eps)
    return torch.log(x1 / x2)


def pad_multiple(inputs, img_metas, size_divisor=32):
    _, _, img_h, img_w = inputs.shape

    pad_h = 0 if img_h % size_divisor == 0 else size_divisor - (img_h % size_divisor)
    pad_w = 0 if img_w % size_divisor == 0 else size_divisor - (img_w % size_divisor)

    B = len(img_metas)
    N = len(img_metas[0]['ori_shape'])

    for b in range(B):
        img_metas[b]['img_shape'] = [(img_h + pad_h, img_w + pad_w, 3) for _ in range(N)]
        img_metas[b]['pad_shape'] = [(img_h + pad_h, img_w + pad_w, 3) for _ in range(N)]

    if pad_h == 0 and pad_w == 0:
        return inputs
    else:
        return F.pad(inputs, [0, pad_w, 0, pad_h], value=0)


def get_rotation_matrix(tensor):
    """Quaternion (w, x, y, z) to 3x3 rotation matrix via Hamilton product."""
    assert tensor.shape[-1] == 4

    tensor = F.normalize(tensor, dim=-1)
    mat1 = torch.zeros(*tensor.shape[:-1], 4, 4, dtype=tensor.dtype, device=tensor.device)
    mat1[..., 0, 0] = tensor[..., 0]
    mat1[..., 0, 1] = -tensor[..., 1]
    mat1[..., 0, 2] = -tensor[..., 2]
    mat1[..., 0, 3] = -tensor[..., 3]
    mat1[..., 1, 0] = tensor[..., 1]
    mat1[..., 1, 1] = tensor[..., 0]
    mat1[..., 1, 2] = -tensor[..., 3]
    mat1[..., 1, 3] = tensor[..., 2]
    mat1[..., 2, 0] = tensor[..., 2]
    mat1[..., 2, 1] = tensor[..., 3]
    mat1[..., 2, 2] = tensor[..., 0]
    mat1[..., 2, 3] = -tensor[..., 1]
    mat1[..., 3, 0] = tensor[..., 3]
    mat1[..., 3, 1] = -tensor[..., 2]
    mat1[..., 3, 2] = tensor[..., 1]
    mat1[..., 3, 3] = tensor[..., 0]

    mat2 = torch.zeros(*tensor.shape[:-1], 4, 4, dtype=tensor.dtype, device=tensor.device)
    mat2[..., 0, 0] = tensor[..., 0]
    mat2[..., 0, 1] = -tensor[..., 1]
    mat2[..., 0, 2] = -tensor[..., 2]
    mat2[..., 0, 3] = -tensor[..., 3]
    mat2[..., 1, 0] = tensor[..., 1]
    mat2[..., 1, 1] = tensor[..., 0]
    mat2[..., 1, 2] = tensor[..., 3]
    mat2[..., 1, 3] = -tensor[..., 2]
    mat2[..., 2, 0] = tensor[..., 2]
    mat2[..., 2, 1] = -tensor[..., 3]
    mat2[..., 2, 2] = tensor[..., 0]
    mat2[..., 2, 3] = tensor[..., 1]
    mat2[..., 3, 0] = tensor[..., 3]
    mat2[..., 3, 1] = tensor[..., 2]
    mat2[..., 3, 2] = -tensor[..., 1]
    mat2[..., 3, 3] = tensor[..., 0]

    mat2 = torch.conj(mat2).transpose(-1, -2)
    mat = torch.matmul(mat1, mat2)
    return mat[..., 1:, 1:]


def quat_to_rotmat(quats):
    """Quaternion (w, x, y, z) to 3x3 rotation matrix."""
    q = quats / torch.sqrt((quats ** 2).sum(dim=-1, keepdim=True))
    r, x, y, z = [i.squeeze(-1) for i in q.split(1, dim=-1)]

    R = torch.zeros((*r.shape, 3, 3)).to(r)
    R[..., 0, 0] = 1 - 2 * (y * y + z * z)
    R[..., 0, 1] = 2 * (x * y - r * z)
    R[..., 0, 2] = 2 * (x * z + r * y)
    R[..., 1, 0] = 2 * (x * y + r * z)
    R[..., 1, 1] = 1 - 2 * (x * x + z * z)
    R[..., 1, 2] = 2 * (y * z - r * x)
    R[..., 2, 0] = 2 * (x * z - r * y)
    R[..., 2, 1] = 2 * (y * z + r * x)
    R[..., 2, 2] = 1 - 2 * (x * x + y * y)
    return R


def get_covariance(s, r):
    """Compute covariance matrix from scales and rotation matrices."""
    L = torch.zeros((*s.shape[:2], 3, 3)).to(s)
    for i in range(s.size(-1)):
        L[..., i, i] = s[..., i]
    L = r @ L
    covariance = L @ L.mT
    return covariance


# --- Feature shape utilities ---

def nlc_to_nchw(x, shape):
    """[N, L, C] -> [N, C, H, W]"""
    B, L, C = x.shape
    return x.transpose(1, 2).reshape(B, C, *shape).contiguous()


def nchw_to_nlc(x):
    """[N, C, H, W] -> [N, L, C]"""
    return x.flatten(2).transpose(1, 2).contiguous()


def flatten_multi_scale_feats(feats):
    feat_flatten = torch.cat([nchw_to_nlc(feat) for feat in feats], dim=1)
    shapes = torch.stack([
        torch.tensor(feat.shape[2:], device=feat_flatten.device)
        for feat in feats
    ])
    return feat_flatten, shapes


def get_level_start_index(shapes):
    return torch.cat((shapes.new_zeros((1,)), shapes.prod(1).cumsum(0)[:-1]))


def generate_grid(grid_shape, value=None, offset=0, normalize=False):
    if value is None:
        value = grid_shape
    grid = []
    for i, (s, val) in enumerate(zip(grid_shape, value)):
        g = torch.linspace(offset, val - 1 + offset, s, dtype=torch.float)
        if normalize:
            g /= val
        shape_ = [1 for _ in grid_shape]
        shape_[i] = s
        g = g.reshape(*shape_).expand(*grid_shape)
        grid.append(g)
    return torch.stack(grid, dim=-1)


# --- Coordinate transforms ---

def cam2world(points, cam2img, cam2ego, img_aug_mat=None):
    if img_aug_mat is not None:
        post_rots = img_aug_mat[..., :3, :3]
        post_trans = img_aug_mat[..., :3, 3]
        points = points - post_trans.unsqueeze(-2)
        points = (torch.inverse(post_rots).unsqueeze(2)
                  @ points.unsqueeze(-1)).squeeze(-1)

    cam2img = cam2img[..., :3, :3]
    with autocast(enabled=False):
        combine = cam2ego[..., :3, :3] @ torch.inverse(cam2img)
        points = points.float()
        points = torch.cat(
            [points[..., :2] * points[..., 2:3], points[..., 2:3]], dim=-1)
        points = combine.unsqueeze(2) @ points.unsqueeze(-1)
    points = points.squeeze(-1) + cam2ego[..., None, :3, 3]
    return points


def world2cam(points, cam2img, cam2ego, img_aug_mat=None, eps=1e-6):
    points = points - cam2ego[..., None, :3, 3]
    points = torch.inverse(cam2ego[..., None, :3, :3]) @ points.unsqueeze(-1)
    points = (cam2img[..., None, :3, :3] @ points).squeeze(-1)
    points = points / points[..., 2:3].clamp(eps)
    if img_aug_mat is not None:
        points = img_aug_mat[..., None, :3, :3] @ points.unsqueeze(-1)
        points = points.squeeze(-1) + img_aug_mat[..., None, :3, 3]
    return points[..., :2]


# --- Color augmentation ---

def _rgb_to_hsv(image, eps=1e-8):
    """RGB [0,255] -> HSV (H in [0,360], S in [0,1], V in [0,255])."""
    image = image / 255.0
    max_rgb, argmax_rgb = image.max(-3)
    min_rgb, _ = image.min(-3)
    deltac = max_rgb - min_rgb
    v = max_rgb
    s = deltac / (max_rgb + eps)
    deltac = torch.where(deltac == 0, torch.ones_like(deltac), deltac)
    rc, gc, bc = torch.unbind((max_rgb.unsqueeze(-3) - image), dim=-3)
    h1 = bc - gc
    h2 = (rc - bc) + 2.0 * deltac
    h3 = (gc - rc) + 4.0 * deltac
    h = torch.stack((h1, h2, h3), dim=-3) / deltac.unsqueeze(-3)
    h = torch.gather(h, dim=-3, index=argmax_rgb.unsqueeze(-3)).squeeze(-3)
    h = (h / 6.0) % 1.0
    h = h * 360.0
    v = v * 255.0
    return torch.stack((h, s, v), dim=-3)


def _hsv_to_rgb(image):
    """HSV (H in [0,360], S in [0,1], V in [0,255]) -> RGB [0,255]."""
    h = image[..., 0, :, :] / 360.0
    s = image[..., 1, :, :]
    v = image[..., 2, :, :] / 255.0
    hi = torch.floor(h * 6) % 6
    f = ((h * 6) % 6) - hi
    one = torch.tensor(1.0, device=image.device, dtype=image.dtype)
    p = v * (one - s)
    q = v * (one - f * s)
    t = v * (one - (one - f) * s)
    hi = hi.long()
    indices = torch.stack([hi, hi + 6, hi + 12], dim=-3)
    out = torch.stack((v, q, p, p, t, v, t, v, v, q, p, p, p, p, t, v, v, q), dim=-3)
    out = torch.gather(out, -3, indices)
    out = out * 255.0
    return out


class GpuPhotoMetricDistortion:
    """GPU-based photometric distortion augmentation (brightness, contrast, saturation, hue)."""

    def __init__(self, brightness_delta=32, contrast_range=(0.5, 1.5),
                 saturation_range=(0.5, 1.5), hue_delta=18):
        self.brightness_delta = brightness_delta
        self.contrast_lower, self.contrast_upper = contrast_range
        self.saturation_lower, self.saturation_upper = saturation_range
        self.hue_delta = hue_delta

    def __call__(self, imgs):
        imgs = imgs[:, [2, 1, 0], :, :]  # BGR to RGB

        contrast_modes = [random.randint(2) for _ in range(imgs.shape[0])]

        for idx in range(imgs.shape[0]):
            if random.randint(2):
                delta = random.uniform(-self.brightness_delta, self.brightness_delta)
                imgs[idx] += delta
            if contrast_modes[idx] == 0:
                if random.randint(2):
                    alpha = random.uniform(self.contrast_lower, self.contrast_upper)
                    imgs[idx] *= alpha

        imgs = _rgb_to_hsv(imgs)

        for idx in range(imgs.shape[0]):
            if random.randint(2):
                imgs[idx, 1] *= random.uniform(self.saturation_lower, self.saturation_upper)
            if random.randint(2):
                imgs[idx, 0] += random.uniform(-self.hue_delta, self.hue_delta)

        imgs[:, 0][imgs[:, 0] > 360] -= 360
        imgs[:, 0][imgs[:, 0] < 0] += 360

        imgs = _hsv_to_rgb(imgs)

        for idx in range(imgs.shape[0]):
            if contrast_modes[idx] == 1:
                if random.randint(2):
                    alpha = random.uniform(self.contrast_lower, self.contrast_upper)
                    imgs[idx] *= alpha
            if random.randint(2):
                imgs[idx] = imgs[idx, random.permutation(3)]

        imgs = imgs[:, [2, 1, 0], :, :]  # RGB to BGR
        return imgs


# --- Batch indexing ---

def batch_indexing(batched_data, batched_indices, layout='channel_first'):
    if layout == 'channel_first':
        assert batched_data.shape[0] == batched_indices.shape[0]
        batch_size, n_channels = batched_data.shape[:2]
        indices_shape = list(batched_indices.shape[1:])
        total = 1
        for s in indices_shape:
            total *= s
        batched_indices = batched_indices.reshape([batch_size, 1, -1])
        batched_indices = batched_indices.expand([batch_size, n_channels, total])
        result = torch.gather(batched_data, dim=2, index=batched_indices.to(torch.int64))
        result = result.view([batch_size, n_channels] + indices_shape)
        return result
    elif layout == 'channel_last':
        assert batched_data.shape[0] == batched_indices.shape[0]
        batch_size = batched_data.shape[0]
        view_shape = [batch_size] + [1] * (len(batched_indices.shape) - 1)
        expand_shape = [batch_size] + list(batched_indices.shape)[1:]
        indices_of_batch = torch.arange(batch_size, dtype=torch.long, device=batched_data.device)
        indices_of_batch = indices_of_batch.view(view_shape).expand(expand_shape)
        if len(batched_data.shape) == 2:
            return batched_data[indices_of_batch, batched_indices.to(torch.long)]
        else:
            return batched_data[indices_of_batch, batched_indices.to(torch.long), :]
    else:
        raise ValueError(f"Unknown layout: {layout}")


def unbatched_forward(func):
    """Decorator: apply func per-sample by slicing batch dim."""
    def wrapper(*args, **kwargs):
        bs = None
        for arg in list(args) + list(kwargs.values()):
            if isinstance(arg, torch.Tensor):
                if bs is None:
                    bs = arg.size(0)
                else:
                    assert bs == arg.size(0)

        outputs = []
        for i in range(bs):
            output = func(
                *[arg[i] if isinstance(arg, torch.Tensor) else arg for arg in args],
                **{k: v[i] if isinstance(v, torch.Tensor) else v for k, v in kwargs.items()})
            outputs.append(output)

        if isinstance(outputs[0], tuple):
            return tuple([
                torch.stack([out[i] for out in outputs])
                for i in range(len(outputs[0]))
            ])
        else:
            return torch.stack(outputs)

    return wrapper


# --- OCC3D category mapping (21 fine-grained -> 16 merged) ---

OCC3D_CATEGORIES = (
    ['barrier'],
    ['bicycle'],
    ['bus'],
    ['car'],
    ['construction vehicle'],
    ['motorcycle'],
    ['person'],
    ['cone'],
    ['trailer'],
    ['truck'],
    ['road'],
    ['sidewalk'],
    ['terrain', 'grass'],
    ['building', 'wall', 'fence', 'pole', 'sign'],
    ['vegetation'],
    ['sky'],
)


class DumpConfig:
    def __init__(self):
        self.enabled = False
        self.out_dir = 'outputs'
        self.stage_count = 0
        self.frame_count = 0


DUMP = DumpConfig()
