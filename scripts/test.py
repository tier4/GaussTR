#!/usr/bin/env python
"""Testing/evaluation script for GaussTR Lightning.

Temporal fusion testing with multi-GPU support.

GT Format (Occ3D-nuScenes, 18 classes):
- GT 0: others (ignored in mIoU)
- GT 1-16: semantic classes (barrier, bicycle, bus, car, construction_vehicle,
           motorcycle, pedestrian, traffic_cone, trailer, truck, driveable_surface,
           other_flat, sidewalk, terrain, manmade, vegetation)
- GT 17: free/empty

Model outputs GT classes 1-11, 13-17 (never predicts GT 0 or GT 12=other_flat).
"""

import copy
import os
import sys
import pickle
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Optional, Any, Tuple
from concurrent.futures import ThreadPoolExecutor
import torch.multiprocessing as mp

import hydra
import pytorch_lightning as pl
import torch
import numpy as np
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.callbacks import Callback
from tqdm import tqdm
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.data._utils.collate import default_collate

import torch.multiprocessing
try:
    torch.multiprocessing.set_sharing_strategy('file_system')
except Exception:
    pass

torch.set_float32_matmul_precision('high')

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from models import GaussTRLightning
from dataset import GaussTRDataModule

# --- CONSTANTS ---
# Standard Occ3D-nuScenes Class Names (17 classes)
OCC_CLASSES = [
    'others', 'barrier', 'bicycle', 'bus', 'car', 'construction_vehicle',
    'motorcycle', 'pedestrian', 'traffic_cone', 'trailer', 'truck',
    'driveable_surface', 'other_flat', 'sidewalk', 'terrain', 'manmade', 'vegetation'
]


class SavePredictionsCallback(Callback):
    """Callback to save predictions during testing."""

    def __init__(self, output_dir: str):
        super().__init__()
        self.output_dir = output_dir
        self.pred_dir = os.path.join(output_dir, 'predictions')
        os.makedirs(self.pred_dir, exist_ok=True)

    def on_test_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs,
        batch,
        batch_idx: int,
        dataloader_idx: int = 0,
    ):
        """Save predictions after each test batch."""
        if outputs is None:
            return

        preds = outputs.get('preds')
        if preds is None:
            return

        if isinstance(preds, torch.Tensor):
            preds = preds.cpu().numpy()

        gt_occ = batch.get('gt_occ')
        mask = batch.get('mask_camera')
        timestamps = batch.get('timestamp', [])
        batch_size = preds.shape[0]

        for i in range(batch_size):
            if timestamps is not None and len(timestamps) > i:
                ts = timestamps[i]
                if isinstance(ts, torch.Tensor):
                    ts = ts.item()
                filename = f"{ts:.6f}"
            else:
                sample_idx = batch.get('sample_idx')
                if sample_idx is not None:
                    idx = sample_idx[i].item() if isinstance(sample_idx[i], torch.Tensor) else sample_idx[i]
                    filename = f"{idx:06d}"
                else:
                    world_size = trainer.world_size if trainer.world_size else 1
                    rank = trainer.global_rank if trainer.global_rank else 0
                    filename = f"{batch_idx * batch_size * world_size + rank * batch_size + i:06d}"

            save_dict = {'pred': preds[i]}
            if gt_occ is not None:
                gt = gt_occ[i]
                if isinstance(gt, torch.Tensor):
                    gt = gt.cpu().numpy()
                save_dict['gt'] = gt
            if mask is not None:
                m = mask[i]
                if isinstance(m, torch.Tensor):
                    m = m.cpu().numpy()
                save_dict['mask'] = m

            np.savez_compressed(os.path.join(self.pred_dir, f'{filename}.npz'), **save_dict)

    def on_test_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        if trainer.global_rank == 0:
            num_files = len([f for f in os.listdir(self.pred_dir) if f.endswith('.npz')])
            print(f"\nSaved {num_files} predictions to: {self.pred_dir}")


class FrameDataset(Dataset):
    """Flattened dataset for parallel IO."""
    def __init__(self, scene_infos, transforms):
        self.flat_infos = []
        for scene_id, frames in scene_infos.items():
            for frame_info in frames:
                # Use deepcopy to prevent transforms from mutating nested dicts
                f_info = copy.deepcopy(frame_info)
                f_info['scene_id'] = scene_id
                self.flat_infos.append(f_info)
        self.transforms = transforms

    def __len__(self):
        return len(self.flat_infos)

    def __getitem__(self, idx):
        info = self.flat_infos[idx]
        data = self.transforms(info)
        data['scene_id'] = info['scene_id']
        data['token'] = info['token']
        data['ego2global'] = torch.from_numpy(np.array(info['ego2global'], dtype=np.float32))
        return data


def custom_collate_fn(batch):
    meta_keys = ['scene_id', 'token', 'img_metas']
    batch_data = []
    batch_meta = defaultdict(list)
    for item in batch:
        data_item = {}
        for k, v in item.items():
            if k in meta_keys or isinstance(v, str):
                batch_meta[k].append(v)
            else:
                data_item[k] = v
        batch_data.append(data_item)
    collated = default_collate(batch_data)
    collated.update(batch_meta)
    return collated


class OccupancyIoU(torch.nn.Module):
    """Simple Metric Implementation to avoid import errors."""
    def __init__(self, num_classes=18):
        super().__init__()
        self.num_classes = num_classes
        # Use float64 to avoid precision loss when accumulating large voxel counts
        self.register_buffer('hist', torch.zeros(num_classes, num_classes, dtype=torch.float64))

    def update(self, pred, gt, mask=None):
        if mask is not None:
            pred = pred[mask]
            gt = gt[mask]
        
        # GT should be 0-17. Pred should be 0-17.
        # Filter out invalid values just in case
        valid = (gt < self.num_classes) & (gt >= 0)
        pred = pred[valid]
        gt = gt[valid]
        
        k = (gt * self.num_classes) + pred
        self.hist += torch.bincount(k, minlength=self.num_classes**2).view(self.num_classes, self.num_classes)

    def compute(self):
        return self.hist

    def reset(self):
        self.hist.zero_()


class TemporalTester:
    """Temporal fusion tester - Fixed Alignment."""

    def __init__(
        self,
        model: GaussTRLightning,
        device: torch.device,
        num_frames_before: int = 2,
        num_frames_after: int = 2,
        density_threshold: float = 0.04,
        inference_batch_size: int = 8
    ):
        self.model = model.to(device).eval()
        self.device = device
        self.num_frames_before = num_frames_before
        self.num_frames_after = num_frames_after
        self.density_threshold = density_threshold
        self.inference_batch_size = inference_batch_size
        self.io_pool = ThreadPoolExecutor(max_workers=4)

        self.head = model.gauss_heads[-1]
        self.voxelizer = self.head.voxelizer

        from models.utils import OCC3D_CATEGORIES, flatten_bsn_forward, cam2world, rotmat_to_quat, quat_to_rotmat
        from models.gausstr_head import merge_probs, prompt_denoising, inverse_sigmoid

        self.OCC3D_CATEGORIES = OCC3D_CATEGORIES
        self.flatten_bsn_forward = flatten_bsn_forward
        self.cam2world = cam2world
        self.rotmat_to_quat = rotmat_to_quat
        self.quat_to_rotmat = quat_to_rotmat
        self.merge_probs = merge_probs
        self.prompt_denoising = prompt_denoising
        self.inverse_sigmoid = inverse_sigmoid
        self.F = F

        self.metric = OccupancyIoU(num_classes=18).to(device)

        # Dynamic GT class indices (moving objects):
        # GT 2=bicycle, GT 3=bus, GT 4=car, GT 5=construction_vehicle,
        # GT 6=motorcycle, GT 7=pedestrian, GT 9=trailer, GT 10=truck
        self.DYNAMIC_IDS = [2, 3, 4, 5, 6, 7, 9, 10]
        self.dynamic_lut = torch.zeros(18, device=device, dtype=torch.bool)
        self.dynamic_lut[self.DYNAMIC_IDS] = True

    @torch.inference_mode()
    def extract_gaussians_batched(self, batch_list: List[Dict[str, torch.Tensor]]) -> List[Dict[str, torch.Tensor]]:
        """Batched Inference for Backbone."""
        if not batch_list: return []
        
        bs = len(batch_list)
        feats = torch.cat([b['feats'] for b in batch_list], dim=0).to(self.device, non_blocking=True)
        depth = torch.cat([b['depth'] for b in batch_list], dim=0).to(self.device, non_blocking=True)
        cam2img = torch.cat([b['cam2img'] for b in batch_list], dim=0).to(self.device, non_blocking=True)
        cam2ego = torch.cat([b['cam2ego'] for b in batch_list], dim=0).to(self.device, non_blocking=True)
        
        img_aug_mat = None
        if 'img_aug_mat' in batch_list[0] and batch_list[0]['img_aug_mat'] is not None:
            img_aug_mat = torch.cat([b['img_aug_mat'] for b in batch_list], dim=0).to(self.device, non_blocking=True)

        decoder_outputs = self.model._forward_features(feats, bs)
        
        n = feats.shape[1]
        x = decoder_outputs['hidden_states'][-1]
        ref_pts = decoder_outputs['references'][-1]
        x = x.reshape((bs, n) + tuple(x.shape[1:]))

        deltas = self.head.regress_head(x)
        ref_pts_reshaped = ref_pts.reshape(tuple(x.shape[:-1]) + (-1,))
        ref_pts = (deltas[..., :2] + self.inverse_sigmoid(ref_pts_reshaped)).sigmoid()

        depth = depth.clamp(max=self.head.depth_limit)
        if depth.dim() == 5: depth = depth.squeeze(2)

        sample_depth = self.flatten_bsn_forward(
            self.F.grid_sample, depth[:, :n, None],
            ref_pts.unsqueeze(2) * 2 - 1,
            mode='bilinear', align_corners=False)
        sample_depth = sample_depth[:, :, 0, 0, :, None]

        points = torch.cat([
            ref_pts * self.head.image_shape_tensor.to(ref_pts.device),
            sample_depth * (1 + deltas[..., 2:3])
        ], -1)
        
        means3d = self.cam2world(points, cam2img, cam2ego, img_aug_mat)
        
        opacities = self.head.opacity_head(x).float()
        features = self.head.feature_head(x).float()
        scales = self.head.scale_head(x) * self.head.scale_transform(
            sample_depth, cam2img[..., 0, 0]).clamp(1e-6)
        rotations = self.flatten_bsn_forward(self.rotmat_to_quat, cam2ego[..., :3, :3])
        rotations = rotations.unsqueeze(2).expand(-1, -1, x.size(2), -1)

        if self.head.text_proto_embeds is not None:
            semantic_logits = features @ self.head.text_proto_embeds
        else:
            semantic_logits = features
        semantic_probs = self.merge_probs(semantic_logits.softmax(-1), self.OCC3D_CATEGORIES)

        results = []
        for i in range(bs):
            results.append({
                'means3d': means3d[i].flatten(0, 1),
                'opacities': opacities[i].flatten(0, 1),
                'features': features[i].flatten(0, 1),
                'scales': scales[i].flatten(0, 1),
                'rotations': rotations[i].flatten(0, 1),
                'semantic_logits': semantic_logits[i].flatten(0, 1),
                'semantic_probs': semantic_probs[i].flatten(0, 1),
            })
        
        return results

    def rigid_inverse(self, matrix: torch.Tensor) -> torch.Tensor:
        R = matrix[:3, :3]
        t = matrix[:3, 3]
        inv = torch.eye(4, device=matrix.device, dtype=matrix.dtype)
        inv[:3, :3] = R.t()
        inv[:3, 3] = -R.t() @ t
        return inv

    def transform_gaussians(
        self,
        gaussians: Dict[str, torch.Tensor],
        src_ego2global: torch.Tensor,
        dst_ego2global: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        result = dict(gaussians)
        dst_inv = self.rigid_inverse(dst_ego2global)
        Tmat = dst_inv @ src_ego2global
        R = Tmat[:3, :3]
        t = Tmat[:3, 3]

        means = gaussians['means3d']
        means_transformed = means @ R.t() + t
        result['means3d'] = means_transformed

        rot_matrices = self.quat_to_rotmat(gaussians['rotations'])
        rot_transformed = R @ rot_matrices
        quats_transformed = self.rotmat_to_quat(rot_transformed)
        result['rotations'] = quats_transformed
        
        return result

    def collect_transformed_gaussians(
        self,
        all_gaussians: List[Dict[str, torch.Tensor]],
        all_ego2global: List[torch.Tensor],
        current_idx: int,
    ) -> Tuple[List[Dict[str, torch.Tensor]], int]:
        current_ego = all_ego2global[current_idx]
        start_idx = max(0, current_idx - self.num_frames_before)
        end_idx = min(len(all_gaussians), current_idx + self.num_frames_after + 1)

        transformed_list = []
        current_idx_in_list = 0

        for t in range(start_idx, end_idx):
            g = self.transform_gaussians(all_gaussians[t], all_ego2global[t], current_ego)
            transformed_list.append(g)
            if t == current_idx:
                current_idx_in_list = len(transformed_list) - 1

        return transformed_list, current_idx_in_list

    def _voxelize_gaussians(self, gaussians, density_threshold):
        grid_shape = tuple(self.voxelizer.grid_shape)
        # GT Format: 0=others, 1-16=semantic classes, 17=free
        FREE_CLASS = 17

        if gaussians is None or len(gaussians['means3d']) == 0:
            return (torch.full(grid_shape, FREE_CLASS, dtype=torch.long, device=self.device),
                    torch.zeros(grid_shape, dtype=torch.bool, device=self.device),
                    torch.zeros(grid_shape, dtype=torch.float16, device=self.device))

        opacity_valid = gaussians['opacities'].squeeze(-1) > 0.05
        if opacity_valid.sum() == 0:
            return (torch.full(grid_shape, FREE_CLASS, dtype=torch.long, device=self.device),
                    torch.zeros(grid_shape, dtype=torch.bool, device=self.device),
                    torch.zeros(grid_shape, dtype=torch.float16, device=self.device))

        g = {k: v[opacity_valid] for k, v in gaussians.items()}
        semantic_probs = g['semantic_logits'].softmax(-1)

        density, grid_feats = self.voxelizer(
            means3d=g['means3d'].unsqueeze(0),
            opacities=g['opacities'].unsqueeze(0),
            features=semantic_probs.unsqueeze(0),
            scales=g['scales'].unsqueeze(0),
            rotations=g['rotations'].unsqueeze(0),
        )

        probs = self.prompt_denoising(grid_feats)
        probs = self.merge_probs(probs, self.OCC3D_CATEGORIES)

        raw_preds = probs.argmax(-1)  # 0-15 (16 classes after merge_probs)
        confidence = probs.max(-1).values

        # Model's formula (same as gausstr_head.py):
        # GT format: 0=others, 1=barrier, ..., 11=driveable_surface, 12=other_flat,
        #            13=sidewalk, 14=terrain, 15=manmade, 16=vegetation, 17=free
        # Model 0-10 (barrier..road) -> GT 1-11: +1
        # Model 11-15 (sidewalk..sky) -> GT 13-17: +2 (skip GT 12 other_flat)
        preds = raw_preds + (raw_preds > 10).long() + 1

        density_check = density.squeeze(-1).squeeze(0)
        valid = density_check > density_threshold
        preds = preds.squeeze(0)
        confidence = confidence.squeeze(0).to(torch.float16)

        # Empty voxels → GT 17 (free)
        preds[~valid] = 17

        return preds, valid, confidence

    def generate_occupancy_hybrid(
        self,
        all_gaussians_transformed: List[Dict[str, torch.Tensor]],
        current_idx: int,
        static_min_agreement: float = 2.0,
        dynamic_min_agreement: float = 1.0,
    ) -> torch.Tensor:
        grid_shape = tuple(self.voxelizer.grid_shape)
        # GT Format: 0=others, 1-16=semantic classes, 17=free
        FREE_CLASS = 17
        num_classes = 18

        T = len(all_gaussians_transformed)
        streams = [torch.cuda.Stream() for _ in range(T)]
        results = [None] * T

        for i, g in enumerate(all_gaussians_transformed):
            with torch.cuda.stream(streams[i]):
                results[i] = self._voxelize_gaussians(g, self.density_threshold)
        torch.cuda.synchronize()

        all_preds = torch.stack([r[0] for r in results], dim=0)
        all_valid = torch.stack([r[1] for r in results], dim=0)
        all_conf = torch.stack([r[2] for r in results], dim=0)

        flattened_preds = all_preds.view(T, -1)
        flattened_valid = all_valid.view(T, -1)
        flattened_conf = all_conf.view(T, -1)
        N = flattened_preds.shape[1]

        weighted_counts = torch.zeros((N, num_classes), device=self.device, dtype=torch.float32)
        for t in range(T):
            idx = flattened_preds[t].unsqueeze(1)
            # Only weight valid votes (non-free)
            # Free votes (class 17) have valid=False, so weight=0
            weight = (flattened_valid[t].float() * flattened_conf[t].float()).unsqueeze(1)
            weighted_counts.scatter_add_(dim=1, index=idx, src=weight)

        total_vote_counts = weighted_counts.view(*grid_shape, num_classes)

        # Look at classes 1-16 (Index 0=others ignored, Index 17=free excluded)
        # We want the best semantic class vote (excluding others and free)
        occupied_votes_map = total_vote_counts[..., 1:17]
        best_occupied_votes, best_occupied_sub_idx = occupied_votes_map.max(dim=-1)
        best_occupied_class = best_occupied_sub_idx + 1  # Convert 0-15 back to GT 1-16

        # Current Frame
        current_preds = all_preds[current_idx]
        current_valid = all_valid[current_idx]
        current_is_dynamic = self.dynamic_lut[current_preds]
        history_is_dynamic = self.dynamic_lut[best_occupied_class]

        # Logic: Base + Inpaint
        keep_current = current_valid & (current_preds != FREE_CLASS)

        fill_from_history = (~keep_current) & \
                            (best_occupied_votes >= static_min_agreement) & \
                            (~history_is_dynamic)

        final_preds = torch.full(grid_shape, FREE_CLASS, dtype=torch.long, device=self.device)
        final_preds = torch.where(fill_from_history, best_occupied_class, final_preds)
        final_preds = torch.where(keep_current, current_preds, final_preds)

        return final_preds

    def test_scenes(
        self,
        data_cfg: Dict[str, Any],
        scene_ids: List[str],
        pred_dir: Optional[str] = None,
        static_min_agreement: float = 2.0,
        dynamic_min_agreement: float = 1.0,
        show_progress: bool = True,
    ) -> Tuple[torch.Tensor, int]:
        from dataset.transforms import LoadMultiViewImages, ImageAug3D, LoadFeatMaps, PackInputs, Compose

        gt_root = data_cfg.get('gt_root', os.path.join(data_cfg['data_root'], 'gts'))

        transforms = Compose([
            LoadMultiViewImages(to_float32=True, num_views=6, data_root=data_cfg['data_root']),
            ImageAug3D(final_dim=tuple(data_cfg.get('image_size', (432, 768))),
                      resize_lim=(0.48, 0.48), is_train=False),
            LoadFeatMaps(data_root=data_cfg.get('depth_root', '/mnt/nvme0/data/nuscenes_unidepth'),
                        key='depth', apply_aug=True),
            LoadFeatMaps(data_root=data_cfg.get('feats_root', '/mnt/nvme0/data/nuscenes_dinov3clip'),
                        key='feats'),
            PackInputs(),
        ])

        ann_file = os.path.join(data_cfg['data_root'], 'nuscenes_infos_val.pkl')
        with open(ann_file, 'rb') as f:
            ann_data = pickle.load(f)
        infos = ann_data.get('data_list', ann_data.get('infos', ann_data))

        target_scene_infos = defaultdict(list)
        for info in infos:
            if info['scene_idx'] in scene_ids:
                for cam_name, cam_info in info['images'].items():
                    img_path = cam_info['img_path']
                    if not os.path.isabs(img_path) and 'samples/' not in img_path:
                        cam_info['img_path'] = os.path.join(data_cfg['data_root'], 'samples', cam_name, img_path)
                target_scene_infos[info['scene_idx']].append(info)
        
        for sid in target_scene_infos:
            target_scene_infos[sid].sort(key=lambda x: x['timestamp'])

        dataset = FrameDataset(target_scene_infos, transforms)
        dataloader = DataLoader(
            dataset, 
            batch_size=self.inference_batch_size, 
            shuffle=False, 
            num_workers=8, 
            pin_memory=True, 
            collate_fn=custom_collate_fn,
            prefetch_factor=2, 
            persistent_workers=True
        )

        self.metric.reset()
        total_frames_processed = 0
        current_scene_buffer = {'id': None, 'data': []}

        iterator = tqdm(dataloader, desc=f"GPU {self.device}", disable=not show_progress)

        def process_scene_buffer(scene_id, buffer_data):
            if not buffer_data: return
            
            all_ego2global = [f['ego2global'] for f in buffer_data]
            all_gaussians = [f['gaussians'] for f in buffer_data]

            for idx in range(len(buffer_data)):
                transformed_gaussians, current_idx_in_list = self.collect_transformed_gaussians(
                    all_gaussians, all_ego2global, idx
                )

                pred = self.generate_occupancy_hybrid(
                    transformed_gaussians, current_idx_in_list,
                    static_min_agreement=static_min_agreement,
                    dynamic_min_agreement=dynamic_min_agreement
                )

                token = buffer_data[idx]['token']
                
                if pred_dir:
                    save_path = os.path.join(pred_dir, scene_id, f'{token}.npz')
                    os.makedirs(os.path.dirname(save_path), exist_ok=True)
                    pred_np = pred.cpu().numpy().astype(np.uint8)
                    self.io_pool.submit(np.savez_compressed, save_path, semantics=pred_np)

                gt_path = os.path.join(gt_root, scene_id, token, 'labels.npz')
                if os.path.exists(gt_path):
                    gt_data = np.load(gt_path)
                    gt = torch.from_numpy(gt_data['semantics']).long().to(self.device, non_blocking=True)
                    mask = torch.from_numpy(gt_data['mask_camera']).bool().to(self.device, non_blocking=True)
                    self.metric.update(pred.unsqueeze(0), gt.unsqueeze(0), mask.unsqueeze(0))

        with torch.no_grad():
            for batch in iterator:
                bs = len(batch['token'])
                input_list = []
                for i in range(bs):
                    input_item = {
                        'feats': batch['feats'][i:i+1].to(self.device, non_blocking=True),
                        'depth': batch['depth'][i:i+1].to(self.device, non_blocking=True),
                        'cam2img': batch['cam2img'][i:i+1].to(self.device, non_blocking=True),
                        'cam2ego': batch['cam2ego'][i:i+1].to(self.device, non_blocking=True),
                    }
                    if 'img_aug_mat' in batch:
                        input_item['img_aug_mat'] = batch['img_aug_mat'][i:i+1].to(self.device, non_blocking=True)
                    input_list.append(input_item)

                batch_gaussians = self.extract_gaussians_batched(input_list)

                for i in range(bs):
                    scene_id = batch['scene_id'][i]
                    token = batch['token'][i]
                    ego2global = batch['ego2global'][i].to(self.device)

                    if current_scene_buffer['id'] is not None and current_scene_buffer['id'] != scene_id:
                        process_scene_buffer(current_scene_buffer['id'], current_scene_buffer['data'])
                        current_scene_buffer['data'] = [] 
                    
                    current_scene_buffer['id'] = scene_id
                    current_scene_buffer['data'].append({
                        'token': token,
                        'ego2global': ego2global,
                        'gaussians': batch_gaussians[i]
                    })
                    total_frames_processed += 1

            if current_scene_buffer['data']:
                process_scene_buffer(current_scene_buffer['id'], current_scene_buffer['data'])

        return self.metric.hist.clone(), total_frames_processed


def _multigpu_worker(
    rank: int,
    world_size: int,
    checkpoint_path: str,
    model_cfg: dict,
    temporal_cfg: dict,
    data_cfg: dict,
    scene_ids: List[str],
    result_queue: mp.Queue,
    pred_dir: Optional[str],
    static_min: float,
    dynamic_min: float,
):
    try:
        device = torch.device(f'cuda:{rank}')
        torch.cuda.set_device(device)

        model = GaussTRLightning.load_from_checkpoint(checkpoint_path, **model_cfg) if checkpoint_path.endswith('.ckpt') else None
        if model is None:
            # Fallback loading for .pth files
            model = GaussTRLightning(**model_cfg)
            ckpt = torch.load(checkpoint_path, map_location='cpu')
            state = ckpt.get('state_dict', ckpt)
            # Use removeprefix to only strip the 'model.' prefix, not all occurrences
            state = {k.removeprefix('model.'): v for k, v in state.items()}
            model.load_state_dict(state, strict=False)

        tester = TemporalTester(
            model, device,
            num_frames_before=temporal_cfg.get('num_frames_before', 2),
            num_frames_after=temporal_cfg.get('num_frames_after', 2),
            density_threshold=temporal_cfg.get('density_threshold', 0.04),
            inference_batch_size=8,
        )

        hist, total_frames = tester.test_scenes(
            data_cfg, scene_ids, pred_dir, static_min, dynamic_min, show_progress=True
        )

        tester.io_pool.shutdown(wait=True)
        result_queue.put({'rank': rank, 'hist': hist.cpu().numpy(), 'total_frames': total_frames, 'error': None})
    except Exception as e:
        import traceback
        traceback.print_exc()
        result_queue.put({'rank': rank, 'hist': None, 'total_frames': 0, 'error': str(e)})


def test_multigpu(
    checkpoint_path: str,
    model_cfg: dict,
    temporal_cfg: dict,
    data_cfg: dict,
    num_gpus: int,
    output_dir: str,
    save_predictions: bool,
    pred_dir: Optional[str],
    static_min: float,
    dynamic_min: float,
) -> Dict[str, float]:

    ann_file = os.path.join(data_cfg['data_root'], 'nuscenes_infos_val.pkl')
    with open(ann_file, 'rb') as f:
        ann_data = pickle.load(f)
    infos = ann_data.get('data_list', ann_data.get('infos', ann_data))
    all_scenes = sorted(set(info['scene_idx'] for info in infos))

    print(f"\n{'='*80}")
    print(f"MULTI-GPU TEMPORAL TEST ({num_gpus} GPUs) - Version 9.0 Fixed")
    print(f"{'='*80}")
    print(f"Total scenes: {len(all_scenes)}, Total frames: {len(infos)}")

    assignments = [[] for _ in range(num_gpus)]
    for i, scene in enumerate(all_scenes):
        assignments[i % num_gpus].append(scene)

    if save_predictions and pred_dir:
        os.makedirs(pred_dir, exist_ok=True)
        print(f"Saving predictions to: {pred_dir}")

    mp.set_start_method('spawn', force=True)
    result_queue = mp.Queue()
    processes = []

    for rank in range(num_gpus):
        p = mp.Process(
            target=_multigpu_worker,
            args=(
                rank, num_gpus, checkpoint_path, model_cfg, temporal_cfg,
                data_cfg, assignments[rank], result_queue,
                pred_dir if save_predictions else None,
                static_min, dynamic_min
            )
        )
        p.start()
        processes.append(p)

    results = []
    for _ in range(num_gpus):
        results.append(result_queue.get())

    for p in processes:
        p.join()

    for r in results:
        if r['error']:
            raise RuntimeError(f"Worker {r['rank']} failed: {r['error']}")

    # Aggregate Histograms (18x18 matrix: Index 0=Free, 1..17=Classes)
    total_hist = torch.from_numpy(sum(r['hist'] for r in results))
    total_frames = sum(r['total_frames'] for r in results)

    # ==========================================================
    #                 DETAILED METRIC CALCULATION
    # ==========================================================
    EPS = 1e-6
    
    # Per-Class Metrics
    tp = torch.diag(total_hist)
    fp = total_hist.sum(dim=0) - tp
    fn = total_hist.sum(dim=1) - tp

    iou_per_class = tp / (tp + fp + fn + EPS)
    recall_per_class = tp / (tp + fn + EPS)
    precision_per_class = tp / (tp + fp + EPS)

    # Calculate mIoU excluding free class (Index 17)
    # Match official: iou[:-1] excludes last class (free)
    miou = torch.nanmean(iou_per_class[:-1])
    mrecall = torch.nanmean(recall_per_class[:-1])
    mprecision = torch.nanmean(precision_per_class[:-1])

    # Occupancy IoU (match official evaluation/occ_metric.py:66-87)
    # "Occupied" = classes 0-16 (not free=17)
    free_index = 17
    # Sum of entire non-free submatrix: voxels where both GT and Pred are 0-16
    tp_occ = total_hist[:free_index, :free_index].sum()
    # Total excluding free-to-free diagonal
    total_occ = total_hist.sum() - total_hist[free_index, free_index]
    occ_iou = tp_occ / (total_occ + EPS)

    # Recall/Precision for occupancy
    gt_occupied = total_hist[:free_index, :].sum()  # GT in 0-16
    pred_occupied = total_hist[:, :free_index].sum()  # Pred in 0-16
    occ_recall = tp_occ / (gt_occupied + EPS)
    occ_precision = tp_occ / (pred_occupied + EPS)

    print(f"\n{'='*80}")
    print(f"Test Results ({total_frames} frames)")
    print(f"{'='*80}")
    print(f"  mIoU:      {miou:.4f}")
    print(f"  Occ IoU:   {occ_iou:.4f}")
    print(f"  Occ Recall:{occ_recall:.4f}")
    print(f"  Occ Prec:  {occ_precision:.4f}")
    print(f"{'-'*80}")
    print(f"{'Class Name':<25} | {'IoU':<8} | {'Recall':<8} | {'Precision':<8}")
    print(f"{'-'*80}")
    
    for i, class_name in enumerate(OCC_CLASSES):
        # OCC_CLASSES[i] corresponds to GT class i (0=others, 1=barrier, ...)
        if i < len(iou_per_class):
            print(f"{class_name:<25} | {iou_per_class[i]:.4f}   | {recall_per_class[i]:.4f}   | {precision_per_class[i]:.4f}")
    
    print(f"{'='*80}")

    return {'test/miou': miou.item(), 'test/occ_iou': occ_iou.item()}


@hydra.main(version_base=None, config_path="../config", config_name="gausstr_dinov3")
def main(cfg: DictConfig):
    checkpoint_path = cfg.get('checkpoint')
    if not checkpoint_path: raise ValueError("checkpoint required")
    print(f"Loading: {checkpoint_path}")

    output_dir = cfg.get('output_dir', os.path.join(os.path.dirname(checkpoint_path), 'test_results'))
    os.makedirs(output_dir, exist_ok=True)

    temporal_mode = cfg.get('temporal', False)
    save_predictions = cfg.get('save_predictions', False)
    skip_inference = cfg.get('skip_inference', False)
    pred_dir = cfg.get('pred_dir', None)

    model_cfg = OmegaConf.to_container(cfg.model, resolve=True)

    if temporal_mode:
        temporal_cfg = cfg.get('temporal_config', {})
        if not isinstance(temporal_cfg, dict):
            temporal_cfg = OmegaConf.to_container(temporal_cfg, resolve=True)

        method = temporal_cfg.get('method', 'hybrid')
        static_min = float(temporal_cfg.get('static_min_agreement', 2.0))
        dynamic_min = float(temporal_cfg.get('dynamic_min_agreement', 1.0))

        num_gpus = cfg.get('num_gpus', 1)
        if num_gpus == -1:
            num_gpus = torch.cuda.device_count()

        if not skip_inference:
            if num_gpus > 1:
                test_multigpu(
                    checkpoint_path=checkpoint_path,
                    model_cfg=model_cfg,
                    temporal_cfg=temporal_cfg,
                    data_cfg=OmegaConf.to_container(cfg.data, resolve=True),
                    num_gpus=num_gpus,
                    output_dir=output_dir,
                    save_predictions=save_predictions,
                    pred_dir=pred_dir,
                    static_min=static_min,
                    dynamic_min=dynamic_min,
                )
            else:
                # Single GPU path
                test_multigpu(
                    checkpoint_path=checkpoint_path,
                    model_cfg=model_cfg,
                    temporal_cfg=temporal_cfg,
                    data_cfg=OmegaConf.to_container(cfg.data, resolve=True),
                    num_gpus=1,
                    output_dir=output_dir,
                    save_predictions=save_predictions,
                    pred_dir=pred_dir,
                    static_min=static_min,
                    dynamic_min=dynamic_min,
                )
    else:
        # Single-frame mode (default) - use Lightning trainer
        print("Running single-frame test with Lightning trainer...")

        # Load model
        if checkpoint_path.endswith('.ckpt'):
            model = GaussTRLightning.load_from_checkpoint(checkpoint_path, **model_cfg)
        else:
            model = GaussTRLightning(**model_cfg)
            ckpt = torch.load(checkpoint_path, map_location='cpu')
            state_dict = ckpt.get('state_dict', ckpt)
            state_dict = {k.replace('model.', ''): v for k, v in state_dict.items()}
            model.load_state_dict(state_dict, strict=False)

        # Build datamodule
        data_cfg = OmegaConf.to_container(cfg.data, resolve=True)
        datamodule = GaussTRDataModule(**data_cfg)

        # Trainer config
        trainer_cfg = OmegaConf.to_container(cfg.get('trainer', {}), resolve=True)

        # Setup callbacks
        callbacks = []
        if save_predictions:
            if pred_dir is None:
                pred_dir = os.path.join(os.path.dirname(checkpoint_path), 'visualizations')
            callbacks.append(SavePredictionsCallback(pred_dir))
            print(f"Will save predictions to: {pred_dir}/predictions")

        trainer = pl.Trainer(
            accelerator=trainer_cfg.get('accelerator', 'gpu'),
            devices=trainer_cfg.get('devices', 1),
            precision=trainer_cfg.get('precision', '16-mixed'),
            enable_progress_bar=True,
            logger=False,
            limit_test_batches=trainer_cfg.get('limit_test_batches', None),
            callbacks=callbacks,
        )

        # Run test
        results = trainer.test(model, datamodule)

        # Print summary (detailed results already printed by model's on_test_epoch_end)
        if trainer.global_rank == 0:
            print("\n" + "=" * 80)
            print("Summary:")
            print("=" * 80)
            res = results[0]
            print(f"  mIoU:      {res.get('test/miou', 0):.4f}")
            print(f"  Occ IoU:   {res.get('test/occ_iou', 0):.4f}")
            print("=" * 80)


if __name__ == '__main__':
    main()