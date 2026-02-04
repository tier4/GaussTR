# T4 Dataset Processing Scripts

Utilities for processing the T4 dataset for use with GaussTR.

## Overview

| Script | Description |
|--------|-------------|
| `convert_annotations.py` | Convert T4 JSON annotations to GaussTR pickle format |
| `generate_depth.py` | Generate per-camera depth maps from LiDAR point clouds |
| `precompute_pca.py` | Compute PCA for feature dimensionality reduction (multi-GPU) |
| `infer.py` | Run inference and generate visualizations |

## Data Preparation Pipeline

Run scripts in this order:

```
1. convert_annotations.py  →  Creates train/val pickle files
2. generate_depth.py       →  Creates depth maps from LiDAR
3. precompute_pca.py       →  (Optional) Compute PCA for features
4. infer.py                →  Visualize predictions
```

---

## 1. Convert Annotations

Converts T4 JSON annotation format to GaussTR-compatible pickle files containing camera calibration, ego poses, and file paths.

```bash
python -m scripts.t4.convert_annotations \
    --input-dir /path/to/T4_datasets \
    --output-dir /path/to/T4_processed \
    --val-ratio 0.1
```

**Arguments:**
- `--input-dir`: Root directory containing T4 dataset chunks
- `--output-dir`: Output directory for pickle files
- `--val-ratio`: Fraction of scenes for validation (default: 0.1)
- `--seed`: Random seed for reproducible splits (default: 42)
- `--camera-names`: Comma-separated camera list (default: all 6 cameras)
- `--allow-missing-cams`: Keep samples even if some cameras are missing
- `--absolute-paths`: Store absolute paths instead of relative
- `--no-val`: Output single file without train/val split

**Output:**
- `{output_dir}/t4_infos_train.pkl`
- `{output_dir}/t4_infos_val.pkl`

---

## 2. Generate Depth Maps

Projects LiDAR point clouds to camera views to create pseudo ground-truth depth maps.

```bash
python -m scripts.t4.generate_depth \
    --ann-file /path/to/t4_infos_train.pkl \
    --output-dir /path/to/T4_lidar_depth \
    --data-root /path/to/T4_datasets \
    --num-workers 8
```

**Arguments:**
- `--ann-file`: Path to annotation pickle from step 1
- `--output-dir`: Output directory for depth maps
- `--data-root`: Root path for resolving relative file paths
- `--num-workers`: Number of parallel workers (default: 4)
- `--min-depth`: Minimum valid depth in meters (default: 0.1)
- `--max-depth`: Maximum valid depth in meters (default: 80.0)
- `--interpolation`: `none` or `nearest` for sparse depth (default: none)
- `--overwrite`: Overwrite existing depth maps

**Output:**
- `{output_dir}/{chunk_name}/{camera_name}/{frame_id}.npz` (sparse compressed format)

---

## 3. Precompute PCA (Optional)

Computes PCA transformation for reducing feature dimensionality. Uses distributed computation across multiple GPUs for speed.

**Requirements:** cuML environment with RAPIDS

```bash
# Activate cuML environment first
source /path/to/cuml_venv/bin/activate

python -m scripts.t4.precompute_pca \
    --feats_root /path/to/T4_features \
    --camera CAM_FRONT_WIDE \
    --reduce_dims 128 \
    --output ckpts/t4_pca_front_wide.pth \
    --n_gpus 8
```

**Arguments:**
- `--feats_root`: Root directory containing extracted features
- `--camera`: Camera name to process (default: CAM_FRONT_WIDE)
- `--num_samples`: Number of files to sample (-1 for all)
- `--reduce_dims`: Target dimensionality (default: 128)
- `--output`: Output path for PCA weights
- `--spatial_subsample`: Spatial subsampling factor (default: 4)
- `--n_gpus`: Number of GPUs to use (default: 8)

**Output:**
- PyTorch checkpoint with `v` (projection matrix), `mean`, `s` (singular values)

---

## 4. Inference & Visualization

Runs model inference on T4 data and generates visualizations.

```bash
python -m scripts.t4.infer \
    --checkpoint /path/to/checkpoint.ckpt \
    --num-samples 50 \
    --split val
```

**Arguments:**
- `--checkpoint`: Path to model checkpoint
- `--config`: Config YAML (default: uses checkpoint_dir/config.yaml)
- `--num-samples`: Number of samples to visualize (default: 10)
- `--output-dir`: Output directory (default: checkpoint_dir/visualizations)
- `--batch-size`: Batch size for inference (default: 1)
- `--device`: Device to run on (default: cuda:0)
- `--split`: Dataset split: train, val, or test (default: val)

**Output:**
- `{output_dir}/{sample_id}_bev.jpg` - Bird's eye view
- `{output_dir}/{sample_id}_3d.jpg` - 3D render
- `{output_dir}/{sample_id}_composite.jpg` - Combined visualization

---

## T4 Camera Configuration

The T4 dataset uses 6 cameras:

| Camera | Description |
|--------|-------------|
| `CAM_FRONT` | Front narrow FOV |
| `CAM_FRONT_WIDE` | Front wide FOV |
| `CAM_FRONT_LEFT_WIDE` | Front-left wide FOV |
| `CAM_FRONT_RIGHT_WIDE` | Front-right wide FOV |
| `CAM_BACK_LEFT_WIDE` | Rear-left wide FOV |
| `CAM_BACK_RIGHT_WIDE` | Rear-right wide FOV |
