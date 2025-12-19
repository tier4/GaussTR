# GaussTR Visualization Module

Visualization tools for 3D occupancy predictions from GaussTR.

## Installation

The visualization module requires additional dependencies beyond the core GaussTR requirements:

```bash
# Using uv (recommended)
uv pip install opencv-python scipy matplotlib

# Optional: Open3D for high-quality 3D rendering (slower than PyTorch GPU)
uv pip install open3d
```

## Two Visualization Workflows

### 1. Development Workflow: `visualize.py`

For quick visualization during development. Loads model, runs inference, generates visualizations in one step.

```bash
# Basic usage
python -m scripts.visualize checkpoint=path/to/checkpoint.ckpt

# Customize output
python -m scripts.visualize checkpoint=path/to/checkpoint.ckpt \
    vis.num_samples=100 \
    vis.mode=composite \
    vis.output_format=video
```

**Pros**: Simple, single command
**Cons**: Requires GPU, can't parallelize inference

### 2. Production Workflow: `test.py` + `visualize_standalone.py`

For large-scale evaluation. Separates inference (multi-GPU) from visualization (CPU).

```bash
# Step 1: Run distributed test and save predictions
python -m scripts.test checkpoint=path/to/checkpoint.ckpt \
    save_predictions=true \
    trainer.devices=8

# Step 2: Generate visualizations from saved predictions (no GPU needed)
python -m scripts.visualize_standalone \
    --pred-dir path/to/visualizations \
    --both  # cameras + 3D + BEV + legend
```

**Pros**:
- Multi-GPU inference support
- Re-run visualization without re-running inference
- Visualization can run on CPU-only machine

**Cons**: Two-step process

## Visualization Modes

| Mode | Description | Output |
|------|-------------|--------|
| `--bev` | Camera surround view + BEV occupancy | `bev/*.jpg` |
| `--3d` | Camera surround view + 3D render | `3d/*.jpg` |
| `--both` | Full composite (cameras + 3D + BEV + legend) | `both/*.jpg` |
| `--video` | Generate video from visualizations | `visualization.mp4` |

## Module API

### BEV Visualization

```python
from visualization import draw_bev_occupancy, save_bev_image

# Draw BEV from 3D occupancy grid
bev_image = draw_bev_occupancy(
    occupancy,           # (X, Y, Z) numpy array with class labels
    output_size=(800, 800),
    draw_ego=True,       # Draw ego vehicle marker at center
)

# Save directly to file
save_bev_image(occupancy, "output.jpg")
```

### 3D Visualization

```python
from visualization import render_occupancy_3d, visualize_occupancy_3d

# Render to file (headless, GPU-accelerated)
render_occupancy_3d(
    occupancy,
    "output.jpg",
    mask=camera_mask,    # Optional visibility mask
    image_size=(1280, 720),
)

# Interactive visualization (requires display)
visualize_occupancy_3d(occupancy)
```

### Composite Visualization

```python
from visualization import (
    create_composite_visualization,
    create_comparison_visualization,
    create_full_composite_visualization,
)

# Camera images dict: {'CAM_FRONT': img, 'CAM_FRONT_LEFT': img, ...}
images = load_camera_images(data_info, data_root)

# Prediction only
composite = create_composite_visualization(images, pred_occ)

# Prediction vs Ground Truth comparison
comparison = create_comparison_visualization(images, pred_occ, gt_occ)

# Full layout: cameras + 3D + BEV + legend
full = create_full_composite_visualization(images, pred_occ, gt_occ)
```

### Lightning Callbacks

```python
from visualization import VisualizationCallback, ValidationVisualizationCallback

# Add to trainer for automatic visualization after test
callback = VisualizationCallback(
    num_samples=50,
    mode='composite',
    output_format='image',
)

# Or enable via config
# visualization:
#   enabled: true
#   num_samples: 50
#   mode: composite
```

## Rendering Backend Priority

The 3D renderer tries backends in this order:

1. **PyTorch GPU** (fastest) - Custom CUDA point cloud renderer
2. **Matplotlib** (fallback) - CPU-based scatter plot

Open3D is available but not used by default due to slower performance.

## Output Directory Structure

```
visualizations/
├── predictions/     # Raw .npz files (pred, gt, mask)
├── bev/            # BEV composite images
├── 3d/             # 3D composite images
├── both/           # Full composite images
└── visualization.mp4  # Video (if --video flag)
```

## Color Map

Uses nuScenes 18-class occupancy color scheme:

| Class | Color | Class | Color |
|-------|-------|-------|-------|
| barrier | orange | bicycle | red |
| bus | yellow | car | blue |
| construction | dark orange | motorcycle | magenta |
| pedestrian | pink | traffic_cone | dark red |
| trailer | cyan | truck | olive |
| driveable | purple | other_flat | dark cyan |
| sidewalk | coral | terrain | wheat |
| manmade | gray | vegetation | green |
| free | white | | |
