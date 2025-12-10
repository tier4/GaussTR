<div align="center">

# [GaussTR](): Foundation Model-Aligned [Gauss]()ian [Tr]()ansformer for Self-Supervised 3D Spatial Understanding

## PyTorch Lightning Implementation

[Haoyi Jiang](https://scholar.google.com/citations?user=_45BVtQAAAAJ)<sup>1</sup>, Liu Liu<sup>2</sup>, [Tianheng Cheng](https://scholar.google.com/citations?user=PH8rJHYAAAAJ)<sup>1</sup>, Xinjie Wang<sup>2</sup>,
[Tianwei Lin](https://wzmsltw.github.io/)<sup>2</sup>, Zhizhong Su<sup>2</sup>, Wenyu Liu<sup>1</sup>, [Xinggang Wang](https://xwcv.github.io/)<sup>1</sup><br>
<sup>1</sup>Huazhong University of Science & Technology, <sup>2</sup>Horizon Robotics

[**CVPR 2025**](https://openaccess.thecvf.com/content/CVPR2025/papers/Jiang_GaussTR_Foundation_Model-Aligned_Gaussian_Transformer_for_Self-Supervised_3D_Spatial_Understanding_CVPR_2025_paper.pdf)

[![Project page](https://img.shields.io/badge/project%20page-hustvl.github.io%2FGaussTR-blue)](https://hustvl.github.io/GaussTR/)
[![arXiv](https://img.shields.io/badge/arXiv-2412.13193-red?logo=arXiv&logoColor=red)](https://arxiv.org/abs/2412.13193)
[![License: MIT](https://img.shields.io/github/license/hustvl/GaussTR)](LICENSE)

</div>

> **Note:** This is a PyTorch Lightning reimplementation of [GaussTR](https://github.com/hustvl/GaussTR). The original implementation uses MMEngine/MMDetection3D. This version provides a cleaner, standalone PyTorch Lightning interface without MMEngine dependencies.

## News

* ***Feb 27 '25:*** Our paper has been accepted at CVPR 2025. 🎉
* ***Feb 11 '25:*** Released the model integrated with Talk2DINO, achieving new state-of-the-art results.
* ***Dec 17 '24:*** Released our arXiv paper along with the source code.

## Setup

### Installation

We use [uv](https://docs.astral.sh/uv/) for fast, reproducible dependency management:

```bash
# Install uv if not already installed
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create virtual environment and install dependencies
uv venv
source .venv/bin/activate
uv pip install -r pyproject.toml

# Optional: install with tensorboard or wandb support
uv pip install tensorboard
uv pip install wandb
```

### Build CUDA Extensions

Two custom CUDA extensions must be compiled before running. Ensure you have CUDA toolkit installed and `nvcc` available in your PATH:

```bash
# Verify CUDA is available
nvcc --version

# Build CUDA voxelizer (fast 3D Gaussian voxelization)
cd models/cuda_voxelizer
python setup.py build_ext --inplace
cd ../..

# Build MultiScaleDeformableAttention (deformable attention CUDA kernel)
cd models/cuda_msda
python setup.py build_ext --inplace
cd ../..
```

After successful compilation, you should see `.so` files in each directory:
- `models/cuda_voxelizer/voxelize_cuda_ext.cpython-3XX-x86_64-linux-gnu.so`
- `models/cuda_msda/MultiScaleDeformableAttention.cpython-3XX-x86_64-linux-gnu.so`

**Supported GPU architectures:** V100 (sm_70), T4 (sm_75), A100 (sm_80), RTX 3090 (sm_86), RTX 4090 (sm_89), H100 (sm_90)

### Dataset Preparation

1. Download or manually prepare the nuScenes dataset following the instructions in the [mmdetection3d docs](https://mmdetection3d.readthedocs.io/en/latest/user_guides/dataset_prepare.html#nuscenes) and place it in `data/nuscenes`.
   **NOTE:** Please be aware that we are using the latest OpenMMLab V2.0 format. If you've previously prepared the nuScenes dataset from other repositories, it might be outdated. For more information, please refer to [update_infos_to_v2.py](https://github.com/open-mmlab/mmdetection3d/blob/main/tools/dataset_converters/update_infos_to_v2.py).
2. **Update the prepared dataset `.pkl` files with the `scene_idx` field to match the occupancy ground truths:**

    ```bash
    python tools/update_data.py nuscenes --root-path ./data/nuscenes --out-dir ./data/nuscenes --extra-tag nuscenes
    ```

3. Download the occupancy ground truth data from [CVPR2023-3D-Occupancy-Prediction](https://github.com/CVPR2023-3D-Occupancy-Prediction/CVPR2023-3D-Occupancy-Prediction) and place it in `data/nuscenes/gts`.
4. Generate features and rendering targets:

    * **[Depth Generation]** Choose one of the following options:
      - **Metric3D** (default): `PYTHONPATH=. python tools/generate_depth.py`
      - **Depth Anything V3** (recommended): See original repo for details.
    * **[For GaussTR-FeatUp Only]** Navigate to the [FeatUp](https://github.com/mhamilton723/FeatUp) repository and run `python tools/generate_featup.py`.

### CLIP Text Embeddings

Download the pre-generated CLIP text embeddings from the [Releases](https://github.com/hustvl/GaussTR/releases/) page and place in `ckpts/text_proto_embeds_clip.pth`.

## Usage

|                               Model                               |  IoU  |  mIoU |                                                 Checkpoint                                                 |
| ----------------------------------------------------------------- | ----- | ----- | ---------------------------------------------------------------------------------------------------------- |
| [GaussTR-FeatUp](configs/gausstr_featup.py)                       | 45.19 | 11.70 | [checkpoint](https://github.com/hustvl/GaussTR/releases/download/v1.0/gausstr_featup_e24_miou11.70.pth)    |
| [GaussTR-Talk2DINO](configs/gausstr_talk2dino.py)<sup>*New*</sup> | 44.54 | 12.27 | [checkpoint](https://github.com/hustvl/GaussTR/releases/download/v1.0/gausstr_talk2dino_e20_miou12.27.pth) |

### Training

```bash
# Basic training (uses config/gausstr_featup.yaml)
python -m scripts.train

# With Hydra overrides
python -m scripts.train \
    data.batch_size=4 \
    trainer.devices=8 \
    trainer.precision="16-mixed" \
    trainer.max_epochs=12
```

**Tip:** Due to the current lack of optimization for voxelization operations, evaluation during training can be time-consuming. To accelerate training, consider evaluating using the `mini_train` set or reducing the evaluation frequency.

### Testing

```bash
# Test with Lightning checkpoint
python -m scripts.test checkpoint=path/to/model.ckpt

# Test with original MMEngine checkpoint
python -m scripts.test checkpoint=ckpts/gausstr_featup.pth
```

### Experiment Tracking

This implementation uses MLflow for experiment tracking by default:

```bash
# Launch MLflow UI
mlflow ui --backend-store-uri sqlite:////mnt/nvme0/gausstr_lightning/mlflow.db
```

## Citation

If our paper and code contribute to your research, please consider starring this repository :star: and citing our work:

```BibTeX
@inproceedings{GaussTR,
    title     = {GaussTR: Foundation Model-Aligned Gaussian Transformer for Self-Supervised 3D Spatial Understanding},
    author    = {Haoyi Jiang and Liu Liu and Tianheng Cheng and Xinjie Wang and Tianwei Lin and Zhizhong Su and Wenyu Liu and Xinggang Wang},
    year      = 2025,
    booktitle = {CVPR}
}
```

## Acknowledgements

This project is built upon the pioneering work of [FeatUp](https://github.com/mhamilton723/FeatUp), [Talk2DINO](https://github.com/lorebianchi98/Talk2DINO), [MaskCLIP](https://github.com/chongzhou96/MaskCLIP) and [gsplat](https://github.com/nerfstudio-project/gsplat). We extend our gratitude to these projects for their contributions to the community.

## License

Released under the [MIT](LICENSE) License.
