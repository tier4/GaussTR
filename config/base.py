"""Base configuration classes for GaussTR Lightning using dataclasses."""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict, Any


@dataclass
class MLPConfig:
    """Configuration for MLP layers."""
    input_dim: int
    hidden_dim: Optional[int] = None
    output_dim: Optional[int] = None
    num_layers: int = 2
    activation: str = "relu"
    mode: Optional[str] = None  # 'sigmoid' or None
    range: Optional[Tuple[float, float]] = None


@dataclass
class NeckConfig:
    """Configuration for ViTDetFPN neck."""
    in_channels: int = 512
    out_channels: int = 256
    norm_type: str = "LN2d"


@dataclass
class SelfAttnConfig:
    """Configuration for self-attention in decoder."""
    embed_dims: int = 256
    num_heads: int = 8
    dropout: float = 0.0


@dataclass
class CrossAttnConfig:
    """Configuration for cross-attention in decoder."""
    embed_dims: int = 256
    num_levels: int = 4


@dataclass
class FFNConfig:
    """Configuration for feed-forward network in decoder."""
    embed_dims: int = 256
    feedforward_channels: int = 2048


@dataclass
class DecoderLayerConfig:
    """Configuration for a single decoder layer."""
    self_attn_cfg: SelfAttnConfig = field(default_factory=SelfAttnConfig)
    cross_attn_cfg: CrossAttnConfig = field(default_factory=CrossAttnConfig)
    ffn_cfg: FFNConfig = field(default_factory=FFNConfig)


@dataclass
class DecoderConfig:
    """Configuration for GaussTRDecoder."""
    num_layers: int = 3
    return_intermediate: bool = True
    layer_cfg: DecoderLayerConfig = field(default_factory=DecoderLayerConfig)
    post_norm_cfg: Optional[Dict] = None


@dataclass
class VoxelizerConfig:
    """Configuration for GaussianVoxelizer."""
    vol_range: List[float] = field(
        default_factory=lambda: [-40, -40, -1, 40, 40, 5.4]
    )
    voxel_size: float = 0.4


@dataclass
class GaussHeadConfig:
    """Configuration for GaussTRHead."""
    embed_dims: int = 256
    feat_dims: int = 512
    reduce_dims: int = 128
    image_shape: Tuple[int, int] = (432, 768)
    patch_size: int = 16
    depth_limit: float = 51.2
    text_protos: Optional[str] = "ckpts/text_proto_embeds_clip.pth"
    prompt_denoising: bool = True
    num_segment_classes: int = 26
    voxelizer: VoxelizerConfig = field(default_factory=VoxelizerConfig)

    # MLP configs for each head
    opacity_head: MLPConfig = field(default_factory=lambda: MLPConfig(
        input_dim=256, output_dim=1, mode="sigmoid"
    ))
    feature_head: MLPConfig = field(default_factory=lambda: MLPConfig(
        input_dim=256, output_dim=512
    ))
    scale_head: MLPConfig = field(default_factory=lambda: MLPConfig(
        input_dim=256, output_dim=3, mode="sigmoid", range=(1.0, 16.0)
    ))
    regress_head: MLPConfig = field(default_factory=lambda: MLPConfig(
        input_dim=256, output_dim=3
    ))
    segment_head: Optional[MLPConfig] = field(default_factory=lambda: MLPConfig(
        input_dim=128, output_dim=26
    ))


@dataclass
class DataPreprocessorConfig:
    """Configuration for data preprocessing."""
    mean: List[float] = field(default_factory=lambda: [123.675, 116.28, 103.53])
    std: List[float] = field(default_factory=lambda: [58.395, 57.12, 57.375])


@dataclass
class DataConfig:
    """Configuration for data loading."""
    data_root: str = "data/nuscenes/"
    depth_root: str = "data/nuscenes_unidepth"
    feats_root: str = "data/nuscenes_featup"
    sem_seg_root: Optional[str] = "data/nuscenes_sam3"

    train_ann: str = "nuscenes_infos_train.pkl"
    val_ann: str = "nuscenes_infos_val.pkl"

    input_size: Tuple[int, int] = (432, 768)
    resize_lim: Tuple[float, float] = (0.48, 0.48)
    num_views: int = 6

    batch_size: int = 2
    num_workers: int = 4
    pin_memory: bool = True
    persistent_workers: bool = True

    # Camera names for data loading
    camera_names: List[str] = field(default_factory=lambda: [
        "CAM_FRONT", "CAM_FRONT_LEFT", "CAM_FRONT_RIGHT",
        "CAM_BACK", "CAM_BACK_RIGHT", "CAM_BACK_LEFT"
    ])


@dataclass
class TrainConfig:
    """Configuration for training."""
    max_epochs: int = 24
    val_interval: int = 1

    # Optimizer
    learning_rate: float = 2e-4
    weight_decay: float = 5e-3

    # Gradient clipping
    gradient_clip_val: float = 35.0
    gradient_clip_algorithm: str = "norm"

    # LR scheduler
    warmup_iters: int = 200
    warmup_factor: float = 1e-3
    lr_milestones: List[int] = field(default_factory=lambda: [16])
    lr_gamma: float = 0.1

    # Mixed precision
    precision: str = "16-mixed"

    # Checkpointing
    save_top_k: int = 3
    monitor_metric: str = "miou"

    # Logging
    log_every_n_steps: int = 50


@dataclass
class GaussTRConfig:
    """Main configuration for GaussTR Lightning."""
    # Model components
    num_queries: int = 300
    embed_dims: int = 256
    feat_dims: int = 512

    neck: NeckConfig = field(default_factory=NeckConfig)
    decoder: DecoderConfig = field(default_factory=DecoderConfig)
    gauss_head: GaussHeadConfig = field(default_factory=GaussHeadConfig)
    data_preprocessor: DataPreprocessorConfig = field(
        default_factory=DataPreprocessorConfig
    )

    # Data
    data: DataConfig = field(default_factory=DataConfig)

    # Training
    train: TrainConfig = field(default_factory=TrainConfig)

    # Work directory
    work_dir: str = "work_dirs/gausstr_lightning"

    # Number of GPUs
    num_gpus: int = 8

    @classmethod
    def from_yaml(cls, yaml_path: str) -> "GaussTRConfig":
        """Load configuration from YAML file using OmegaConf."""
        from omegaconf import OmegaConf

        # Load YAML
        cfg = OmegaConf.load(yaml_path)

        # Convert to dataclass
        return cls(**OmegaConf.to_container(cfg, resolve=True))

    @classmethod
    def featup(cls) -> "GaussTRConfig":
        """Create FeatUp configuration preset."""
        config = cls()
        config.feat_dims = 512
        config.neck.in_channels = 512
        config.gauss_head.feat_dims = 512
        config.gauss_head.image_shape = (432, 768)
        config.gauss_head.patch_size = 16
        config.data.feats_root = "data/nuscenes_featup"
        config.data.input_size = (432, 768)
        return config

    @classmethod
    def talk2dino(cls) -> "GaussTRConfig":
        """Create Talk2DINO configuration preset."""
        config = cls()
        config.feat_dims = 768
        config.neck.in_channels = 768
        config.gauss_head.feat_dims = 768
        config.gauss_head.image_shape = (504, 896)
        config.gauss_head.patch_size = 14
        config.gauss_head.text_protos = "ckpts/text_proto_embeds_talk2dino.pth"
        config.data.input_size = (504, 896)
        return config
