#!/usr/bin/env python
"""Testing/evaluation script for GaussTR Lightning.

Usage:
    python -m gausstr_lightning.scripts.test checkpoint=path/to/checkpoint.ckpt
    python -m gausstr_lightning.scripts.test checkpoint=ckpts/gausstr_featup.pth
"""

import os
import sys
from pathlib import Path

import hydra
import pytorch_lightning as pl
import torch
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.profilers import PyTorchProfiler, SimpleProfiler, AdvancedProfiler

# Enable Tensor Core optimization for better performance on supported GPUs (H100, A100, etc.)
torch.set_float32_matmul_precision('high')

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from models import GaussTRLightning
from dataset import GaussTRDataModule


@hydra.main(
    version_base=None,
    config_path="../config",
    config_name="gausstr_featup"
)
def main(cfg: DictConfig) -> None:
    """Main testing function.

    Args:
        cfg: Hydra configuration.
    """
    # Print config
    print(OmegaConf.to_yaml(cfg))

    # Checkpoint path is required
    checkpoint_path = cfg.get('checkpoint')
    if not checkpoint_path:
        raise ValueError("checkpoint path is required. Use: checkpoint=path/to/model.ckpt")

    print(f"Loading checkpoint from: {checkpoint_path}")

    # Build model
    model_cfg = OmegaConf.to_container(cfg.model, resolve=True)

    # Check if this is a Lightning checkpoint or MMEngine checkpoint
    if checkpoint_path.endswith('.ckpt'):
        # Lightning checkpoint
        model = GaussTRLightning.load_from_checkpoint(
            checkpoint_path,
            **model_cfg
        )
    else:
        # Assume MMEngine checkpoint format
        model = GaussTRLightning(**model_cfg)
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        state_dict = checkpoint.get('state_dict', checkpoint)
        # Remove 'model.' prefix if present
        state_dict = {
            k.replace('model.', ''): v
            for k, v in state_dict.items()
        }
        model.load_state_dict(state_dict, strict=False)

    # Build datamodule
    data_cfg = OmegaConf.to_container(cfg.data, resolve=True)
    datamodule = GaussTRDataModule(**data_cfg)

    # Build trainer for testing
    trainer_cfg = OmegaConf.to_container(cfg.get('trainer', {}), resolve=True)

    # Setup profiler if requested
    profiler_type = cfg.get('profiler', None)
    profiler = None
    if profiler_type == 'simple':
        profiler = SimpleProfiler(dirpath=".", filename="profiler_simple")
    elif profiler_type == 'advanced':
        profiler = AdvancedProfiler(dirpath=".", filename="profiler_advanced")
    elif profiler_type == 'pytorch':
        profiler = PyTorchProfiler(
            dirpath=".",
            filename="profiler_pytorch",
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=1),
            on_trace_ready=torch.profiler.tensorboard_trace_handler("./profiler_logs"),
            record_shapes=True,
            profile_memory=True,
            with_stack=True,
        )

    trainer = pl.Trainer(
        accelerator=trainer_cfg.get('accelerator', 'gpu'),
        devices=trainer_cfg.get('devices', 1),  # Use 1 GPU for testing by default
        precision=trainer_cfg.get('precision', '16-mixed'),
        enable_progress_bar=True,
        logger=False,
        limit_test_batches=trainer_cfg.get('limit_test_batches', None),
        profiler=profiler,
    )

    # Run test
    results = trainer.test(model, datamodule)

    # Only print on rank 0 to avoid duplicate output in multi-GPU testing
    if trainer.global_rank == 0:
        print("\n" + "=" * 60)
        print("Test Results:")
        print("=" * 60)
        for key, value in results[0].items():
            print(f"  {key}: {value:.4f}")
        print("=" * 60)


if __name__ == '__main__':
    main()
