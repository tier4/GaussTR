#!/usr/bin/env python
"""Testing/evaluation script for GaussTR Lightning.

Usage:
    # Basic test
    python -m scripts.test checkpoint=path/to/checkpoint.ckpt

    # Multi-GPU test
    python -m scripts.test checkpoint=path/to/checkpoint.ckpt trainer.devices=8

    # Save predictions for visualization
    python -m scripts.test checkpoint=path/to/checkpoint.ckpt save_predictions=true
"""

import os
import sys
from pathlib import Path

import hydra
import pytorch_lightning as pl
import torch
import numpy as np
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.profilers import PyTorchProfiler, SimpleProfiler, AdvancedProfiler
from pytorch_lightning.callbacks import Callback

# Enable Tensor Core optimization for better performance on supported GPUs (H100, A100, etc.)
torch.set_float32_matmul_precision('high')

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from models import GaussTRLightning
from dataset import GaussTRDataModule


class SavePredictionsCallback(Callback):
    """Callback to save predictions during testing.

    Saves predictions using timestamp as filename for proper ordering.
    """

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

        # Get predictions from outputs
        preds = outputs.get('preds')
        if preds is None:
            return

        if isinstance(preds, torch.Tensor):
            preds = preds.cpu().numpy()

        # Get GT, mask, and timestamps
        gt_occ = batch.get('gt_occ')
        mask = batch.get('mask_camera')
        timestamps = batch.get('timestamp', [])

        batch_size = preds.shape[0]

        for i in range(batch_size):
            # Use timestamp as filename for proper ordering
            if timestamps is not None and len(timestamps) > i:
                ts = timestamps[i]
                if isinstance(ts, torch.Tensor):
                    ts = ts.item()
                # Format: 10 digits before decimal, 6 after (microseconds)
                filename = f"{ts:.6f}"
            else:
                # Fallback: use sample_idx
                sample_idx = batch.get('sample_idx')
                if sample_idx is not None:
                    idx = sample_idx[i].item() if isinstance(sample_idx[i], torch.Tensor) else sample_idx[i]
                    filename = f"{idx:06d}"
                else:
                    # Last resort: compute index
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

            np.savez_compressed(
                os.path.join(self.pred_dir, f'{filename}.npz'),
                **save_dict
            )

    def on_test_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        """Print summary after test."""
        if trainer.global_rank == 0:
            num_files = len([f for f in os.listdir(self.pred_dir) if f.endswith('.npz')])
            print(f"\nSaved {num_files} predictions to: {self.pred_dir}")


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

    # Setup callbacks
    callbacks = []

    # Save predictions if requested
    save_predictions = cfg.get('save_predictions', False)
    if save_predictions:
        # Output dir: same as checkpoint dir or specified
        output_dir = cfg.get('output_dir', None)
        if output_dir is None:
            output_dir = os.path.join(os.path.dirname(checkpoint_path), 'visualizations')
        callbacks.append(SavePredictionsCallback(output_dir))
        print(f"Will save predictions to: {output_dir}/predictions")

    trainer = pl.Trainer(
        accelerator=trainer_cfg.get('accelerator', 'gpu'),
        devices=trainer_cfg.get('devices', 1),  # Use 1 GPU for testing by default
        precision=trainer_cfg.get('precision', '16-mixed'),
        enable_progress_bar=True,
        logger=False,
        limit_test_batches=trainer_cfg.get('limit_test_batches', None),
        profiler=profiler,
        callbacks=callbacks,
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
