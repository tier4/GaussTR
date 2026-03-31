#!/usr/bin/env python
"""Training script for PG-Occ on T4 dataset.

Usage:
    python -m scripts.train_pgocc
    python -m scripts.train_pgocc trainer.devices=1 trainer.max_epochs=1
    python -m scripts.train_pgocc data.batch_size=2 trainer.devices=4
"""

# === NUMA Binding Setup (MUST be before any heavy imports) ===
import os
import ctypes as _ctypes

def _bind_numa():
    if os.environ.get('DISABLE_NUMA_BINDING', '0') == '1':
        return
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    gpus_per_numa = int(os.environ.get('GPUS_PER_NUMA', 4))
    numa_node = local_rank // gpus_per_numa
    try:
        libnuma = _ctypes.CDLL('libnuma.so.1')
        libnuma.numa_run_on_node(numa_node)
        libnuma.numa_set_preferred(numa_node)
        print(f"[RANK {local_rank}] Bound to NUMA node {numa_node}")
    except OSError:
        pass

_bind_numa()
del _bind_numa, _ctypes

# === CUDA Toolkit Setup ===
import glob as _glob

def _setup_cuda():
    cuda_home = os.environ.get('CUDA_HOME', '')
    if not cuda_home or not os.path.exists(os.path.join(cuda_home, 'bin', 'nvcc')):
        cuda_dirs = sorted(_glob.glob('/usr/local/cuda-*'), reverse=True)
        if not cuda_dirs and os.path.exists('/usr/local/cuda/bin/nvcc'):
            cuda_dirs = ['/usr/local/cuda']
        for d in cuda_dirs:
            if os.path.exists(os.path.join(d, 'bin', 'nvcc')):
                cuda_home = d
                break
    if cuda_home and os.path.exists(os.path.join(cuda_home, 'bin', 'nvcc')):
        os.environ['CUDA_HOME'] = cuda_home
        cuda_bin = f"{cuda_home}/bin"
        if cuda_bin not in os.environ.get('PATH', ''):
            os.environ['PATH'] = f"{cuda_bin}:{os.environ.get('PATH', '')}"
        cuda_lib = f"{cuda_home}/lib64"
        if cuda_lib not in os.environ.get('LD_LIBRARY_PATH', ''):
            os.environ['LD_LIBRARY_PATH'] = f"{cuda_lib}:{os.environ.get('LD_LIBRARY_PATH', '')}"

_setup_cuda()
del _glob, _setup_cuda

# === W&B Setup ===
if not os.environ.get('WANDB_API_KEY'):
    _key_path = os.path.expanduser('~/.wandb_api_key')
    if os.path.exists(_key_path):
        with open(_key_path) as _f:
            os.environ['WANDB_API_KEY'] = _f.read().strip()

def _load_wandb_config():
    """Load local W&B config (entity, project mappings) from .wandb_config.yaml."""
    import yaml
    _cfg_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '.wandb_config.yaml')
    if os.path.exists(_cfg_path):
        with open(_cfg_path) as f:
            return yaml.safe_load(f) or {}
    return {}
# === End W&B Setup ===

import sys
import warnings
from pathlib import Path

import hydra
import pytorch_lightning as pl
import torch
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.callbacks import (
    LearningRateMonitor,
    ModelCheckpoint,
    RichProgressBar,
)
from pytorch_lightning.loggers import WandbLogger
import wandb
warnings.filterwarnings("ignore", message=".*Default grid_sample and affine_grid behavior.*")
warnings.filterwarnings("ignore", message=".*lr_scheduler.step.*optimizer.step.*")
warnings.filterwarnings("ignore", message=".*Grad strides do not match bucket view strides.*")

torch.set_float32_matmul_precision('high')

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from models.pgocc import PGOccLightning
from dataset import GaussTRDataModule


@hydra.main(
    version_base=None,
    config_path="../config",
    config_name="pgocc_t4"
)
def main(cfg: DictConfig) -> None:
    global_rank = int(os.environ.get('RANK', os.environ.get('LOCAL_RANK', 0)))
    is_main = global_rank == 0

    if is_main:
        print(OmegaConf.to_yaml(cfg))

    seed = cfg.get('seed', 42)
    pl.seed_everything(seed, workers=True)

    hydra_output_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    checkpoint_dir = hydra_output_dir

    run_name = cfg.get('run_name')
    if not run_name:
        run_name = os.path.basename(hydra_output_dir)
        if OmegaConf.is_readonly(cfg):
            OmegaConf.set_readonly(cfg, False)
        cfg.run_name = run_name

    if is_main:
        os.makedirs(checkpoint_dir, exist_ok=True)
        print(f"Run name: {run_name}")
        print(f"Outputs: {checkpoint_dir}")

    # Build model
    model_cfg = OmegaConf.to_container(cfg.model, resolve=True)
    model = PGOccLightning(**model_cfg)

    # Load pretrained checkpoint if specified
    if cfg.get('load_from'):
        if is_main:
            print(f"Loading weights from: {cfg.load_from}")
        checkpoint = torch.load(cfg.load_from, map_location='cpu')
        state_dict = checkpoint.get('state_dict', checkpoint)
        # Filter out shape-mismatched parameters so architectural changes
        # (e.g. different num_queries) don't crash loading.
        model_state = model.state_dict()
        filtered_sd = {
            k: v for k, v in state_dict.items()
            if k in model_state and v.shape == model_state[k].shape
        }
        missing = [k for k in model_state if k not in filtered_sd]
        if is_main and missing:
            print(f"[load_from] Skipped {len(state_dict)-len(filtered_sd)} shape-mismatched params; "
                  f"{len(missing)} model params use random init")
        model.load_state_dict(filtered_sd, strict=False)

    # Build datamodule
    data_cfg = OmegaConf.to_container(cfg.data, resolve=True)
    datamodule = GaussTRDataModule(**data_cfg)

    # Callbacks
    callbacks = [
        ModelCheckpoint(
            dirpath=checkpoint_dir,
            filename='epoch{epoch:02d}-loss{train_loss:.4f}',
            monitor='train_loss',
            mode='min',
            save_top_k=3,
            save_last=True,
            verbose=True,
        ),
        LearningRateMonitor(logging_interval='step'),
    ]
    if cfg.get('use_rich_progress', True):
        callbacks.append(RichProgressBar())

    # Logger — W&B with full config tracking and system metrics
    flat_config = OmegaConf.to_container(cfg, resolve=True)
    wandb_cfg = _load_wandb_config()
    experiment_name = cfg.get('experiment_name', 'pgocc_t4')
    wb_project = wandb_cfg.get('default_projects', {}).get(experiment_name, experiment_name)
    wb_entity = wandb_cfg.get('entity')

    logger = WandbLogger(
        project=wb_project,
        entity=wb_entity,
        name=run_name,
        group=experiment_name,
        save_dir=checkpoint_dir,
        log_model=False,
        config=flat_config,
    )

    # Watch model — auto-log gradients and parameter histograms
    if is_main and wandb.run is not None:
        wandb.watch(model, log='all', log_freq=100)

    # Save config
    if is_main:
        config_path = os.path.join(checkpoint_dir, 'config.yaml')
        with open(config_path, 'w') as f:
            f.write(OmegaConf.to_yaml(cfg))

    # Trainer
    trainer_cfg = OmegaConf.to_container(cfg.get('trainer', {}), resolve=True)
    trainer = pl.Trainer(
        max_epochs=trainer_cfg.get('max_epochs', 8),
        max_steps=trainer_cfg.get('max_steps', -1),
        accelerator=trainer_cfg.get('accelerator', 'gpu'),
        devices=trainer_cfg.get('devices', 'auto'),
        strategy=trainer_cfg.get('strategy', 'ddp'),
        precision=trainer_cfg.get('precision', '16-mixed'),
        callbacks=callbacks,
        logger=logger,
        gradient_clip_val=trainer_cfg.get('gradient_clip_val', 350.0),
        gradient_clip_algorithm=trainer_cfg.get('gradient_clip_algorithm', 'norm'),
        accumulate_grad_batches=trainer_cfg.get('accumulate_grad_batches', 1),
        val_check_interval=trainer_cfg.get('val_check_interval', 1.0),
        check_val_every_n_epoch=trainer_cfg.get('check_val_every_n_epoch', 1),
        limit_train_batches=trainer_cfg.get('limit_train_batches', 1.0),
        limit_val_batches=trainer_cfg.get('limit_val_batches', 0),
        log_every_n_steps=trainer_cfg.get('log_every_n_steps', 50),
        enable_progress_bar=True,
        enable_model_summary=True,
        deterministic=trainer_cfg.get('deterministic', False),
        benchmark=trainer_cfg.get('benchmark', True),
        sync_batchnorm=trainer_cfg.get('sync_batchnorm', False),
        num_sanity_val_steps=trainer_cfg.get('num_sanity_val_steps', 0),
    )

    # Train
    ckpt_path = cfg.get('resume_from', None)
    trainer.fit(model, datamodule, ckpt_path=ckpt_path)

    # Log artifacts, alert, and finish W&B run
    if is_main and wandb.run is not None:
        # Config artifact
        config_path = os.path.join(checkpoint_dir, 'config.yaml')
        if os.path.exists(config_path):
            config_artifact = wandb.Artifact(f'config-{wandb.run.id}', type='config')
            config_artifact.add_file(config_path)
            wandb.log_artifact(config_artifact)

        # Checkpoint artifact
        if trainer.checkpoint_callback:
            ckpt_artifact = wandb.Artifact(f'checkpoints-{wandb.run.id}', type='model')
            has_files = False
            best = trainer.checkpoint_callback.best_model_path
            if best and os.path.exists(best):
                ckpt_artifact.add_file(best, name=os.path.basename(best))
                has_files = True
            last = trainer.checkpoint_callback.last_model_path
            if last and os.path.exists(last) and last != best:
                ckpt_artifact.add_file(last, name=os.path.basename(last))
                has_files = True
            if has_files:
                wandb.log_artifact(ckpt_artifact)

        wandb.alert(
            title=f"Training Complete: {run_name}",
            text=f"PG-Occ run {run_name} finished ({trainer.global_step} steps). Results: {wandb.run.url}",
            level=wandb.AlertLevel.INFO,
        )
        wandb.finish()

    print(f"Training complete. Checkpoints saved to: {checkpoint_dir}")


if __name__ == '__main__':
    main()
