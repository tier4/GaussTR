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

# === MLflow Setup ===
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.environ['MLFLOW_TRACKING_URI'] = f'sqlite:///{_project_root}/mlruns/mlflow.db'
os.environ['MLFLOW_ARTIFACT_ROOT'] = f'{_project_root}/mlruns/mlflow_artifacts'
del _project_root

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
from pytorch_lightning.loggers import MLFlowLogger
import mlflow

mlflow.set_tracking_uri(os.environ['MLFLOW_TRACKING_URI'])

warnings.filterwarnings("ignore", message=".*filesystem tracking backend.*will be deprecated.*")
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
        model.load_state_dict(state_dict, strict=False)

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

    # Logger — use the absolute tracking URI from env (Hydra changes CWD, so relative paths break)
    logger = MLFlowLogger(
        experiment_name=cfg.get('experiment_name', 'pgocc_t4'),
        tracking_uri=os.environ['MLFLOW_TRACKING_URI'],
        run_name=run_name,
        artifact_location=os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            'mlruns', 'mlflow_artifacts',
        ),
        save_dir=None,
    )

    # Save config
    if is_main:
        config_path = os.path.join(checkpoint_dir, 'config.yaml')
        with open(config_path, 'w') as f:
            f.write(OmegaConf.to_yaml(cfg))

    # Trainer
    trainer_cfg = OmegaConf.to_container(cfg.get('trainer', {}), resolve=True)
    trainer = pl.Trainer(
        max_epochs=trainer_cfg.get('max_epochs', 8),
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

    print(f"Training complete. Checkpoints saved to: {checkpoint_dir}")


if __name__ == '__main__':
    main()
