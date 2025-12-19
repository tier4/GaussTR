#!/usr/bin/env python
"""Training script for GaussTR Lightning.

Usage:
    python -m gausstr_lightning.scripts.train
    python -m gausstr_lightning.scripts.train model=talk2dino
    python -m gausstr_lightning.scripts.train trainer.max_epochs=12

With Hydra overrides:
    python -m gausstr_lightning.scripts.train \
        data.batch_size=4 \
        trainer.devices=8 \
        trainer.precision="16-mixed"
"""

# === NUMA Binding Setup (MUST be before any heavy imports) ===
# Binds each rank to its NUMA node for optimal memory locality on multi-socket systems
import os
import ctypes as _ctypes

def _bind_numa():
    """Bind process to NUMA node based on LOCAL_RANK.

    Typical H100 HGX/DGX topology:
    - NUMA 0: GPUs 0-3 (LOCAL_RANK 0-3)
    - NUMA 1: GPUs 4-7 (LOCAL_RANK 4-7)

    Environment variables:
    - GPUS_PER_NUMA: GPUs per NUMA node (default: 4)
    - DISABLE_NUMA_BINDING: Set to "1" to disable
    """
    if os.environ.get('DISABLE_NUMA_BINDING', '0') == '1':
        return

    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    gpus_per_numa = int(os.environ.get('GPUS_PER_NUMA', 4))
    numa_node = local_rank // gpus_per_numa

    try:
        libnuma = _ctypes.CDLL('libnuma.so.1')
        # Bind CPU execution to NUMA node
        libnuma.numa_run_on_node(numa_node)
        # Prefer memory allocation from this NUMA node
        libnuma.numa_set_preferred(numa_node)
        print(f"[RANK {local_rank}] Bound to NUMA node {numa_node}")
    except OSError:
        pass  # libnuma not available (non-NUMA system or missing library)

_bind_numa()
del _bind_numa, _ctypes
# === End NUMA Binding Setup ===

# === CUDA Toolkit Setup (MUST be before any imports that use CUDA) ===
# This ensures gsplat can find nvcc in subprocess workers
import glob as _glob

def _setup_cuda():
    """Auto-detect and setup CUDA toolkit if not already configured."""
    cuda_home = os.environ.get('CUDA_HOME', '')

    # If CUDA_HOME not set, find it
    if not cuda_home or not os.path.exists(os.path.join(cuda_home, 'bin', 'nvcc')):
        cuda_dirs = sorted(_glob.glob('/usr/local/cuda-*'), reverse=True)
        if not cuda_dirs and os.path.exists('/usr/local/cuda/bin/nvcc'):
            cuda_dirs = ['/usr/local/cuda']
        for d in cuda_dirs:
            if os.path.exists(os.path.join(d, 'bin', 'nvcc')):
                cuda_home = d
                break

    # Ensure PATH includes cuda/bin (even if CUDA_HOME was already set)
    if cuda_home and os.path.exists(os.path.join(cuda_home, 'bin', 'nvcc')):
        os.environ['CUDA_HOME'] = cuda_home
        cuda_bin = f"{cuda_home}/bin"
        if cuda_bin not in os.environ.get('PATH', ''):
            os.environ['PATH'] = f"{cuda_bin}:{os.environ.get('PATH', '')}"
        cuda_lib = f"{cuda_home}/lib64"
        if cuda_lib not in os.environ.get('LD_LIBRARY_PATH', ''):
            os.environ['LD_LIBRARY_PATH'] = f"{cuda_lib}:{os.environ.get('LD_LIBRARY_PATH', '')}"

_setup_cuda()

# DEBUG: Print env after setup
_local_rank = os.environ.get('LOCAL_RANK', '0')
print(f"[RANK {_local_rank}] CUDA_HOME={os.environ.get('CUDA_HOME', 'NOT SET')}, nvcc in PATH: {'/usr/local/cuda' in os.environ.get('PATH', '')}")

del _glob, _setup_cuda, _local_rank
# === End CUDA Setup ===

# Set MLflow tracking URI and artifact root via environment variables BEFORE any imports
# This ensures all DDP worker processes use the database instead of creating mlruns/
# Use direct assignment (not setdefault) to override any existing value
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
    Callback,
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
    RichProgressBar,
)
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger, MLFlowLogger
import mlflow

# Explicitly set tracking URI right after import to prevent mlruns/ folder creation
mlflow.set_tracking_uri(os.environ['MLFLOW_TRACKING_URI'])

# Suppress MLflow filesystem deprecation warning
warnings.filterwarnings("ignore", message=".*filesystem tracking backend.*will be deprecated.*")
# Suppress grid_sample align_corners warning
warnings.filterwarnings("ignore", message=".*Default grid_sample and affine_grid behavior.*")
# Suppress lr_scheduler.step() warning - this is expected in Lightning with step-based scheduling
# The warning is misleading: Lightning handles optimizer/scheduler ordering correctly internally
warnings.filterwarnings("ignore", message=".*lr_scheduler.step.*optimizer.step.*")
# Suppress DDP gradient stride mismatch warning for 1x1 convolutions
# This is a known PyTorch issue with 1x1 Conv2d layers - harmless, minimal performance impact
warnings.filterwarnings("ignore", message=".*Grad strides do not match bucket view strides.*")

# Enable Tensor Core optimization for better performance on supported GPUs (H100, A100, etc.)
torch.set_float32_matmul_precision('high')

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from models import GaussTRLightning
from dataset import GaussTRDataModule
from evaluation import OccupancyIoU


class MLflowArtifactCallback(Callback):
    """Callback to log artifacts to MLflow after the run is started.

    Uses MlflowClient with explicit tracking URI to avoid creating mlruns/ folder.
    """

    def __init__(self, config: DictConfig, checkpoint_dir: str):
        self.config = config
        self.checkpoint_dir = checkpoint_dir
        self._logged = False
        self._client = None

    def _get_client(self, trainer):
        """Get MlflowClient using the logger's tracking URI."""
        if self._client is None:
            tracking_uri = os.environ.get('MLFLOW_TRACKING_URI')
            self._client = mlflow.MlflowClient(tracking_uri=tracking_uri)
        return self._client

    def on_train_start(self, trainer, pl_module):
        """Log artifacts once MLflow run is active."""
        if self._logged:
            return
        self._logged = True

        # Get the MLflow run_id from the logger
        if hasattr(trainer.logger, 'run_id') and trainer.logger.run_id:
            run_id = trainer.logger.run_id
            client = self._get_client(trainer)

            # Save and log config
            config_path = os.path.join(self.checkpoint_dir, 'config.yaml')
            with open(config_path, 'w') as f:
                f.write(OmegaConf.to_yaml(self.config))
            client.log_artifact(run_id, config_path)
            print(f"Logged config artifact to MLflow run {run_id}")

    def on_train_end(self, trainer, pl_module):
        """Log final checkpoints and outputs to MLflow."""
        if not hasattr(trainer.logger, 'run_id') or not trainer.logger.run_id:
            return

        run_id = trainer.logger.run_id
        client = self._get_client(trainer)

        # Log best checkpoint
        if trainer.checkpoint_callback and trainer.checkpoint_callback.best_model_path:
            best_ckpt = trainer.checkpoint_callback.best_model_path
            if os.path.exists(best_ckpt):
                client.log_artifact(run_id, best_ckpt, artifact_path="checkpoints")
                print(f"Logged best checkpoint to MLflow: {best_ckpt}")

        # Log last checkpoint
        if trainer.checkpoint_callback and trainer.checkpoint_callback.last_model_path:
            last_ckpt = trainer.checkpoint_callback.last_model_path
            if os.path.exists(last_ckpt) and last_ckpt != trainer.checkpoint_callback.best_model_path:
                client.log_artifact(run_id, last_ckpt, artifact_path="checkpoints")
                print(f"Logged last checkpoint to MLflow: {last_ckpt}")


def build_callbacks(cfg: DictConfig, checkpoint_dir: str, log_artifacts: bool = False) -> list:
    """Build training callbacks.

    Args:
        cfg: Hydra configuration.
        checkpoint_dir: Directory to save checkpoints (includes run_name).
        log_artifacts: Whether to add MLflow artifact logging callback.

    Returns:
        List of Lightning callbacks.
    """
    callbacks = []

    # Model checkpoint - save under run_name subdirectory
    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename='epoch{epoch:02d}-miou{val/miou:.4f}',
        monitor='val/miou',
        mode='max',
        save_top_k=3,
        save_last=True,
        verbose=True,
    )
    callbacks.append(checkpoint_callback)

    # Learning rate monitor
    lr_monitor = LearningRateMonitor(logging_interval='step')
    callbacks.append(lr_monitor)

    # Progress bar
    if cfg.get('use_rich_progress', True):
        callbacks.append(RichProgressBar())

    # Early stopping (optional)
    if cfg.get('early_stopping', False):
        early_stop = EarlyStopping(
            monitor='val/miou',
            patience=cfg.get('early_stopping_patience', 5),
            mode='max',
            verbose=True,
        )
        callbacks.append(early_stop)

    # MLflow artifact logging
    if log_artifacts:
        callbacks.append(MLflowArtifactCallback(cfg, checkpoint_dir))

    # Visualization callback (optional) - runs after test
    vis_cfg = cfg.get('visualization', {})
    if vis_cfg.get('enabled', False):
        from visualization import VisualizationCallback
        vis_callback = VisualizationCallback(
            num_samples=vis_cfg.get('num_samples', 50),
            mode=vis_cfg.get('mode', 'composite'),
            output_format=vis_cfg.get('output_format', 'image'),
            fps=vis_cfg.get('fps', 10),
            save_predictions=vis_cfg.get('save_predictions', False),
            output_dir=os.path.join(checkpoint_dir, 'visualizations'),
        )
        callbacks.append(vis_callback)

    return callbacks


def build_logger(cfg: DictConfig, run_name: str, output_dir: str):
    """Build experiment logger.

    Args:
        cfg: Hydra configuration.
        run_name: Run name for this training run.
        output_dir: Output directory for this run.

    Returns:
        Lightning logger instance.
    """
    logger_type = cfg.get('logger', 'mlflow')

    if logger_type == 'wandb':
        return WandbLogger(
            project=cfg.get('wandb_project', 'gausstr'),
            name=cfg.get('experiment_name', 'gausstr_lightning'),
            save_dir=output_dir,
        )
    elif logger_type == 'mlflow':
        # Use SQLite database for tracking
        tracking_uri = cfg.get('mlflow_tracking_uri', 'sqlite:///mlflow.db')
        artifact_location = cfg.get('mlflow_artifact_location', 'work_dirs/mlflow_artifacts')

        # Ensure artifact directory exists
        os.makedirs(artifact_location, exist_ok=True)

        # Convert to file:// URI if it's a local path
        if not artifact_location.startswith(('file://', 's3://', 'gs://', 'hdfs://')):
            artifact_location = f"file://{os.path.abspath(artifact_location)}"

        return MLFlowLogger(
            experiment_name=cfg.get('experiment_name', 'gausstr_lightning'),
            tracking_uri=tracking_uri,
            run_name=run_name,
            tags=cfg.get('mlflow_tags', None),
            artifact_location=artifact_location,
            save_dir=None,  # Prevent creating ./mlruns folder
        )
    else:  # tensorboard
        return TensorBoardLogger(
            save_dir=output_dir,
            name='logs',
        )


@hydra.main(
    version_base=None,
    config_path="../config",
    config_name="gausstr_featup"
)
def main(cfg: DictConfig) -> None:
    """Main training function.

    Args:
        cfg: Hydra configuration.
    """
    # Check if main process using Lightning's rank detection
    global_rank = int(os.environ.get('RANK', os.environ.get('LOCAL_RANK', 0)))
    is_main = global_rank == 0

    # Print config only on main process
    if is_main:
        print(OmegaConf.to_yaml(cfg))

    # Set random seed for reproducibility
    seed = cfg.get('seed', 42)
    pl.seed_everything(seed, workers=True)

    # Use Hydra's output directory as the run directory
    # This consolidates all outputs (checkpoints, logs, configs) into one place:
    # work_dirs/<experiment_name>/<timestamp>/
    hydra_output_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    checkpoint_dir = hydra_output_dir

    # Extract run_name from the directory structure (timestamp part)
    run_name = cfg.get('run_name')
    if not run_name:
        run_name = os.path.basename(hydra_output_dir)
        # Update cfg so MLflow logger uses the same run_name
        if OmegaConf.is_readonly(cfg):
            OmegaConf.set_readonly(cfg, False)
        cfg.run_name = run_name

    if is_main:
        os.makedirs(checkpoint_dir, exist_ok=True)
        print(f"Run name: {run_name}")
        print(f"All outputs (checkpoints, logs, configs) will be saved to: {checkpoint_dir}")

    # Build model
    model_cfg = OmegaConf.to_container(cfg.model, resolve=True)
    model = GaussTRLightning(**model_cfg)

    # Load pretrained checkpoint if specified
    if cfg.get('resume_from'):
        if is_main:
            print(f"Resuming from checkpoint: {cfg.resume_from}")
        # Lightning will handle this via Trainer
    elif cfg.get('load_from'):
        if is_main:
            print(f"Loading weights from: {cfg.load_from}")
        checkpoint = torch.load(cfg.load_from, map_location='cpu')
        state_dict = checkpoint.get('state_dict', checkpoint)
        # Remove 'model.' prefix if present (from MMEngine checkpoints)
        state_dict = {
            k.replace('model.', ''): v
            for k, v in state_dict.items()
        }
        model.load_state_dict(state_dict, strict=False)

    # Build datamodule
    data_cfg = OmegaConf.to_container(cfg.data, resolve=True)
    datamodule = GaussTRDataModule(**data_cfg)

    # Determine if using MLflow
    use_mlflow = cfg.get('logger', 'mlflow') == 'mlflow'

    # Build callbacks (checkpoints saved under checkpoint_dir = work_dir/run_name)
    # Add artifact logging callback only for MLflow on main process
    callbacks = build_callbacks(cfg, checkpoint_dir, log_artifacts=(use_mlflow and is_main))

    # Build logger
    logger = build_logger(cfg, run_name, checkpoint_dir)

    # Note: MLflow autolog is intentionally NOT used here.
    # autolog() creates mlruns/ folder even with tracking_uri set, and logs duplicate metrics.
    # Instead, we use MLflowArtifactCallback for artifact logging and Lightning's MLFlowLogger for metrics.

    # Build trainer
    trainer_cfg = OmegaConf.to_container(cfg.get('trainer', {}), resolve=True)

    trainer = pl.Trainer(
        max_epochs=trainer_cfg.get('max_epochs', 24),
        accelerator=trainer_cfg.get('accelerator', 'gpu'),
        devices=trainer_cfg.get('devices', 'auto'),
        strategy=trainer_cfg.get('strategy', 'ddp'),
        precision=trainer_cfg.get('precision', '16-mixed'),
        callbacks=callbacks,
        logger=logger,
        gradient_clip_val=trainer_cfg.get('gradient_clip_val', 35.0),
        gradient_clip_algorithm=trainer_cfg.get('gradient_clip_algorithm', 'norm'),
        accumulate_grad_batches=trainer_cfg.get('accumulate_grad_batches', 1),
        val_check_interval=trainer_cfg.get('val_check_interval', 1.0),
        check_val_every_n_epoch=trainer_cfg.get('check_val_every_n_epoch', 1),
        log_every_n_steps=trainer_cfg.get('log_every_n_steps', 50),
        enable_progress_bar=True,
        enable_model_summary=True,
        deterministic=trainer_cfg.get('deterministic', False),
        benchmark=trainer_cfg.get('benchmark', True),
        sync_batchnorm=trainer_cfg.get('sync_batchnorm', False),
        num_sanity_val_steps=trainer_cfg.get('num_sanity_val_steps', 2),
    )

    # Train
    ckpt_path = cfg.get('resume_from', None)
    trainer.fit(model, datamodule, ckpt_path=ckpt_path)

    # Test after training
    if cfg.get('test_after_training', True):
        trainer.test(model, datamodule)

    print(f"Training complete. Checkpoints saved to: {checkpoint_dir}")


if __name__ == '__main__':
    main()
