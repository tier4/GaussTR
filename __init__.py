"""GaussTR Lightning - PyTorch Lightning implementation.

A pure PyTorch Lightning implementation of GaussTR for 3D occupancy prediction,
with minimal external dependencies (no MMEngine/MMDetection3D required for training).

Usage:
    from gausstr_lightning import GaussTRLightning, GaussTRDataModule

    # Create model
    model = GaussTRLightning(
        num_queries=300,
        embed_dims=256,
        feat_dims=512,
    )

    # Create data module
    datamodule = GaussTRDataModule(
        data_root='data/nuscenes',
        batch_size=2,
    )

    # Train with Lightning
    trainer = pl.Trainer(max_epochs=24, accelerator='gpu')
    trainer.fit(model, datamodule)
"""

from models.gausstr import GaussTRLightning
from dataset.datamodule import GaussTRDataModule, GaussTRDataModuleFromConfig
from evaluation.occ_metric import OccupancyIoU

__all__ = [
    "GaussTRLightning",
    "GaussTRDataModule",
    "GaussTRDataModuleFromConfig",
    "OccupancyIoU",
]

__version__ = "0.1.0"
