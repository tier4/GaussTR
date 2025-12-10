"""Occupancy IoU metric implemented with torchmetrics.

Pure PyTorch implementation without MMEngine dependencies.
"""

from typing import Dict, List, Optional, Tuple

import torch
from torchmetrics import Metric


# OCC3D class names
OCC_CLASSES = (
    'others', 'barrier', 'bicycle', 'bus', 'car', 'construction_vehicle',
    'motorcycle', 'pedestrian', 'traffic_cone', 'trailer', 'truck',
    'driveable_surface', 'other_flat', 'sidewalk', 'terrain', 'manmade',
    'vegetation', 'free'
)


def fast_hist(
    pred: torch.Tensor,
    label: torch.Tensor,
    num_classes: int,
) -> torch.Tensor:
    """Compute histogram for fast IoU computation.

    Args:
        pred: Predicted labels [N].
        label: Ground truth labels [N].
        num_classes: Number of classes.

    Returns:
        Confusion matrix histogram [num_classes, num_classes].
    """
    # Filter valid labels
    mask = (label >= 0) & (label < num_classes)
    pred = pred[mask]
    label = label[mask]

    # Compute histogram using bincount
    hist = torch.bincount(
        num_classes * label + pred,
        minlength=num_classes ** 2
    ).reshape(num_classes, num_classes)

    return hist.float()


def per_class_iou(hist: torch.Tensor) -> torch.Tensor:
    """Compute per-class IoU from histogram.

    Matches mmdet3d's per_class_iou implementation exactly.

    Args:
        hist: Confusion matrix [num_classes, num_classes].

    Returns:
        Per-class IoU [num_classes]. May contain NaN for classes with no samples.
    """
    # IoU = diag / (row_sum + col_sum - diag)
    # This matches: np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist))
    return torch.diag(hist) / (hist.sum(dim=1) + hist.sum(dim=0) - torch.diag(hist))


def compute_occ_iou(hist: torch.Tensor, free_index: int) -> float:
    """Compute overall occupancy IoU (excluding free class).

    Matches the original implementation exactly.

    Args:
        hist: Confusion matrix.
        free_index: Index of the free/empty class.

    Returns:
        Overall occupancy IoU.
    """
    # Sum TP for all classes except free (diagonal elements excluding free row/col)
    tp = (
        hist[:free_index, :free_index].sum() +
        hist[free_index + 1:, free_index + 1:].sum()
    )
    # Total predictions excluding free-to-free
    total = hist.sum() - hist[free_index, free_index]

    # Match original: no epsilon, allows inf/nan if total is 0
    return (tp / total).item()


class OccupancyIoU(Metric):
    """Occupancy IoU metric using torchmetrics.

    Computes mean IoU and per-class IoU for 3D occupancy prediction.

    Args:
        num_classes: Number of semantic classes (including free/empty).
        use_camera_mask: Whether to use camera visibility mask.
        use_lidar_mask: Whether to use LiDAR visibility mask.
        ignore_index: Class index to ignore (typically free class).
        class_names: Optional list of class names for logging.
    """

    # Metric state variables
    full_state_update: bool = False

    def __init__(
        self,
        num_classes: int = 18,
        use_camera_mask: bool = True,
        use_lidar_mask: bool = False,
        ignore_index: int = 17,  # Free class
        class_names: Optional[Tuple[str, ...]] = None,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.use_camera_mask = use_camera_mask
        self.use_lidar_mask = use_lidar_mask
        self.ignore_index = ignore_index
        self.class_names = class_names or OCC_CLASSES

        # Register state: confusion matrix histogram
        self.add_state(
            "hist",
            default=torch.zeros(num_classes, num_classes),
            dist_reduce_fx="sum"
        )
        self.add_state(
            "total_samples",
            default=torch.tensor(0),
            dist_reduce_fx="sum"
        )

    def update(
        self,
        preds: torch.Tensor,
        targets: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> None:
        """Update histogram with batch predictions.

        Matches the original OccMetric.process() behavior:
        - Apply visibility mask (camera/lidar)
        - Do NOT filter by ignore_index here - all classes go into histogram
        - ignore_index is only excluded when computing mIoU in compute()

        Args:
            preds: Predicted occupancy labels [B, X, Y, Z] or [B*X*Y*Z].
            targets: Ground truth labels [B, X, Y, Z] or [B*X*Y*Z].
            mask: Optional visibility mask [B, X, Y, Z].
        """
        # Flatten predictions and targets
        preds = preds.flatten()
        targets = targets.flatten()

        # Apply mask if provided (matches original use_image_mask/use_lidar_mask)
        if mask is not None:
            mask = mask.flatten().bool()
            preds = preds[mask]
            targets = targets[mask]

        # NOTE: Do NOT filter by ignore_index here!
        # The original includes all classes in histogram, and only excludes
        # ignore_index when computing mIoU. This is important for occ_iou calculation.

        # Update histogram
        hist = fast_hist(preds.long(), targets.long(), self.num_classes)
        self.hist += hist.to(self.hist.device)
        self.total_samples += 1

    def compute(self) -> Dict[str, torch.Tensor]:
        """Compute metrics from accumulated histogram.

        Matches the original OccMetric._compute_metrics_from_hist() behavior.

        Returns:
            Dictionary of metrics:
                - miou: Mean IoU (excluding free class, using nanmean)
                - occ_iou: Overall occupancy IoU
                - iou_per_class: Per-class IoU tensor
                - iou_{class_name}: Individual class IoUs
        """
        # Compute per-class IoU (may contain NaN for classes with no samples)
        iou = per_class_iou(self.hist)

        # Mean IoU excluding the last class (free class), using nanmean
        # Original: miou = np.nanmean(iou[:-1])  # NOTE: ignore free class
        miou = torch.nanmean(iou[:-1])

        # Overall occupancy IoU
        occ_iou = compute_occ_iou(self.hist, self.ignore_index)

        # Build results dict
        results = {
            'miou': miou,
            'occ_iou': torch.tensor(occ_iou, device=miou.device),
            'iou_per_class': iou,
        }

        # Add per-class IoU with names
        for i, class_name in enumerate(self.class_names):
            if i < len(iou):
                results[f'iou_{class_name}'] = iou[i]

        return results

    def reset(self) -> None:
        """Reset metric state."""
        super().reset()

    def get_table_str(self) -> str:
        """Get formatted table string of results.

        Returns:
            ASCII table string with per-class IoU.
        """
        results = self.compute()
        iou = results['iou_per_class']

        lines = [
            "=" * 60,
            "Occupancy IoU Results",
            "=" * 60,
        ]

        # Print all classes except the last one (free class), matching original
        for i, class_name in enumerate(self.class_names[:-1]):
            if i < len(iou):
                val = iou[i].item()
                # Handle NaN values
                val_str = f"{val:.4f}" if not torch.isnan(iou[i]) else "nan"
                lines.append(f"  {class_name:25s}: {val_str}")

        lines.extend([
            "-" * 60,
            f"  {'mIoU':25s}: {results['miou'].item():.4f}",
            f"  {'Occ IoU':25s}: {results['occ_iou'].item():.4f}",
            "=" * 60,
        ])

        return "\n".join(lines)


class OccupancyAccuracy(Metric):
    """Occupancy accuracy metric.

    Simpler metric that computes overall accuracy.
    """

    full_state_update: bool = False

    def __init__(
        self,
        num_classes: int = 18,
        ignore_index: int = 17,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.ignore_index = ignore_index

        self.add_state("correct", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(
        self,
        preds: torch.Tensor,
        targets: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> None:
        """Update counts."""
        preds = preds.flatten()
        targets = targets.flatten()

        if mask is not None:
            mask = mask.flatten().bool()
            preds = preds[mask]
            targets = targets[mask]

        valid = targets != self.ignore_index
        preds = preds[valid]
        targets = targets[valid]

        self.correct += (preds == targets).sum()
        self.total += targets.numel()

    def compute(self) -> torch.Tensor:
        """Compute accuracy."""
        return self.correct.float() / (self.total.float() + 1e-8)
