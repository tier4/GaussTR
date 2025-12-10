"""Custom collate functions for GaussTR Lightning.

Handles batching of multi-view images and associated metadata.
"""

from typing import Dict, List, Any, Optional

import torch
import numpy as np


def collate_gausstr(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Custom collate function for GaussTR data.

    Handles batching of:
    - Multi-view images [B, N, C, H, W]
    - Camera matrices [B, N, 4, 4]
    - Feature maps [B, N, C, H, W] or [B, N, H, W]
    - Depth maps [B, N, H, W] or [B, N, 1, H, W]
    - Occupancy ground truth [B, X, Y, Z]

    Args:
        batch: List of sample dictionaries.

    Returns:
        Batched dictionary with tensors.
    """
    if len(batch) == 0:
        return {}

    # Keys that should be stacked into tensors
    tensor_keys = {
        'images', 'img', 'cam2img', 'cam2ego', 'ego2global', 'img_aug_mat',
        'depth', 'feats', 'sem_seg', 'gt_occ', 'mask_camera', 'mask_lidar',
    }

    # Keys that should be kept as lists
    list_keys = {
        'img_path', 'filename', 'sample_idx', 'token', 'scene_token',
    }

    collated = {}

    for key in batch[0].keys():
        values = [sample[key] for sample in batch if key in sample]
        if len(values) == 0:
            continue

        if key in tensor_keys:
            collated[key] = _stack_tensors(values)
        elif key in list_keys:
            collated[key] = values
        else:
            # Try to stack, fall back to list
            try:
                collated[key] = _stack_tensors(values)
            except (ValueError, RuntimeError, TypeError):
                collated[key] = values

    return collated


def _stack_tensors(values: List[Any]) -> torch.Tensor:
    """Stack values into a batched tensor.

    Args:
        values: List of tensors or numpy arrays.

    Returns:
        Stacked tensor.
    """
    # Convert to tensors if needed
    tensors = []
    for v in values:
        if isinstance(v, torch.Tensor):
            tensors.append(v)
        elif isinstance(v, np.ndarray):
            tensors.append(torch.from_numpy(v))
        elif isinstance(v, (list, tuple)):
            # Try to convert list to tensor
            arr = np.array(v)
            tensors.append(torch.from_numpy(arr))
        else:
            raise TypeError(f"Cannot convert {type(v)} to tensor")

    # Stack along batch dimension
    return torch.stack(tensors, dim=0)


def collate_gausstr_inference(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Collate function for inference (without ground truth).

    Args:
        batch: List of sample dictionaries.

    Returns:
        Batched dictionary with tensors needed for inference.
    """
    if len(batch) == 0:
        return {}

    # Keys needed for inference
    inference_keys = {
        'images', 'cam2img', 'cam2ego', 'img_aug_mat', 'depth', 'feats',
    }

    collated = {}

    for key in batch[0].keys():
        if key not in inference_keys:
            continue

        values = [sample[key] for sample in batch if key in sample]
        if len(values) == 0:
            continue

        collated[key] = _stack_tensors(values)

    # Keep metadata as lists
    for key in ['img_path', 'filename', 'sample_idx']:
        if key in batch[0]:
            collated[key] = [sample.get(key) for sample in batch]

    return collated


def collate_with_padding(
    batch: List[Dict[str, Any]],
    pad_keys: Optional[List[str]] = None,
    pad_value: float = 0.0,
) -> Dict[str, Any]:
    """Collate function with padding support for variable-sized inputs.

    Args:
        batch: List of sample dictionaries.
        pad_keys: Keys that may need padding.
        pad_value: Value to use for padding.

    Returns:
        Batched dictionary with padded tensors.
    """
    if len(batch) == 0:
        return {}

    if pad_keys is None:
        pad_keys = []

    collated = {}

    for key in batch[0].keys():
        values = [sample[key] for sample in batch if key in sample]
        if len(values) == 0:
            continue

        if key in pad_keys:
            collated[key] = _pad_and_stack(values, pad_value)
        else:
            try:
                collated[key] = _stack_tensors(values)
            except (ValueError, RuntimeError, TypeError):
                collated[key] = values

    return collated


def _pad_and_stack(
    values: List[Any],
    pad_value: float = 0.0,
) -> torch.Tensor:
    """Pad tensors to same size and stack.

    Args:
        values: List of tensors.
        pad_value: Value for padding.

    Returns:
        Padded and stacked tensor.
    """
    # Convert to tensors
    tensors = []
    for v in values:
        if isinstance(v, torch.Tensor):
            tensors.append(v)
        elif isinstance(v, np.ndarray):
            tensors.append(torch.from_numpy(v))
        else:
            tensors.append(torch.tensor(v))

    # Find max size for each dimension
    max_shape = list(tensors[0].shape)
    for t in tensors[1:]:
        for i, s in enumerate(t.shape):
            max_shape[i] = max(max_shape[i], s)

    # Pad each tensor
    padded = []
    for t in tensors:
        pad_sizes = []
        for i in range(len(max_shape) - 1, -1, -1):
            pad_sizes.extend([0, max_shape[i] - t.shape[i]])
        if any(p > 0 for p in pad_sizes):
            t = torch.nn.functional.pad(t, pad_sizes, value=pad_value)
        padded.append(t)

    return torch.stack(padded, dim=0)
