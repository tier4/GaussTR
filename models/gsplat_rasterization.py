"""Gaussian splatting rasterization using gsplat library.

Pure PyTorch implementation without MMEngine dependencies.
"""

from typing import Tuple, Optional

import torch
from gsplat import rasterization

from .utils import unbatched_forward


@unbatched_forward
def rasterize_gaussians(
    means3d: torch.Tensor,
    colors: torch.Tensor,
    opacities: torch.Tensor,
    scales: torch.Tensor,
    rotations: torch.Tensor,
    cam2imgs: torch.Tensor,
    cam2egos: torch.Tensor,
    image_size: Tuple[int, int],
    img_aug_mats: Optional[torch.Tensor] = None,
    **kwargs
) -> torch.Tensor:
    """Rasterize 3D Gaussians to 2D images.

    Args:
        means3d: 3D Gaussian centers [N, 3].
        colors: Gaussian colors/features [N, C].
        opacities: Gaussian opacities [N].
        scales: Gaussian scales [N, 3].
        rotations: Gaussian rotations (quaternions) [N, 4].
        cam2imgs: Camera intrinsic matrices [num_cams, 3, 3] or [num_cams, 4, 4].
        cam2egos: Camera to ego transformations [num_cams, 4, 4].
        image_size: Output image size (height, width).
        img_aug_mats: Image augmentation matrices [num_cams, 4, 4]. Optional.
        **kwargs: Additional arguments passed to gsplat.rasterization.

    Returns:
        Rendered images of shape [num_cams, C+1, H, W] (C features + depth).
    """
    # cam2world to world2cam
    R = cam2egos[:, :3, :3].mT
    T = -R @ cam2egos[:, :3, 3:4]
    viewmat = torch.zeros_like(cam2egos)
    viewmat[:, :3, :3] = R
    viewmat[:, :3, 3:] = T
    viewmat[:, 3, 3] = 1

    if cam2imgs.shape[-2:] == (4, 4):
        cam2imgs = cam2imgs[:, :3, :3]
    if img_aug_mats is not None:
        cam2imgs = cam2imgs.clone()
        cam2imgs[:, :2, :2] *= img_aug_mats[:, :2, :2]
        image_size = list(image_size)
        for i in range(2):
            cam2imgs[:, i, 2] *= img_aug_mats[:, i, i]
            cam2imgs[:, i, 2] += img_aug_mats[:, i, 3]
            image_size[1 - i] = round(image_size[1 - i] *
                                      img_aug_mats[0, i, i].item() +
                                      img_aug_mats[0, i, 3].item())

    rendered_image = rasterization(
        means3d,
        rotations,
        scales,
        opacities,
        colors,
        viewmat,
        cam2imgs,
        width=image_size[1],
        height=image_size[0],
        **kwargs)[0]
    return rendered_image.permute(0, 3, 1, 2)
