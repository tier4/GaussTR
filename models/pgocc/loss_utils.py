import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


_SSIM_INSTANCE = None


def get_ssim():
    global _SSIM_INSTANCE
    if _SSIM_INSTANCE is None:
        _SSIM_INSTANCE = SSIM().cuda()
    return _SSIM_INSTANCE


class SSIM(nn.Module):
    """Structural Similarity Index loss between image pairs."""

    def __init__(self):
        super().__init__()
        self.mu_x_pool = nn.AvgPool2d(3, 1)
        self.mu_y_pool = nn.AvgPool2d(3, 1)
        self.sig_x_pool = nn.AvgPool2d(3, 1)
        self.sig_y_pool = nn.AvgPool2d(3, 1)
        self.sig_xy_pool = nn.AvgPool2d(3, 1)
        self.refl = nn.ReflectionPad2d(1)
        self.C1 = 0.01 ** 2
        self.C2 = 0.03 ** 2

    def forward(self, x, y):
        x = self.refl(x)
        y = self.refl(y)
        mu_x = self.mu_x_pool(x)
        mu_y = self.mu_y_pool(y)
        sigma_x = self.sig_x_pool(x ** 2) - mu_x ** 2
        sigma_y = self.sig_y_pool(y ** 2) - mu_y ** 2
        sigma_xy = self.sig_xy_pool(x * y) - mu_x * mu_y
        SSIM_n = (2 * mu_x * mu_y + self.C1) * (2 * sigma_xy + self.C2)
        SSIM_d = (mu_x ** 2 + mu_y ** 2 + self.C1) * (sigma_x + sigma_y + self.C2)
        return torch.clamp((1 - SSIM_n / SSIM_d) / 2, 0, 1)


def compute_reprojection_loss(pred, target):
    """85% SSIM + 15% L1 reprojection loss."""
    ssim = get_ssim()
    l1_loss = torch.abs(target - pred).mean(1, True)
    ssim_loss = ssim(pred, target).mean(1, True)
    return 0.85 * ssim_loss + 0.15 * l1_loss


class BackprojectDepth(nn.Module):
    """Backproject depth map to 3D camera-space point cloud."""

    def __init__(self, batch_size, height, width):
        super().__init__()
        self.batch_size = batch_size
        self.height = height
        self.width = width

        meshgrid = np.meshgrid(range(self.width), range(self.height), indexing='xy')
        self.id_coords = np.stack(meshgrid, axis=0).astype(np.float32)
        self.id_coords = nn.Parameter(
            torch.from_numpy(self.id_coords), requires_grad=False)

        self.ones = nn.Parameter(
            torch.ones(self.batch_size, 1, self.height * self.width),
            requires_grad=False)

        self.pix_coords = torch.unsqueeze(torch.stack(
            [self.id_coords[0].view(-1), self.id_coords[1].view(-1)], 0), 0)
        self.pix_coords = self.pix_coords.repeat(batch_size, 1, 1)
        self.pix_coords = nn.Parameter(
            torch.cat([self.pix_coords, self.ones], 1), requires_grad=False)

    def forward(self, depth, inv_K):
        """
        Args:
            depth: [B, H, W]
            inv_K: [B, 4, 4]
        Returns:
            cam_points: [B, 4, H*W]
        """
        cam_points = torch.matmul(inv_K[:, :3, :3], self.pix_coords)
        cam_points = depth.view(self.batch_size, 1, -1) * cam_points
        cam_points = torch.cat([cam_points, self.ones], 1)
        return cam_points


class Project3D(nn.Module):
    """Project 3D camera-space points to 2D pixel coordinates."""

    def __init__(self, batch_size, height, width, eps=1e-7):
        super().__init__()
        self.batch_size = batch_size
        self.height = height
        self.width = width
        self.eps = eps

    def forward(self, points, K, T):
        """
        Args:
            points: [B, 4, N]
            K: [B, 4, 4]
            T: [B, 4, 4]
        Returns:
            pix_coords: Normalized [-1, 1] for grid_sample [B, H, W, 2]
            pix_coords_unnorm: Unnormalized [B, H, W, 2]
        """
        P = torch.matmul(K, T)[:, :3, :]
        cam_points = torch.matmul(P, points)

        pix_coords = cam_points[:, :2, :] / (cam_points[:, 2, :].unsqueeze(1) + self.eps)
        pix_coords = pix_coords.view(self.batch_size, 2, self.height, self.width)
        pix_coords = pix_coords.permute(0, 2, 3, 1)

        pix_coords_unnorm = pix_coords.clone()

        pix_coords[..., 0] /= self.width - 1
        pix_coords[..., 1] /= self.height - 1
        pix_coords = (pix_coords - 0.5) * 2

        return pix_coords, pix_coords_unnorm


def calc_time_warping_loss(depths, t0_2_tn, render_gt, backproject_depth, project_3d, k,
                           num_cams=5, valid_row=0, min_translation=0.5, pixel_mask=None):
    """Temporal depth warping loss with auto-masking.

    Args:
        depths: Predicted depth maps [N, H, W]
        t0_2_tn: Ego-to-ego transforms [B, (T-1)*N, 4, 4]
        render_gt: Ground truth images [B, T*N, H, W, 3]
        backproject_depth: BackprojectDepth module
        project_3d: Project3D module
        k: Camera intrinsics [N, 4, 4]
        num_cams: Number of cameras
        valid_row: First render row with backbone feature coverage (0 = no masking)
        min_translation: Skip warp for frames with ego motion below this (meters).
            T4 10Hz data has ~35% stationary samples where warping provides no
            gradient signal. Filtering these prevents identity loss from dominating.
        pixel_mask: Optional [N, 1, H, W] bool mask. True = valid pixel for loss.
            Used to exclude ego car, sky, and dynamic objects from warp loss.

    Returns:
        Scalar warping loss, or zero tensor if all frames are stationary
    """
    reprojection_losses = []
    identity_reprojection_losses = []

    render_gt = render_gt[0].permute(0, 3, 1, 2) / 255.0
    t0_img = render_gt[0:num_cams]
    tn_img = render_gt[num_cams:3 * num_cams]  # past 2 frames
    k = k[0:num_cams].float()
    inv_k = torch.inverse(k).float()

    depths = depths.float()
    cam_points = backproject_depth(depths, inv_k)

    num_past_frames = int(len(tn_img) / num_cams)
    for past_i in range(num_past_frames):
        T_slice = t0_2_tn[0][num_cams * past_i:num_cams * (past_i + 1)].float()

        # Skip frames with insufficient ego motion — warping produces no useful
        # gradient when the camera barely moved (identity always wins).
        avg_translation = T_slice[:, :3, 3].norm(dim=1).mean()
        if avg_translation < min_translation:
            continue

        pix_coords, _ = project_3d(cam_points, k, T_slice)

        warped_img = F.grid_sample(
            tn_img[past_i * num_cams:(past_i + 1) * num_cams],
            pix_coords,
            padding_mode="border",
            align_corners=True)

        reprojection_loss = compute_reprojection_loss(warped_img, t0_img)
        reprojection_losses.append(reprojection_loss)

        identity_reprojection_loss = compute_reprojection_loss(
            tn_img[past_i * num_cams:(past_i + 1) * num_cams], t0_img)
        identity_reprojection_losses.append(identity_reprojection_loss)

    # All frames were stationary — return zero loss (no gradient)
    if len(reprojection_losses) == 0:
        return torch.tensor(0.0, device=depths.device, requires_grad=True)

    reprojection_losses = torch.cat(reprojection_losses, dim=0)
    identity_reprojection_losses = torch.cat(identity_reprojection_losses, dim=0)

    # Small noise to break ties in auto-masking
    identity_reprojection_losses += torch.randn(
        identity_reprojection_losses.shape,
        device=identity_reprojection_losses.device) * 0.00001

    combined = torch.cat((identity_reprojection_losses, reprojection_losses), dim=1)
    to_optimise, _ = torch.min(combined, dim=1)

    # Exclude blind rows (no backbone features) from loss
    if valid_row > 0:
        to_optimise = to_optimise[:, valid_row:, :]

    # Apply per-pixel mask (ego car, sky, dynamic objects)
    if pixel_mask is not None:
        # pixel_mask is [N, 1, H, W] — slice to match to_optimise shape
        pm = pixel_mask[:num_cams].float()
        if valid_row > 0:
            pm = pm[:, :, valid_row:, :]
        # to_optimise may be [P*N, 1, H', W'] from cat across past frames
        if to_optimise.shape[0] > pm.shape[0]:
            num_past = to_optimise.shape[0] // pm.shape[0]
            pm = pm.repeat(num_past, 1, 1, 1)
        to_optimise = to_optimise * pm
        count = pm.sum().clamp(min=1.0)
        return to_optimise.sum() / count

    return to_optimise.mean()
