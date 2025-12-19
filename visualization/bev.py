"""Bird's Eye View (BEV) visualization for occupancy predictions.

Generates 2D top-down projections of 3D occupancy grids.
"""

import cv2
import numpy as np
from typing import Optional, Tuple
from scipy.ndimage import rotate as scipy_rotate

from .color_maps import OCC_COLORS, OCC_CLASS_NAMES


def occ_to_bev(
    occupancy: np.ndarray,
    free_class: int = 17,
    projection: str = "max",
) -> np.ndarray:
    """Convert 3D occupancy to 2D BEV.

    Args:
        occupancy: 3D occupancy grid (X, Y, Z) with class labels.
        free_class: Class ID for 'free' voxels.
        projection: Projection method ('max' or 'mean').

    Returns:
        2D BEV grid (X, Y) with class labels.
    """
    if occupancy.ndim != 3:
        raise ValueError(f"Expected 3D occupancy, got shape {occupancy.shape}")

    X, Y, Z = occupancy.shape

    # Project along Z axis - take non-free class with highest Z (closest to camera)
    bev = np.full((X, Y), free_class, dtype=np.int32)

    # Iterate from top to bottom, keeping the first non-free voxel
    for z in range(Z - 1, -1, -1):
        layer = occupancy[:, :, z]
        mask = (layer != free_class) & (bev == free_class)
        bev[mask] = layer[mask]

    return bev


def colorize_bev(
    bev: np.ndarray,
    colors: Optional[np.ndarray] = None,
    free_class: int = 17,
    bgr: bool = True,
) -> np.ndarray:
    """Colorize BEV grid.

    Args:
        bev: 2D BEV grid with class labels.
        colors: Color array (N_classes, 3). Uses OCC_COLORS by default.
        free_class: Class ID for 'free' voxels.
        bgr: Whether to output BGR (True) or RGB (False).

    Returns:
        Colored BEV image (H, W, 3).
    """
    if colors is None:
        colors = OCC_COLORS if bgr else OCC_COLORS[:, ::-1]

    # Clip class IDs to valid range
    bev_clipped = np.clip(bev, 0, len(colors) - 1)
    colored = colors[bev_clipped]

    return colored


def draw_bev_occupancy(
    occupancy: np.ndarray,
    output_size: Tuple[int, int] = (800, 800),
    free_class: int = 17,
    rotate_deg: float = -90,
    flip_horizontal: bool = False,
    colors: Optional[np.ndarray] = None,
    draw_grid: bool = False,
    draw_ego: bool = True,
    ego_size: int = 10,
) -> np.ndarray:
    """Draw BEV visualization of occupancy.

    Args:
        occupancy: 3D occupancy grid (X, Y, Z) with class labels.
        output_size: Output image size (height, width).
        free_class: Class ID for 'free' voxels.
        rotate_deg: Rotation angle in degrees.
        flip_horizontal: Whether to flip horizontally.
        colors: Custom color array.
        draw_grid: Whether to draw grid lines.
        draw_ego: Whether to draw ego vehicle marker.
        ego_size: Size of ego marker in pixels.

    Returns:
        BEV visualization image (H, W, 3) in BGR format.
    """
    # Convert to BEV
    bev = occ_to_bev(occupancy, free_class=free_class)

    # Colorize
    colored = colorize_bev(bev, colors=colors, free_class=free_class, bgr=True)

    # Rotate
    if rotate_deg != 0:
        colored = scipy_rotate(colored, rotate_deg, reshape=False, order=0)

    # Flip
    if flip_horizontal:
        colored = np.flip(colored, axis=1).copy()

    # Resize to output size
    colored = cv2.resize(colored, output_size, interpolation=cv2.INTER_NEAREST)

    # Draw grid
    if draw_grid:
        h, w = colored.shape[:2]
        grid_color = (128, 128, 128)
        for i in range(0, w, w // 10):
            cv2.line(colored, (i, 0), (i, h), grid_color, 1)
        for i in range(0, h, h // 10):
            cv2.line(colored, (0, i), (w, i), grid_color, 1)

    # Draw ego vehicle marker (at center)
    if draw_ego:
        h, w = colored.shape[:2]
        cx, cy = w // 2, h // 2
        cv2.circle(colored, (cx, cy), ego_size, (0, 0, 0), -1)
        cv2.circle(colored, (cx, cy), ego_size - 2, (255, 255, 255), -1)

    return colored


def draw_bev_comparison(
    pred_occ: np.ndarray,
    gt_occ: np.ndarray,
    mask: Optional[np.ndarray] = None,
    output_size: Tuple[int, int] = (800, 800),
    free_class: int = 17,
    gap: int = 10,
) -> np.ndarray:
    """Draw side-by-side BEV comparison of prediction and ground truth.

    Args:
        pred_occ: Predicted 3D occupancy.
        gt_occ: Ground truth 3D occupancy.
        mask: Optional camera visibility mask.
        output_size: Size for each BEV image.
        free_class: Class ID for 'free' voxels.
        gap: Gap between images in pixels.

    Returns:
        Comparison image with pred (left) and GT (right).
    """
    pred_bev = draw_bev_occupancy(pred_occ, output_size, free_class)
    gt_bev = draw_bev_occupancy(gt_occ, output_size, free_class)

    # Create gap
    h, w = pred_bev.shape[:2]
    gap_img = np.full((h, gap, 3), 255, dtype=np.uint8)

    # Concatenate
    comparison = np.concatenate([pred_bev, gap_img, gt_bev], axis=1)

    # Add labels
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(comparison, "Prediction", (10, 30), font, 1, (0, 0, 0), 2)
    cv2.putText(comparison, "Ground Truth", (w + gap + 10, 30), font, 1, (0, 0, 0), 2)

    return comparison


def save_bev_image(
    occupancy: np.ndarray,
    output_path: str,
    **kwargs,
) -> None:
    """Save BEV visualization to file.

    Args:
        occupancy: 3D occupancy grid.
        output_path: Output file path.
        **kwargs: Additional arguments for draw_bev_occupancy.
    """
    bev_img = draw_bev_occupancy(occupancy, **kwargs)
    cv2.imwrite(output_path, bev_img)


def create_legend(
    width: int = 200,
    height: int = 600,
    exclude_free: bool = True,
) -> np.ndarray:
    """Create a legend image showing class colors and names.

    Args:
        width: Legend image width.
        height: Legend image height.
        exclude_free: Whether to exclude 'free' class.

    Returns:
        Legend image (H, W, 3) in BGR format.
    """
    n_classes = len(OCC_CLASS_NAMES) - (1 if exclude_free else 0)
    cell_height = height // n_classes

    legend = np.full((height, width, 3), 255, dtype=np.uint8)

    for i, (name, color) in enumerate(zip(OCC_CLASS_NAMES, OCC_COLORS)):
        if exclude_free and name == 'free':
            continue

        y_start = i * cell_height
        y_end = y_start + cell_height

        # Draw color box
        cv2.rectangle(
            legend,
            (5, y_start + 5),
            (35, y_end - 5),
            tuple(int(c) for c in color),
            -1
        )
        cv2.rectangle(
            legend,
            (5, y_start + 5),
            (35, y_end - 5),
            (0, 0, 0),
            1
        )

        # Draw class name
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(
            legend,
            name,
            (45, y_start + cell_height // 2 + 5),
            font,
            0.4,
            (0, 0, 0),
            1
        )

    return legend
