"""Composite visualization combining camera views and occupancy.

Creates publication-quality visualizations with multiple views.
"""

import cv2
import numpy as np
from typing import Dict, List, Optional, Tuple, Any

from .color_maps import OCC_COLORS, OCC_CLASS_NAMES
from .bev import draw_bev_occupancy, create_legend


# Camera view order for nuScenes surround view
SURROUND_VIEW_ORDER = [
    'CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT',
    'CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT'
]


def create_surround_view(
    images: Dict[str, np.ndarray],
    output_size: Optional[Tuple[int, int]] = None,
    flip_back: bool = True,
) -> np.ndarray:
    """Create surround view from 6 camera images.

    Args:
        images: Dictionary mapping camera names to images.
        output_size: Optional output size (width, height).
        flip_back: Whether to horizontally flip back cameras.

    Returns:
        Surround view image (2 rows x 3 columns of cameras).
    """
    # Get images in order
    ordered_images = []
    for cam_name in SURROUND_VIEW_ORDER:
        if cam_name in images:
            img = images[cam_name]
            # Flip back cameras for natural view
            if flip_back and 'BACK' in cam_name:
                img = np.flip(img, axis=1).copy()
            ordered_images.append(img)
        else:
            # Create placeholder
            h, w = ordered_images[0].shape[:2] if ordered_images else (900, 1600)
            ordered_images.append(np.zeros((h, w, 3), dtype=np.uint8))

    # Stack into 2x3 grid
    row1 = np.concatenate(ordered_images[:3], axis=1)
    row2 = np.concatenate(ordered_images[3:], axis=1)
    surround = np.concatenate([row1, row2], axis=0)

    # Resize if needed
    if output_size is not None:
        surround = cv2.resize(surround, output_size, interpolation=cv2.INTER_LINEAR)

    return surround


def create_composite_visualization(
    images: Dict[str, np.ndarray],
    occupancy: np.ndarray,
    output_size: Tuple[int, int] = (1600, 900),
    bev_size: int = 400,
    free_class: int = 17,
    show_legend: bool = True,
    title: Optional[str] = None,
) -> np.ndarray:
    """Create composite visualization with cameras and BEV.

    Layout:
    +-------------------+-------+
    |  Front cameras    |  BEV  |
    +-------------------+-------+
    |  Back cameras     |Legend |
    +-------------------+-------+

    Args:
        images: Dictionary mapping camera names to images.
        occupancy: 3D occupancy grid.
        output_size: Output image size (width, height).
        bev_size: Size of BEV image.
        free_class: Class ID for 'free' voxels.
        show_legend: Whether to show class legend.
        title: Optional title text.

    Returns:
        Composite visualization image.
    """
    out_w, out_h = output_size
    cam_w = out_w - bev_size
    cam_h = out_h // 2

    # Create output canvas
    canvas = np.ones((out_h, out_w, 3), dtype=np.uint8) * 255

    # Get camera images in order
    front_cams = ['CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT']
    back_cams = ['CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT']

    # Process front cameras
    front_images = []
    for cam in front_cams:
        if cam in images:
            img = images[cam]
        else:
            img = np.zeros((900, 1600, 3), dtype=np.uint8)
        front_images.append(img)

    front_row = np.concatenate(front_images, axis=1)
    front_row = cv2.resize(front_row, (cam_w, cam_h), interpolation=cv2.INTER_LINEAR)
    canvas[:cam_h, :cam_w] = front_row

    # Process back cameras (flipped)
    back_images = []
    for cam in back_cams:
        if cam in images:
            img = np.flip(images[cam], axis=1).copy()
        else:
            img = np.zeros((900, 1600, 3), dtype=np.uint8)
        back_images.append(img)

    back_row = np.concatenate(back_images, axis=1)
    back_row = cv2.resize(back_row, (cam_w, cam_h), interpolation=cv2.INTER_LINEAR)
    canvas[cam_h:, :cam_w] = back_row

    # Draw BEV occupancy
    bev_img = draw_bev_occupancy(
        occupancy,
        output_size=(bev_size, bev_size),
        free_class=free_class,
        draw_ego=True,
    )
    canvas[:bev_size, cam_w:] = bev_img

    # Draw legend
    if show_legend and bev_size < out_h:
        legend_h = out_h - bev_size
        legend = create_legend(width=bev_size, height=legend_h)
        canvas[bev_size:, cam_w:] = legend

    # Add title
    if title:
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(canvas, title, (10, 30), font, 1, (0, 0, 0), 2)

    return canvas


def create_comparison_visualization(
    images: Dict[str, np.ndarray],
    pred_occ: np.ndarray,
    gt_occ: np.ndarray,
    mask: Optional[np.ndarray] = None,
    output_size: Tuple[int, int] = (2000, 1000),
    bev_size: int = 400,
    free_class: int = 17,
    title: Optional[str] = None,
) -> np.ndarray:
    """Create comparison visualization with pred and GT.

    Layout:
    +-------------------+---------+---------+
    |                   |  Pred   |   GT    |
    |  Surround view    |   BEV   |   BEV   |
    |                   +---------+---------+
    |                   |     Legend        |
    +-------------------+-------------------+

    Args:
        images: Dictionary mapping camera names to images.
        pred_occ: Predicted 3D occupancy.
        gt_occ: Ground truth 3D occupancy.
        mask: Optional camera visibility mask.
        output_size: Output image size (width, height).
        bev_size: Size of each BEV image.
        free_class: Class ID for 'free' voxels.
        title: Optional title text.

    Returns:
        Comparison visualization image.
    """
    out_w, out_h = output_size
    cam_w = out_w - bev_size * 2
    bev_w = bev_size * 2

    # Create output canvas
    canvas = np.ones((out_h, out_w, 3), dtype=np.uint8) * 255

    # Create surround view
    surround = create_surround_view(images)
    surround = cv2.resize(surround, (cam_w, out_h), interpolation=cv2.INTER_LINEAR)
    canvas[:, :cam_w] = surround

    # Draw prediction BEV
    pred_bev = draw_bev_occupancy(
        pred_occ,
        output_size=(bev_size, bev_size),
        free_class=free_class,
        draw_ego=True,
    )
    canvas[:bev_size, cam_w:cam_w + bev_size] = pred_bev

    # Draw GT BEV
    gt_bev = draw_bev_occupancy(
        gt_occ,
        output_size=(bev_size, bev_size),
        free_class=free_class,
        draw_ego=True,
    )
    canvas[:bev_size, cam_w + bev_size:] = gt_bev

    # Add labels
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(canvas, "Prediction", (cam_w + 10, 30), font, 0.8, (0, 0, 0), 2)
    cv2.putText(canvas, "Ground Truth", (cam_w + bev_size + 10, 30), font, 0.8, (0, 0, 0), 2)

    # Draw legend
    legend_h = out_h - bev_size
    if legend_h > 0:
        legend = create_legend(width=bev_w, height=legend_h)
        canvas[bev_size:, cam_w:] = legend

    # Add title
    if title:
        cv2.putText(canvas, title, (10, 30), font, 1, (255, 255, 255), 2)

    return canvas


def create_video_frame(
    frame_data: Dict[str, Any],
    output_size: Tuple[int, int] = (1920, 1080),
    show_metrics: bool = True,
) -> np.ndarray:
    """Create a single video frame with visualization.

    Args:
        frame_data: Dictionary with:
            - images: Camera images dict
            - pred_occ: Predicted occupancy
            - gt_occ: Ground truth occupancy (optional)
            - metrics: Metrics dict (optional)
            - sample_idx: Sample index
        output_size: Output frame size.
        show_metrics: Whether to display metrics.

    Returns:
        Video frame image.
    """
    images = frame_data['images']
    pred_occ = frame_data['pred_occ']
    gt_occ = frame_data.get('gt_occ')
    metrics = frame_data.get('metrics', {})
    sample_idx = frame_data.get('sample_idx', 0)

    if gt_occ is not None:
        frame = create_comparison_visualization(
            images, pred_occ, gt_occ,
            output_size=output_size,
            title=f"Sample {sample_idx}"
        )
    else:
        frame = create_composite_visualization(
            images, pred_occ,
            output_size=output_size,
            title=f"Sample {sample_idx}"
        )

    # Add metrics overlay
    if show_metrics and metrics:
        y_offset = 60
        font = cv2.FONT_HERSHEY_SIMPLEX
        for key, value in metrics.items():
            text = f"{key}: {value:.4f}" if isinstance(value, float) else f"{key}: {value}"
            cv2.putText(frame, text, (10, y_offset), font, 0.6, (0, 0, 0), 1)
            y_offset += 25

    return frame


def create_3d_composite_visualization(
    images: Dict[str, np.ndarray],
    pred_occ: np.ndarray,
    gt_occ: Optional[np.ndarray] = None,
    mask: Optional[np.ndarray] = None,
    output_size: Tuple[int, int] = (2000, 1000),
    render_size: int = 400,
    free_class: int = 17,
    title: Optional[str] = None,
) -> np.ndarray:
    """Create composite visualization with cameras and 3D render.

    Layout:
    +-------------------+---------+---------+
    |                   | Pred 3D | GT 3D   |
    |  Surround view    |         |         |
    |                   +---------+---------+
    |                   |     Legend        |
    +-------------------+-------------------+

    Args:
        images: Dictionary mapping camera names to images.
        pred_occ: Predicted 3D occupancy.
        gt_occ: Ground truth 3D occupancy (optional).
        mask: Optional camera visibility mask.
        output_size: Output image size (width, height).
        render_size: Size of each 3D render.
        free_class: Class ID for 'free' voxels.
        title: Optional title text.

    Returns:
        Composite visualization image with 3D renders.
    """
    from .voxel_3d import render_occupancy_3d_to_array

    out_w, out_h = output_size
    cam_w = out_w - render_size * 2
    render_w = render_size * 2

    # Create output canvas
    canvas = np.ones((out_h, out_w, 3), dtype=np.uint8) * 255

    # Create surround view
    surround = create_surround_view(images)
    surround = cv2.resize(surround, (cam_w, out_h), interpolation=cv2.INTER_LINEAR)
    canvas[:, :cam_w] = surround

    # Render prediction 3D
    try:
        pred_3d = render_occupancy_3d_to_array(
            pred_occ,
            image_size=(render_size, render_size),
            mask=mask,
            free_class=free_class,
        )
        canvas[:render_size, cam_w:cam_w + render_size] = pred_3d
    except Exception:
        # Fallback to BEV if 3D render fails
        pred_bev = draw_bev_occupancy(
            pred_occ,
            output_size=(render_size, render_size),
            free_class=free_class,
            draw_ego=True,
        )
        canvas[:render_size, cam_w:cam_w + render_size] = pred_bev

    # Render GT 3D if available
    if gt_occ is not None:
        try:
            gt_3d = render_occupancy_3d_to_array(
                gt_occ,
                image_size=(render_size, render_size),
                mask=mask,
                free_class=free_class,
            )
            canvas[:render_size, cam_w + render_size:] = gt_3d
        except Exception:
            # Fallback to BEV
            gt_bev = draw_bev_occupancy(
                gt_occ,
                output_size=(render_size, render_size),
                free_class=free_class,
                draw_ego=True,
            )
            canvas[:render_size, cam_w + render_size:] = gt_bev
    else:
        # Just show prediction twice or empty
        pass

    # Add labels
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(canvas, "Prediction", (cam_w + 10, 30), font, 0.8, (0, 0, 0), 2)
    if gt_occ is not None:
        cv2.putText(canvas, "Ground Truth", (cam_w + render_size + 10, 30), font, 0.8, (0, 0, 0), 2)

    # Draw legend
    legend_h = out_h - render_size
    if legend_h > 0:
        legend = create_legend(width=render_w, height=legend_h)
        canvas[render_size:, cam_w:] = legend

    # Add title
    if title:
        cv2.putText(canvas, title, (10, 30), font, 1, (255, 255, 255), 2)

    return canvas


def create_full_composite_visualization(
    images: Dict[str, np.ndarray],
    pred_occ: np.ndarray,
    gt_occ: Optional[np.ndarray] = None,
    mask: Optional[np.ndarray] = None,
    output_size: Tuple[int, int] = (1600, 1800),
    render_size: int = 300,
    free_class: int = 17,
    title: Optional[str] = None,
) -> np.ndarray:
    """Create full composite visualization with cameras, 3D, BEV, and legend.

    Layout (vertical rows):
    +--------------------------------------------------+
    | Row 1: CAM_FL      | CAM_F       | CAM_FR        |
    +--------------------------------------------------+
    | Row 2: CAM_BL      | CAM_B       | CAM_BR        |
    +--------------------------------------------------+
    | Row 3: Pred 3D                | GT 3D           |
    +--------------------------------------------------+
    | Row 4: Pred BEV               | GT BEV          |
    +--------------------------------------------------+
    | Row 5: Legend (17 classes)                       |
    +--------------------------------------------------+

    Args:
        images: Dictionary mapping camera names to images.
        pred_occ: Predicted 3D occupancy.
        gt_occ: Ground truth 3D occupancy (optional).
        mask: Optional camera visibility mask.
        output_size: Output image size (width, height).
        render_size: Size of each 3D/BEV render.
        free_class: Class ID for 'free' voxels.
        title: Optional title text.

    Returns:
        Full composite visualization image.
    """
    from .voxel_3d import render_occupancy_3d_to_array

    out_w, out_h = output_size

    # Calculate row heights
    # Rows 1-2: cameras (2 rows for 2x3 grid)
    # Row 3: 3D renders (pred | GT)
    # Row 4: BEV renders (pred | GT)
    # Row 5: Legend
    legend_h = 80
    available_h = out_h - legend_h

    # Camera rows take 40%, render rows take 60%
    cam_total_h = int(available_h * 0.4)
    cam_row_h = cam_total_h // 2
    render_row_h = (available_h - cam_total_h) // 2

    # Create output canvas
    canvas = np.ones((out_h, out_w, 3), dtype=np.uint8) * 255

    # Row 1-2: Camera views (2x3 grid)
    front_cams = ['CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT']
    back_cams = ['CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT']
    cam_cell_w = out_w // 3

    # Row 1: Front cameras
    y_row1 = 0
    for i, cam in enumerate(front_cams):
        if cam in images:
            img = images[cam]
            img = cv2.resize(img, (cam_cell_w, cam_row_h), interpolation=cv2.INTER_LINEAR)
            canvas[y_row1:y_row1 + cam_row_h, i * cam_cell_w:(i + 1) * cam_cell_w] = img

    # Row 2: Back cameras (flipped)
    y_row2 = cam_row_h
    for i, cam in enumerate(back_cams):
        if cam in images:
            img = np.flip(images[cam], axis=1).copy()
            img = cv2.resize(img, (cam_cell_w, cam_row_h), interpolation=cv2.INTER_LINEAR)
            canvas[y_row2:y_row2 + cam_row_h, i * cam_cell_w:(i + 1) * cam_cell_w] = img

    # Row 3: 3D visualization (pred left, GT right)
    y_row3 = cam_total_h
    half_w = out_w // 2

    # Pred 3D
    try:
        pred_3d = render_occupancy_3d_to_array(
            pred_occ,
            image_size=(half_w, render_row_h),
            mask=mask,
            free_class=free_class,
        )
        canvas[y_row3:y_row3 + render_row_h, :half_w] = pred_3d
    except Exception as e:
        print(f"Pred 3D render failed: {e}")

    # GT 3D
    if gt_occ is not None:
        try:
            gt_3d = render_occupancy_3d_to_array(
                gt_occ,
                image_size=(half_w, render_row_h),
                mask=mask,
                free_class=free_class,
            )
            canvas[y_row3:y_row3 + render_row_h, half_w:] = gt_3d
        except Exception as e:
            print(f"GT 3D render failed: {e}")

    # Row 4: BEV visualization (pred left, GT right)
    y_row4 = cam_total_h + render_row_h

    # Pred BEV
    pred_bev = draw_bev_occupancy(
        pred_occ,
        output_size=(half_w, render_row_h),
        free_class=free_class,
        draw_ego=True,
    )
    canvas[y_row4:y_row4 + render_row_h, :half_w] = pred_bev

    # GT BEV
    if gt_occ is not None:
        gt_bev = draw_bev_occupancy(
            gt_occ,
            output_size=(half_w, render_row_h),
            free_class=free_class,
            draw_ego=True,
        )
        canvas[y_row4:y_row4 + render_row_h, half_w:] = gt_bev

    # Add labels
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(canvas, "Pred 3D", (10, y_row3 + 25), font, 0.7, (0, 0, 0), 2)
    cv2.putText(canvas, "GT 3D", (half_w + 10, y_row3 + 25), font, 0.7, (0, 0, 0), 2)
    cv2.putText(canvas, "Pred BEV", (10, y_row4 + 25), font, 0.7, (0, 0, 0), 2)
    cv2.putText(canvas, "GT BEV", (half_w + 10, y_row4 + 25), font, 0.7, (0, 0, 0), 2)

    # Row 5: Legend at bottom
    y_row5 = available_h
    legend = create_horizontal_legend(width=out_w, height=legend_h)
    canvas[y_row5:, :] = legend

    # Add title at top
    if title:
        # Draw title background
        cv2.rectangle(canvas, (0, 0), (out_w, 30), (50, 50, 50), -1)
        cv2.putText(canvas, title, (10, 22), font, 0.7, (255, 255, 255), 2)

    return canvas


def create_horizontal_legend(
    width: int = 1600,
    height: int = 100,
    exclude_free: bool = True,
) -> np.ndarray:
    """Create a horizontal legend showing class colors and names.

    Args:
        width: Legend image width.
        height: Legend image height.
        exclude_free: Whether to exclude 'free' class.

    Returns:
        Legend image (H, W, 3) in BGR format.
    """
    n_classes = len(OCC_CLASS_NAMES) - (1 if exclude_free else 0)

    # Calculate cell dimensions - arrange in 2 rows
    n_per_row = (n_classes + 1) // 2
    cell_w = width // n_per_row
    cell_h = height // 2

    legend = np.full((height, width, 3), 255, dtype=np.uint8)

    font = cv2.FONT_HERSHEY_SIMPLEX
    box_size = min(20, cell_h - 10)

    idx = 0
    for i, (name, color) in enumerate(zip(OCC_CLASS_NAMES, OCC_COLORS)):
        if exclude_free and name == 'free':
            continue

        row = idx // n_per_row
        col = idx % n_per_row

        x_start = col * cell_w + 5
        y_start = row * cell_h + (cell_h - box_size) // 2

        # Draw color box
        cv2.rectangle(
            legend,
            (x_start, y_start),
            (x_start + box_size, y_start + box_size),
            tuple(int(c) for c in color),
            -1
        )
        cv2.rectangle(
            legend,
            (x_start, y_start),
            (x_start + box_size, y_start + box_size),
            (0, 0, 0),
            1
        )

        # Draw class name
        text_x = x_start + box_size + 5
        text_y = y_start + box_size - 3
        cv2.putText(legend, name[:12], (text_x, text_y), font, 0.35, (0, 0, 0), 1)

        idx += 1

    return legend


def save_video(
    frames: List[np.ndarray],
    output_path: str,
    fps: int = 10,
    codec: str = 'mp4v',
) -> None:
    """Save frames as video file.

    Args:
        frames: List of frame images.
        output_path: Output video path.
        fps: Frames per second.
        codec: Video codec.
    """
    if not frames:
        return

    h, w = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*codec)
    writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

    for frame in frames:
        writer.write(frame)

    writer.release()
