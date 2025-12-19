"""3D voxel visualization using PyTorch GPU rendering.

Provides 3D visualization of occupancy grids using PyTorch for GPU-accelerated
rendering, with Open3D and matplotlib fallbacks.
"""

import numpy as np
import cv2
import torch
from typing import Optional, Tuple, List, Dict, Any
import matplotlib
matplotlib.use('Agg')  # Headless backend
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from .color_maps import OCC_COLORS_RGB, OCC_CLASS_NAMES

# Voxel grid configuration (nuScenes default)
VOXEL_SIZE = [0.4, 0.4, 0.4]
POINT_CLOUD_RANGE = [-40, -40, -1, 40, 40, 5.4]
FREE_LABEL = 17


def render_occupancy_pytorch(
    occ_state: np.ndarray,
    mask_camera: Optional[np.ndarray] = None,
    voxel_size: List[float] = None,
    width: int = 800,
    height: int = 800,
    device: str = 'cuda',
) -> np.ndarray:
    """Render occupancy grid using PyTorch GPU acceleration.

    Args:
        occ_state: 3D occupancy grid (X, Y, Z) with class labels.
        mask_camera: Optional visibility mask.
        voxel_size: Voxel size [dx, dy, dz].
        width: Output image width.
        height: Output image height.
        device: PyTorch device ('cuda' or 'cpu').

    Returns:
        Rendered image as numpy array (H, W, 3) in BGR format.
    """
    if voxel_size is None:
        voxel_size = VOXEL_SIZE

    # Create visibility mask
    if mask_camera is not None:
        occ_show = np.logical_and(occ_state != FREE_LABEL, mask_camera)
    else:
        occ_show = occ_state != FREE_LABEL

    # Get occupied voxel indices
    occ_idx = np.where(occ_show)
    if len(occ_idx[0]) == 0:
        return np.ones((height, width, 3), dtype=np.uint8) * 255

    # Convert to world coordinates
    points = np.stack([
        occ_idx[0] * voxel_size[0] + POINT_CLOUD_RANGE[0],
        occ_idx[1] * voxel_size[1] + POINT_CLOUD_RANGE[1],
        occ_idx[2] * voxel_size[2] + POINT_CLOUD_RANGE[2]
    ], axis=1).astype(np.float32)

    labels = occ_state[occ_idx]

    # Move to GPU
    points_t = torch.from_numpy(points).to(device)
    labels_t = torch.from_numpy(labels).long().to(device)

    # Camera parameters (bird's eye view with angle)
    eye = torch.tensor([-50.0, -50.0, 50.0], device=device)
    center = torch.tensor([0.0, 0.0, 2.0], device=device)
    up = torch.tensor([0.0, 0.0, 1.0], device=device)

    # Compute view matrix
    forward = center - eye
    forward = forward / torch.norm(forward)
    right = torch.linalg.cross(forward, up)
    right = right / torch.norm(right)
    up_new = torch.linalg.cross(right, forward)

    # Transform points to camera space
    points_centered = points_t - eye
    x_cam = torch.sum(points_centered * right, dim=1)
    y_cam = torch.sum(points_centered * up_new, dim=1)
    z_cam = torch.sum(points_centered * forward, dim=1)

    # Filter points behind camera
    valid = z_cam > 0.1
    x_cam = x_cam[valid]
    y_cam = y_cam[valid]
    z_cam = z_cam[valid]
    labels_valid = labels_t[valid]

    if len(x_cam) == 0:
        return np.ones((height, width, 3), dtype=np.uint8) * 255

    # Perspective projection
    fov = 60.0
    f = 1.0 / np.tan(np.radians(fov / 2))
    aspect = width / height

    x_ndc = (f / aspect) * x_cam / z_cam
    y_ndc = f * y_cam / z_cam

    # Convert to pixel coordinates
    x_pix = ((x_ndc + 1) * 0.5 * width).long()
    y_pix = ((1 - (y_ndc + 1) * 0.5) * height).long()

    # Clamp to image bounds
    valid_pix = (x_pix >= 0) & (x_pix < width) & (y_pix >= 0) & (y_pix < height)
    x_pix = x_pix[valid_pix]
    y_pix = y_pix[valid_pix]
    z_cam = z_cam[valid_pix]
    labels_valid = labels_valid[valid_pix]

    if len(x_pix) == 0:
        return np.ones((height, width, 3), dtype=np.uint8) * 255

    # Sort by depth (far to near for painter's algorithm)
    sort_idx = torch.argsort(z_cam, descending=True)
    x_pix = x_pix[sort_idx]
    y_pix = y_pix[sort_idx]
    labels_valid = labels_valid[sort_idx]

    # Create output image
    colors = torch.from_numpy(OCC_COLORS_RGB).to(device)
    img = torch.ones((height, width, 3), dtype=torch.uint8, device=device) * 255

    # Render points (painter's algorithm - far points first)
    labels_clamped = torch.clamp(labels_valid, 0, len(colors) - 1)
    point_colors = colors[labels_clamped]

    # Draw larger points for better visibility
    for dy in range(-1, 2):
        for dx in range(-1, 2):
            y_draw = torch.clamp(y_pix + dy, 0, height - 1)
            x_draw = torch.clamp(x_pix + dx, 0, width - 1)
            img[y_draw, x_draw] = point_colors

    # Convert to numpy BGR
    img_np = img.cpu().numpy()
    img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

    return img_bgr


# Check if Open3D is available
try:
    import open3d as o3d
    # Suppress Open3D info messages
    o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Warning)
    HAS_OPEN3D = True
except ImportError:
    HAS_OPEN3D = False
    o3d = None

# Suppress Filament engine messages by redirecting stderr temporarily during render
import os
import sys
import contextlib

@contextlib.contextmanager
def suppress_stdout_stderr():
    """Suppress stdout and stderr (for Filament engine messages)."""
    with open(os.devnull, 'w') as devnull:
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        old_stdout_fd = os.dup(1)
        old_stderr_fd = os.dup(2)
        try:
            os.dup2(devnull.fileno(), 1)
            os.dup2(devnull.fileno(), 2)
            yield
        finally:
            os.dup2(old_stdout_fd, 1)
            os.dup2(old_stderr_fd, 2)
            os.close(old_stdout_fd)
            os.close(old_stderr_fd)
            sys.stdout = old_stdout
            sys.stderr = old_stderr


def occ_to_points(
    occupancy: np.ndarray,
    free_class: int = 17,
    voxel_size: List[float] = None,
    point_cloud_range: List[float] = None,
    mask: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Convert 3D occupancy grid to point cloud.

    Args:
        occupancy: 3D occupancy grid (X, Y, Z) with class labels.
        free_class: Class ID for 'free' voxels (will be excluded).
        voxel_size: Voxel size [dx, dy, dz].
        point_cloud_range: Point cloud range [x_min, y_min, z_min, x_max, y_max, z_max].
        mask: Optional visibility mask to filter points.

    Returns:
        Tuple of (points, labels):
            - points: (N, 3) array of 3D coordinates.
            - labels: (N,) array of class labels.
    """
    if voxel_size is None:
        voxel_size = VOXEL_SIZE
    if point_cloud_range is None:
        point_cloud_range = POINT_CLOUD_RANGE

    # Find occupied voxels
    if mask is not None:
        occ_mask = (occupancy != free_class) & mask
    else:
        occ_mask = occupancy != free_class

    indices = np.where(occ_mask)
    labels = occupancy[indices]

    # Convert indices to world coordinates
    x = indices[0] * voxel_size[0] + point_cloud_range[0]
    y = indices[1] * voxel_size[1] + point_cloud_range[1]
    z = indices[2] * voxel_size[2] + point_cloud_range[2]

    points = np.stack([x, y, z], axis=1)

    return points, labels


def create_voxel_grid_o3d(occ_state, occ_show, voxel_size, colors_norm):
    """Create Open3D voxel grid from occupancy."""
    import torch

    points, labels, _ = voxel2points_torch(
        torch.from_numpy(occ_state),
        torch.from_numpy(occ_show),
        voxel_size
    )

    if len(points) == 0:
        return None

    points = points.numpy()
    labels = labels.numpy()

    # Get colors for each point
    _labels = labels % len(colors_norm)
    point_colors = colors_norm[_labels.astype(int)][:, :3]

    # Create point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(point_colors)

    # Create voxel grid from point cloud
    voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(
        pcd, voxel_size=voxel_size[0]
    )

    return voxel_grid


def voxel2points_torch(voxel, occ_show, voxel_size):
    """Convert voxel grid to 3D points using torch."""
    import torch

    occ_idx = torch.where(occ_show)
    points = torch.cat([
        occ_idx[0][:, None] * voxel_size[0] + POINT_CLOUD_RANGE[0],
        occ_idx[1][:, None] * voxel_size[1] + POINT_CLOUD_RANGE[1],
        occ_idx[2][:, None] * voxel_size[2] + POINT_CLOUD_RANGE[2]
    ], dim=1)
    return points, voxel[occ_idx], occ_idx


def render_occupancy_open3d(
    occ_state: np.ndarray,
    mask_camera: Optional[np.ndarray] = None,
    voxel_size: List[float] = None,
    width: int = 800,
    height: int = 800,
) -> np.ndarray:
    """Render occupancy grid using Open3D offscreen renderer.

    Args:
        occ_state: 3D occupancy grid (X, Y, Z) with class labels.
        mask_camera: Optional visibility mask.
        voxel_size: Voxel size [dx, dy, dz].
        width: Output image width.
        height: Output image height.

    Returns:
        Rendered image as numpy array (H, W, 3) in BGR format.
    """
    if voxel_size is None:
        voxel_size = VOXEL_SIZE

    # Gamma pre-correction (sRGB -> Linear space)
    colors_norm = (OCC_COLORS_RGB / 255.0) ** 2.2

    # Create visibility mask
    if mask_camera is not None:
        occ_show = np.logical_and(occ_state != FREE_LABEL, mask_camera)
    else:
        occ_show = occ_state != FREE_LABEL

    # Create voxel grid
    voxel_grid = create_voxel_grid_o3d(occ_state, occ_show, voxel_size, colors_norm)

    # Setup offscreen renderer (suppress Filament engine messages)
    with suppress_stdout_stderr():
        render = o3d.visualization.rendering.OffscreenRenderer(width, height)
        render.scene.set_background([1.0, 1.0, 1.0, 1.0])  # White background

        # Add geometry with unlit material
        mat = o3d.visualization.rendering.MaterialRecord()
        mat.shader = "defaultUnlit"
        mat.base_color = [1.0, 1.0, 1.0, 1.0]

        if voxel_grid is not None:
            render.scene.add_geometry("voxels", voxel_grid, mat)

        # Add ego frame (coordinate axes)
        ego_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
            size=5.0, origin=[0, 0, 0])
        render.scene.add_geometry("ego_frame", ego_frame, mat)

        # Setup camera - bird's eye view with slight angle
        center = np.array([0.0, 0.0, 2.0])
        eye = np.array([-50.0, -50.0, 50.0])
        up = np.array([0.0, 0.0, 1.0])

        render.setup_camera(60.0, center, eye, up)

        # Render
        img = render.render_to_image()
        img = np.asarray(img).copy()  # Copy to release renderer reference

        # Explicitly delete renderer to trigger cleanup while suppressed
        del render
        import gc
        gc.collect()

    # Convert RGB to BGR for OpenCV
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    return img


def render_occupancy_3d_matplotlib(
    occupancy: np.ndarray,
    free_class: int = 17,
    mask: Optional[np.ndarray] = None,
    image_size: Tuple[int, int] = (400, 400),
    elev: float = 30,
    azim: float = -60,
    downsample: int = 2,
) -> np.ndarray:
    """Render 3D occupancy using matplotlib (fallback for headless servers).

    Args:
        occupancy: 3D occupancy grid (X, Y, Z) with class labels.
        free_class: Class ID for 'free' voxels.
        mask: Optional visibility mask.
        image_size: Output image size (width, height).
        elev: Elevation angle for view.
        azim: Azimuth angle for view.
        downsample: Downsample factor to reduce points.

    Returns:
        Rendered image as numpy array (H, W, 3) in BGR format.
    """
    # Convert to points
    points, labels = occ_to_points(occupancy, free_class, mask=mask)

    if len(points) == 0:
        # Return white image if no points
        return np.ones((image_size[1], image_size[0], 3), dtype=np.uint8) * 255

    # Downsample for faster rendering
    if downsample > 1 and len(points) > 10000:
        indices = np.random.choice(len(points), len(points) // downsample, replace=False)
        points = points[indices]
        labels = labels[indices]

    # Get colors
    colors = OCC_COLORS_RGB[np.clip(labels, 0, len(OCC_COLORS_RGB) - 1)] / 255.0

    # Create figure
    dpi = 100
    fig = plt.figure(figsize=(image_size[0]/dpi, image_size[1]/dpi), dpi=dpi)
    ax = fig.add_subplot(111, projection='3d')

    # Plot scatter
    ax.scatter(points[:, 0], points[:, 1], points[:, 2],
               c=colors, s=1, alpha=0.8)

    # Set view
    ax.view_init(elev=elev, azim=azim)

    # Set limits
    ax.set_xlim([-40, 40])
    ax.set_ylim([-40, 40])
    ax.set_zlim([-1, 5.4])

    # Labels
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    # Remove background
    ax.set_facecolor('white')
    fig.patch.set_facecolor('white')

    # Render to array
    fig.canvas.draw()
    img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close(fig)

    # Resize if needed
    if img.shape[:2] != (image_size[1], image_size[0]):
        img = cv2.resize(img, image_size, interpolation=cv2.INTER_LINEAR)

    # Convert RGB to BGR
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    return img


def render_occupancy_3d_to_array(
    occupancy: np.ndarray,
    free_class: int = 17,
    mask: Optional[np.ndarray] = None,
    voxel_size: float = 0.4,
    image_size: Tuple[int, int] = (400, 400),
    view_params: Optional[Dict[str, Any]] = None,
    background_color: List[float] = None,
    show_ego: bool = True,
) -> np.ndarray:
    """Render 3D occupancy to numpy array.

    Tries PyTorch GPU first (fastest), then Open3D, falls back to matplotlib.

    Args:
        occupancy: 3D occupancy grid (X, Y, Z) with class labels.
        free_class: Class ID for 'free' voxels.
        mask: Optional visibility mask.
        voxel_size: Display voxel size.
        image_size: Output image size (width, height).
        view_params: Camera view parameters dict.
        background_color: Background RGB color [0-1].
        show_ego: Whether to show ego vehicle box.

    Returns:
        Rendered image as numpy array (H, W, 3) in BGR format.
    """
    # Try PyTorch GPU first (fastest)
    if torch.cuda.is_available():
        try:
            return render_occupancy_pytorch(
                occupancy,
                mask_camera=mask,
                voxel_size=[voxel_size] * 3,
                width=image_size[0],
                height=image_size[1],
                device='cuda',
            )
        except Exception as e:
            pass  # Silent fallback

    # Fallback to matplotlib (Open3D is too slow)
    return render_occupancy_3d_matplotlib(
        occupancy,
        free_class=free_class,
        mask=mask,
        image_size=image_size
    )


def render_occupancy_3d(
    occupancy: np.ndarray,
    output_path: str,
    free_class: int = 17,
    mask: Optional[np.ndarray] = None,
    voxel_size: float = 0.4,
    image_size: Tuple[int, int] = (1280, 720),
    view_params: Optional[Dict[str, Any]] = None,
    background_color: List[float] = None,
    show_ego: bool = True,
) -> np.ndarray:
    """Render 3D occupancy to image file.

    Args:
        occupancy: 3D occupancy grid (X, Y, Z) with class labels.
        output_path: Path to save the rendered image.
        free_class: Class ID for 'free' voxels.
        mask: Optional visibility mask.
        voxel_size: Display voxel size.
        image_size: Output image size (width, height).
        view_params: Camera view parameters dict.
        background_color: Background RGB color [0-1].
        show_ego: Whether to show ego vehicle box.

    Returns:
        Rendered image as numpy array.
    """
    img = render_occupancy_3d_to_array(
        occupancy,
        free_class=free_class,
        mask=mask,
        voxel_size=voxel_size,
        image_size=image_size,
        view_params=view_params,
        background_color=background_color,
        show_ego=show_ego,
    )

    cv2.imwrite(output_path, img)
    return img


def visualize_occupancy_3d(
    occupancy: np.ndarray,
    free_class: int = 17,
    mask: Optional[np.ndarray] = None,
    voxel_size: float = 0.4,
    point_size: float = 3.0,
    window_name: str = "Occupancy 3D",
    background_color: List[float] = None,
) -> None:
    """Interactive 3D visualization of occupancy (requires display).

    Args:
        occupancy: 3D occupancy grid (X, Y, Z) with class labels.
        free_class: Class ID for 'free' voxels.
        mask: Optional visibility mask.
        voxel_size: Display voxel size.
        point_size: Point size for visualization.
        window_name: Window title.
        background_color: Background RGB color [0-1].
    """
    if not HAS_OPEN3D:
        raise ImportError("Open3D required for interactive visualization")

    points, labels = occ_to_points(occupancy, free_class, mask=mask)

    if len(points) == 0:
        print("No points to visualize")
        return

    # Get colors
    colors = OCC_COLORS_RGB[np.clip(labels, 0, len(OCC_COLORS_RGB) - 1)] / 255.0

    # Create point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)

    # Create visualizer
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name=window_name)

    if background_color is not None:
        vis.get_render_option().background_color = np.array(background_color)
    else:
        vis.get_render_option().background_color = np.array([1, 1, 1])

    vis.get_render_option().point_size = point_size

    # Add geometry
    vis.add_geometry(pcd)

    # Add coordinate frame
    coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=5.0)
    vis.add_geometry(coord_frame)

    # Run
    vis.run()
    vis.destroy_window()
