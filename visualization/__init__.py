"""Visualization module for GaussTR occupancy predictions.

Provides tools for visualizing 3D occupancy predictions:
- BEV (Bird's Eye View) 2D visualization
- 3D voxel visualization with Open3D
- Composite views with camera images
"""

from .color_maps import (
    OCC_COLORS,
    OCC_COLORS_RGB,
    OCC_CLASS_NAMES,
    get_occ_colormap,
)
from .bev import (
    occ_to_bev,
    draw_bev_occupancy,
    save_bev_image,
)
from .voxel_3d import (
    occ_to_points,
    visualize_occupancy_3d,
    render_occupancy_3d,
    render_occupancy_3d_to_array,
)
from .composite import (
    create_surround_view,
    create_composite_visualization,
    create_comparison_visualization,
    create_3d_composite_visualization,
    create_full_composite_visualization,
    create_horizontal_legend,
    save_video,
)
from .utils import (
    load_gt_occupancy,
    load_camera_images,
    get_scene_info,
)
from .callbacks import (
    VisualizationCallback,
    ValidationVisualizationCallback,
)

__all__ = [
    # Color maps
    "OCC_COLORS",
    "OCC_COLORS_RGB",
    "OCC_CLASS_NAMES",
    "get_occ_colormap",
    # BEV visualization
    "occ_to_bev",
    "draw_bev_occupancy",
    "save_bev_image",
    # 3D visualization
    "occ_to_points",
    "visualize_occupancy_3d",
    "render_occupancy_3d",
    "render_occupancy_3d_to_array",
    # Composite visualization
    "create_surround_view",
    "create_composite_visualization",
    "create_comparison_visualization",
    "create_3d_composite_visualization",
    "create_full_composite_visualization",
    "create_horizontal_legend",
    "save_video",
    # Utilities
    "load_gt_occupancy",
    "load_camera_images",
    "get_scene_info",
    # Callbacks
    "VisualizationCallback",
    "ValidationVisualizationCallback",
]
