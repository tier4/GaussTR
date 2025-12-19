"""Color definitions for occupancy visualization.

Based on nuScenes/Occ3D color conventions for 18-class occupancy.
"""

import numpy as np
from typing import Tuple, Optional

# 18 nuScenes occupancy classes
OCC_CLASS_NAMES = (
    'others',              # 0
    'barrier',             # 1
    'bicycle',             # 2
    'bus',                 # 3
    'car',                 # 4
    'construction_vehicle', # 5
    'motorcycle',          # 6
    'pedestrian',          # 7
    'traffic_cone',        # 8
    'trailer',             # 9
    'truck',               # 10
    'driveable_surface',   # 11
    'other_flat',          # 12
    'sidewalk',            # 13
    'terrain',             # 14
    'manmade',             # 15
    'vegetation',          # 16
    'free',                # 17 (ignored in visualization)
)

# Colors in BGR format (for OpenCV compatibility)
# Following FlashOCC/Occ3D conventions
OCC_COLORS = np.array([
    [0, 0, 0],           # 0 others - black
    [50, 120, 255],      # 1 barrier - orange
    [203, 192, 255],     # 2 bicycle - pink
    [0, 255, 255],       # 3 bus - yellow
    [245, 150, 0],       # 4 car - blue
    [255, 255, 0],       # 5 construction_vehicle - cyan
    [0, 180, 200],       # 6 motorcycle - dark orange
    [0, 0, 255],         # 7 pedestrian - red
    [150, 240, 255],     # 8 traffic_cone - light yellow
    [0, 60, 135],        # 9 trailer - brown
    [240, 32, 160],      # 10 truck - purple
    [191, 207, 0],       # 11 driveable_surface - turquoise
    [75, 0, 175],        # 12 other_flat - dark red
    [75, 0, 75],         # 13 sidewalk - dark purple
    [80, 240, 150],      # 14 terrain - light green
    [250, 230, 230],     # 15 manmade - light gray/white
    [0, 175, 0],         # 16 vegetation - green
    [255, 255, 255],     # 17 free - white (usually not rendered)
], dtype=np.uint8)

# RGB colors (for matplotlib/PIL compatibility)
OCC_COLORS_RGB = OCC_COLORS[:, ::-1].copy()

# Alternative more vibrant color scheme
OCC_COLORS_VIBRANT = np.array([
    [0, 0, 0],           # 0 others
    [112, 128, 144],     # 1 barrier - slate gray
    [220, 20, 60],       # 2 bicycle - crimson
    [255, 127, 80],      # 3 bus - coral
    [255, 158, 0],       # 4 car - orange
    [233, 150, 70],      # 5 construction_vehicle
    [255, 61, 99],       # 6 motorcycle
    [0, 0, 230],         # 7 pedestrian - blue
    [47, 79, 79],        # 8 traffic_cone - dark slate gray
    [255, 140, 0],       # 9 trailer - dark orange
    [255, 99, 71],       # 10 truck - tomato
    [0, 207, 191],       # 11 driveable_surface - turquoise
    [175, 0, 75],        # 12 other_flat
    [75, 0, 75],         # 13 sidewalk
    [112, 180, 60],      # 14 terrain
    [222, 184, 135],     # 15 manmade - burlywood
    [0, 175, 0],         # 16 vegetation
    [255, 255, 255],     # 17 free
], dtype=np.uint8)


def get_occ_colormap(
    colormap: str = "default",
    alpha: bool = False
) -> np.ndarray:
    """Get occupancy colormap.

    Args:
        colormap: Colormap name ('default', 'vibrant', or 'rgb').
        alpha: Whether to include alpha channel.

    Returns:
        Color array of shape (18, 3) or (18, 4) if alpha=True.
    """
    if colormap == "default":
        colors = OCC_COLORS.copy()
    elif colormap == "vibrant":
        colors = OCC_COLORS_VIBRANT.copy()
    elif colormap == "rgb":
        colors = OCC_COLORS_RGB.copy()
    else:
        raise ValueError(f"Unknown colormap: {colormap}")

    if alpha:
        # Add alpha channel (255 for all except free class)
        alpha_channel = np.full((len(colors), 1), 255, dtype=np.uint8)
        alpha_channel[-1] = 0  # Make 'free' class transparent
        colors = np.concatenate([colors, alpha_channel], axis=1)

    return colors


def class_to_color(
    class_id: int,
    colormap: str = "default"
) -> Tuple[int, int, int]:
    """Get color for a specific class.

    Args:
        class_id: Class index (0-17).
        colormap: Colormap name.

    Returns:
        BGR color tuple.
    """
    colors = get_occ_colormap(colormap)
    return tuple(colors[class_id].tolist())


def get_class_name(class_id: int) -> str:
    """Get class name for a given class ID.

    Args:
        class_id: Class index (0-17).

    Returns:
        Class name string.
    """
    if 0 <= class_id < len(OCC_CLASS_NAMES):
        return OCC_CLASS_NAMES[class_id]
    return f"unknown_{class_id}"
