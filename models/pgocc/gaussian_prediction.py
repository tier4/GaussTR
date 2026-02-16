from dataclasses import dataclass
from typing import Optional

import torch


@dataclass
class GaussianPrediction:
    means: Optional[torch.Tensor] = None
    scales: Optional[torch.Tensor] = None
    rotations: Optional[torch.Tensor] = None
    opacities: Optional[torch.Tensor] = None
    colors: Optional[torch.Tensor] = None
    ovs: Optional[torch.Tensor] = None
    semantics: Optional[torch.Tensor] = None
