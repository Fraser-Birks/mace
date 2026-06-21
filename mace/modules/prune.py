"""Channel-pruning distillation for MACE models."""
import copy
from typing import Optional

import torch
import torch.nn as nn
from e3nn import o3


class ChannelGate(nn.Module):
    """Per-channel multiplicative gate applied to reshaped node features [n_nodes, mul, d]."""

    def __init__(self, num_features: int) -> None:
        super().__init__()
        self.g = nn.Parameter(torch.ones(num_features))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [n_nodes, mul, d]
        return x * self.g[None, :, None]

    def mask(self, kept_idx: torch.Tensor) -> None:
        """Set gate to binary: 1 for kept_idx, 0 elsewhere."""
        with torch.no_grad():
            self.g.zero_()
            self.g[kept_idx] = 1.0
