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


def _infer_num_features(model: nn.Module) -> int:
    """Read num_features (uniform channel multiplicity) from the first interaction block."""
    return model.interactions[0].hidden_irreps.count(o3.Irrep(0, 1))


def make_gated(model: nn.Module) -> nn.Module:
    """Return a deep copy of model with per-layer ChannelGates attached (all=1.0).

    The returned model is the same class as the input (MACE, ScaleShiftMACE, etc.)
    and behaves identically until gates are modified.
    """
    if getattr(model, "cueq_config", None) is not None:
        raise NotImplementedError("make_gated does not support cuEquivariance models.")
    if getattr(model, "oeq_config", None) is not None:
        raise NotImplementedError("make_gated does not support OpenEquivariance models.")
    student = copy.deepcopy(model)
    num_features = _infer_num_features(student)
    student.channel_gates = nn.ModuleList(
        [ChannelGate(num_features) for _ in student.interactions]
    )
    return student
