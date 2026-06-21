"""Channel-pruning distillation for MACE models."""
import copy
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
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


class TeacherWrapper:
    """Wraps a frozen teacher model for label generation."""

    def __init__(self, model: nn.Module) -> None:
        self.model = copy.deepcopy(model)
        self.model.eval()
        self.model.requires_grad_(False)

    def get_labels(
        self, batch: dict
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Return (E_teacher, F_teacher) with no gradient.

        Note: torch.no_grad() cannot be used here because MACE computes forces
        via torch.autograd.grad, which requires a live computation graph.
        Instead, model parameters are already frozen via requires_grad_(False),
        so no parameter gradients accumulate. Outputs are detached before return.
        """
        out = self.model(batch, compute_force=True)
        energy = out["energy"].detach()
        forces = out.get("forces")
        if forces is not None:
            forces = forces.detach()
        return energy, forces


class DistillationLoss(nn.Module):
    """Weighted MSE on energies (and optionally forces) between student and teacher."""

    def __init__(self, w_energy: float = 1.0, w_forces: float = 100.0) -> None:
        super().__init__()
        self.w_energy = w_energy
        self.w_forces = w_forces

    def forward(
        self,
        student_out: dict,
        E_teacher: torch.Tensor,
        F_teacher: Optional[torch.Tensor],
    ) -> torch.Tensor:
        loss = self.w_energy * F.mse_loss(student_out["energy"], E_teacher)
        if self.w_forces > 0 and F_teacher is not None:
            F_student = student_out.get("forces")
            if F_student is not None:
                loss = loss + self.w_forces * F.mse_loss(F_student, F_teacher)
        return loss
