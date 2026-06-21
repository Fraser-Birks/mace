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


class TaylorImportance:
    """EMA accumulator of Molchanov first-order Taylor importance: (g * g.grad)^2."""

    def __init__(self, n_layers: int, num_features: int, alpha: float = 0.9) -> None:
        self.alpha = alpha
        self.scores = [torch.zeros(num_features) for _ in range(n_layers)]
        self._initialised = [False] * n_layers

    def update(self, student: nn.Module) -> None:
        """Call after loss.backward(). Reads gate.g and gate.g.grad."""
        for i, gate in enumerate(student.channel_gates):
            if gate.g.grad is None:
                continue
            score = (gate.g.detach() * gate.g.grad.detach()) ** 2
            if not self._initialised[i]:
                self.scores[i] = score.clone()
                self._initialised[i] = True
            else:
                self.scores[i] = self.alpha * self.scores[i] + (1 - self.alpha) * score

    def rank(self, layer_idx: int) -> torch.Tensor:
        """Return channel indices sorted ascending (least important first)."""
        return torch.argsort(self.scores[layer_idx])

    def reset(self) -> None:
        self.scores = [torch.zeros_like(s) for s in self.scores]
        self._initialised = [False] * len(self.scores)


class CubicPruningSchedule:
    """Zhu & Gupta cubic sparsity schedule.

    s_t = s_f * (1 - ((t - t0) / (n * dt))^3)
    kept_count_t = round(n_total * (1 - s_t))
    """

    def __init__(
        self,
        n_total: int,
        target_channels: int,
        start_step: int,
        n_steps: int,
        dt: int,
    ) -> None:
        self.n_total = n_total
        self.target = target_channels
        self.t0 = start_step
        self.n_steps = n_steps
        self.dt = dt
        self.s_f = 1.0 - target_channels / n_total

    def kept_count(self, step: int) -> int:
        if step <= self.t0:
            return self.n_total
        end = self.t0 + self.n_steps * self.dt
        if step >= end:
            return self.target
        progress = (step - self.t0) / (self.n_steps * self.dt)
        s_t = self.s_f * (1.0 - (1.0 - progress) ** 3)
        return max(self.target, round(self.n_total * (1.0 - s_t)))
