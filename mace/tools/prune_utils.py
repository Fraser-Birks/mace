"""ChannelPruner: orchestrates gradual pruning using prune.py components."""
import dataclasses

import torch
import torch.nn as nn

from mace.modules.prune import (
    DistillationLoss,
    CubicPruningSchedule,
    TaylorImportance,
    TeacherWrapper,
    _infer_num_features,
    make_gated,
    rebuild,
)


@dataclasses.dataclass
class PruneConfig:
    target_channels: int
    start_step: int = 0
    n_steps: int = 100
    dt: int = 10
    w_energy: float = 1.0
    w_forces: float = 100.0
    fine_tune_steps: int = 50
    lr: float = 1e-4


class ChannelPruner:
    """Orchestrates gradual channel pruning + distillation of a MACE model."""

    def __init__(
        self,
        teacher: nn.Module,
        data_loader,
        config: PruneConfig,
    ) -> None:
        self.teacher = TeacherWrapper(teacher)
        self.loader = data_loader
        self.config = config

    def run(self) -> nn.Module:
        """Execute the full pruning pipeline and return the pruned model."""
        cfg = self.config
        student = make_gated(self.teacher.model)
        student.train()

        num_features = _infer_num_features(student)
        n_layers = len(student.interactions)

        opt = torch.optim.Adam(student.parameters(), lr=cfg.lr)
        loss_fn = DistillationLoss(w_energy=cfg.w_energy, w_forces=cfg.w_forces)
        importance = TaylorImportance(n_layers=n_layers, num_features=num_features)
        schedule = CubicPruningSchedule(
            n_total=num_features,
            target_channels=cfg.target_channels,
            start_step=cfg.start_step,
            n_steps=cfg.n_steps,
            dt=cfg.dt,
        )

        prev_kept = {i: num_features for i in range(n_layers)}
        global_step = 0

        for batch in self._infinite_loader():
            E_t, F_t = self.teacher.get_labels(batch)
            opt.zero_grad()
            student_out = student(
                batch, compute_force=(cfg.w_forces > 0)
            )
            loss = loss_fn(student_out, E_t, F_t)
            loss.backward()
            importance.update(student)
            opt.step()

            # Check schedule
            for i in range(n_layers):
                new_kept = schedule.kept_count(global_step)
                if new_kept < prev_kept[i]:
                    kept_idx = importance.rank(i)[-new_kept:]
                    student.channel_gates[i].mask(kept_idx.sort().values)
                    prev_kept[i] = new_kept

            global_step += 1
            if global_step >= cfg.start_step + cfg.n_steps * cfg.dt:
                break

        # Final binary mask (ensure target is exactly met)
        for i in range(n_layers):
            kept_idx = importance.rank(i)[-cfg.target_channels:]
            student.channel_gates[i].mask(kept_idx.sort().values)

        # Structural export
        pruned = rebuild(student)

        # Optional brief fine-tune of exported model
        if cfg.fine_tune_steps > 0:
            pruned = self._fine_tune(pruned, cfg.fine_tune_steps, cfg.lr)

        return pruned

    def _fine_tune(self, model: nn.Module, steps: int, lr: float) -> nn.Module:
        model.train()
        opt = torch.optim.Adam(model.parameters(), lr=lr)
        loss_fn = DistillationLoss()
        for batch in self._infinite_loader():
            E_t, F_t = self.teacher.get_labels(batch)
            opt.zero_grad()
            out = model(batch, compute_force=False)
            loss = loss_fn(out, E_t, F_t)
            loss.backward()
            opt.step()
            steps -= 1
            if steps <= 0:
                break
        model.eval()
        return model

    def _infinite_loader(self):
        """Cycle through the data loader indefinitely."""
        while True:
            for batch in self.loader:
                yield batch.to_dict() if hasattr(batch, "to_dict") else batch
