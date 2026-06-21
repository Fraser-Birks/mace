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


# ---------------------------------------------------------------------------
# Weight slicing helpers
# ---------------------------------------------------------------------------


def _prune_irreps(irreps, new_mul: int) -> o3.Irreps:
    """Reduce all multiplicities in irreps to new_mul."""
    irreps = o3.Irreps(irreps)  # ensure it's an Irreps object (may be a str)
    return o3.Irreps([(new_mul, ir) for _, ir in irreps])


def _slice_linear(
    linear,
    kept_idx: torch.Tensor,
    slice_in: bool = False,
    slice_out: bool = False,
) -> torch.Tensor:
    """Return sliced weight tensor for an o3.Linear (MACE Linear factory returns o3.Linear).

    Iterates instructions in order; each block has shape [mul_in, mul_out] (flat in weight).
    slice_in: keep only kept_idx rows (input channels).
    slice_out: keep only kept_idx cols (output channels).

    Path-weight correction: e3nn's o3.Linear uses path_normalization="element" where
    path_weight = 1/sqrt(mul_in).  When mul_in changes from M to K (slice_in=True), the
    effective weight scales by old_pw/new_pw = sqrt(K/M).  We compensate by scaling the
    stored weights so the effective output is unchanged.  Slicing mul_out alone (slice_out
    only, no slice_in) does NOT change path_weight, so no correction is applied.
    """
    # MACE's Linear factory returns o3.Linear directly (not a wrapper class)
    inner = getattr(linear, "linear", linear)
    w = inner.weight.detach()
    K = len(kept_idx)
    blocks = []
    offset = 0
    for ins in inner.instructions:
        mul_in, mul_out = ins.path_shape
        block = w[offset : offset + mul_in * mul_out].reshape(mul_in, mul_out)
        if slice_in:
            # Rescale to compensate for path_weight change (1/sqrt(mul_in)):
            # stored_new = stored_old * sqrt(K / mul_in)
            scale = (K / mul_in) ** 0.5
            block = block[kept_idx, :] * scale
        if slice_out:
            block = block[:, kept_idx]
        blocks.append(block.reshape(-1))
        offset += mul_in * mul_out
    return torch.cat(blocks)


def _slice_skip_tp(
    skip_tp,
    kept_idx: torch.Tensor,
    slice_in1: bool,
) -> torch.Tensor:
    """Return sliced weight for FullyConnectedTensorProduct (uvw mode).

    Weight per instruction has shape [mul_in1, mul_in2, mul_out].
    slice_in1: slice input-1 (node_feats) channels.
    Always slices the output (hidden_irreps) channels.
    mul_in2 (node_attrs multiplicity) is never pruned.

    Path-weight correction: e3nn FCTP uses path_weight = 1/sqrt(mul_in1 * mul_in2).
    When mul_in1 changes from M to K, we rescale stored weights by sqrt(K/M) to
    keep the effective computation unchanged.  Slicing mul_out alone does NOT change
    path_weight for FCTP, so no correction is needed for the output slice.
    """
    # MACE's FullyConnectedTensorProduct factory returns o3.FullyConnectedTensorProduct
    inner = getattr(skip_tp, "tp", skip_tp)
    w = inner.weight.detach()
    K = len(kept_idx)
    blocks = []
    offset = 0
    for ins in inner.instructions:
        mul_in1, mul_in2, mul_out = ins.path_shape
        n = mul_in1 * mul_in2 * mul_out
        block = w[offset : offset + n].reshape(mul_in1, mul_in2, mul_out)
        if slice_in1:
            # Rescale for path_weight change 1/sqrt(mul_in1 * mul_in2):
            # stored_new = stored_old * sqrt(K / mul_in1)
            scale = (K / mul_in1) ** 0.5
            block = block[kept_idx, :, :] * scale
        block = block[:, :, kept_idx]  # always slice output (no path_weight change)
        blocks.append(block.reshape(-1))
        offset += n
    return torch.cat(blocks)


def _rebuild_conv_tp_weights(old_conv_tp_weights, old_conv_tp, kept_idx: torch.Tensor):
    """Rebuild the radial MLP (conv_tp_weights) for the pruned interaction.

    The radial MLP's hidden layers are copied exactly.  Only the output layer
    is sliced: for each conv_tp instruction (UVU mode) the weight_numel
    allocates one weight per channel, so we keep columns kept_idx per instruction.

    Args:
        old_conv_tp_weights: e3nn FullyConnectedNet from the original interaction.
        old_conv_tp: the original conv_tp (o3.TensorProduct), used to read instructions.
        kept_idx: indices of kept channels.

    Returns:
        The new_conv_tp_weights is taken from the NEW interaction block passed in
        and updated in-place (so this helper returns the column-selection indices
        for the caller to apply).
    """
    # Build column indices for the sliced output layer
    n_instr = len(old_conv_tp.instructions)
    mul = old_conv_tp.instructions[0].path_shape[0]  # e.g., 8
    col_idx = torch.cat([
        kept_idx + k * mul for k in range(n_instr)
    ])
    return col_idx


def _copy_conv_tp_weights(new_conv_tp_weights, old_conv_tp_weights, col_idx: torch.Tensor):
    """Copy conv_tp_weights from old to new, slicing the output layer columns by col_idx."""
    # Copy all layers except the last exactly, then slice the last layer's weight
    new_layers = list(new_conv_tp_weights.named_children())
    old_layers = list(old_conv_tp_weights.named_children())
    with torch.no_grad():
        for (new_name, new_layer), (old_name, old_layer) in zip(new_layers, old_layers):
            if hasattr(new_layer, 'weight') and hasattr(old_layer, 'weight'):
                if new_layer.weight.shape == old_layer.weight.shape:
                    # Same shape: copy exactly (hidden layers)
                    new_layer.weight.copy_(old_layer.weight)
                else:
                    # Different shape: this must be the output layer; slice columns
                    new_layer.weight.copy_(old_layer.weight[:, col_idx])


def _copy_linear_weight(dst_linear, weight: torch.Tensor) -> None:
    """Assign weight to dst_linear (handles MACE-wrapped or raw o3.Linear)."""
    inner = getattr(dst_linear, "linear", dst_linear)
    with torch.no_grad():
        inner.weight.copy_(weight)


def _copy_skip_tp_weight(dst_tp, weight: torch.Tensor) -> None:
    """Assign weight to a FullyConnectedTensorProduct."""
    inner = getattr(dst_tp, "tp", dst_tp)
    with torch.no_grad():
        inner.weight.copy_(weight)


# ---------------------------------------------------------------------------
# Block rebuilders
# ---------------------------------------------------------------------------


def _rebuild_interaction(old_inter, kept_idx: torch.Tensor, first_layer: bool):
    """Construct a new interaction block with pruned irreps and copy sliced weights.

    For first_layer=True: node_feats_irreps and linear_up are kept at FULL width so
    that the skip_tp (sc) exactly reproduces the gated model's sc at kept channels.
    Only the OUTPUT (target_irreps, hidden_irreps) is pruned.

    For non-first layers: everything is pruned to K channels.
    """
    K = len(kept_idx)
    new_hidden = _prune_irreps(old_inter.hidden_irreps, K)

    if first_layer:
        # Keep node_feats_irreps at full width — embedding is not pruned.
        # The skip_tp and linear_up use the full embedding; only outputs are pruned.
        new_node_feats = old_inter.node_feats_irreps
    else:
        new_node_feats = _prune_irreps(old_inter.node_feats_irreps, K)

    new_inter = type(old_inter)(
        node_attrs_irreps=old_inter.node_attrs_irreps,
        node_feats_irreps=new_node_feats,
        edge_attrs_irreps=old_inter.edge_attrs_irreps,
        edge_feats_irreps=old_inter.edge_feats_irreps,
        target_irreps=_prune_irreps(old_inter.target_irreps, K),
        hidden_irreps=new_hidden,
        avg_num_neighbors=old_inter.avg_num_neighbors,
        radial_MLP=old_inter.radial_MLP,
    )

    if first_layer:
        # linear_up: full_node_feats → full_edge_irreps (same, no slicing needed)
        # Copy exactly — the rebuilt interaction still sees full embedding.
        w = _slice_linear(old_inter.linear_up, kept_idx, slice_in=False, slice_out=False)
        _copy_linear_weight(new_inter.linear_up, w)

        # linear: 8-channel irreps_mid → K-target (slice output only)
        w = _slice_linear(old_inter.linear, kept_idx, slice_in=False, slice_out=True)
        _copy_linear_weight(new_inter.linear, w)

        # skip_tp: full_node_feats × node_attrs → K-hidden (slice output only)
        w = _slice_skip_tp(old_inter.skip_tp, kept_idx, slice_in1=False)
        _copy_skip_tp_weight(new_inter.skip_tp, w)
    else:
        # linear_up: K-node_feats → K-edge_irreps (slice both in and out)
        w = _slice_linear(old_inter.linear_up, kept_idx, slice_in=True, slice_out=True)
        _copy_linear_weight(new_inter.linear_up, w)

        # linear: K-irreps_mid → K-target (slice both in and out)
        w = _slice_linear(old_inter.linear, kept_idx, slice_in=True, slice_out=True)
        _copy_linear_weight(new_inter.linear, w)

        # skip_tp: K-node_feats × node_attrs → K-hidden (slice in1 and out)
        w = _slice_skip_tp(old_inter.skip_tp, kept_idx, slice_in1=True)
        _copy_skip_tp_weight(new_inter.skip_tp, w)

    # conv_tp_weights (radial MLP): output size = n_instructions × mul (UVU mode).
    # For first_layer: edge_irreps stays at full width (8 channels), so weight_numel
    # is unchanged → copy exactly.
    # For non-first_layer: edge_irreps is pruned to K channels, so weight_numel decreases
    # → slice the output layer columns by kept_idx (one weight per channel per instruction).
    if first_layer:
        new_inter.conv_tp_weights.load_state_dict(old_inter.conv_tp_weights.state_dict())
    else:
        col_idx = _rebuild_conv_tp_weights(old_inter.conv_tp_weights, old_inter.conv_tp, kept_idx)
        _copy_conv_tp_weights(new_inter.conv_tp_weights, old_inter.conv_tp_weights, col_idx)

    return new_inter


def _rebuild_product(old_prod, kept_idx: torch.Tensor):
    """Construct new EquivariantProductBasisBlock with pruned irreps and copy weights."""
    K = len(kept_idx)
    sc = old_prod.symmetric_contractions
    new_node_feats = _prune_irreps(sc.irreps_in, K)
    new_target = _prune_irreps(sc.irreps_out, K)

    # Infer correlation from the first Contraction
    old_corr = sc.contractions[0].correlation
    num_elements = sc.contractions[0].weights[0].shape[0] if len(sc.contractions[0].weights) > 0 else sc.contractions[0].weights_max.shape[0]

    new_prod = type(old_prod)(
        node_feats_irreps=new_node_feats,
        target_irreps=new_target,
        correlation=old_corr,
        use_sc=old_prod.use_sc,
        num_elements=num_elements,
        use_agnostic_product=getattr(old_prod, "use_agnostic_product", False),
    )

    # Copy symmetric_contractions weights: each weight has shape
    # [num_elements, num_params, num_features]. Slice last dim with kept_idx.
    for new_c, old_c in zip(
        new_prod.symmetric_contractions.contractions,
        old_prod.symmetric_contractions.contractions,
    ):
        # weights_max (highest degree term)
        with torch.no_grad():
            new_c.weights_max.copy_(old_c.weights_max[..., kept_idx])
        # weights (lower degree terms)
        for new_w, old_w in zip(new_c.weights, old_c.weights):
            with torch.no_grad():
                new_w.copy_(old_w[..., kept_idx])

    # product.linear: target_irreps → target_irreps (slice both in and out)
    w = _slice_linear(old_prod.linear, kept_idx, slice_in=True, slice_out=True)
    _copy_linear_weight(new_prod.linear, w)

    return new_prod


def _rebuild_readout(old_readout, kept_idx: torch.Tensor):
    """Rebuild a readout block with pruned input irreps."""
    from mace.modules.blocks import LinearReadoutBlock, NonLinearReadoutBlock
    K = len(kept_idx)
    if isinstance(old_readout, LinearReadoutBlock):
        # Linear readout: slice input irreps, copy output irreps unchanged
        new_in = _prune_irreps(old_readout.linear.irreps_in, K)
        # Determine irreps_out (e.g. "1x0e" or "Nx0e" for multi-head)
        new_ro = LinearReadoutBlock(new_in, old_readout.linear.irreps_out)
        # Slice input channels only (output = scalar, no pruning)
        w = _slice_linear(old_readout.linear, kept_idx, slice_in=True, slice_out=False)
        _copy_linear_weight(new_ro.linear, w)
        return new_ro
    if isinstance(old_readout, NonLinearReadoutBlock):
        # NonLinear readout: slice linear_1 input; linear_2 is unchanged (MLP_irreps → out)
        # Extract gate function from the old readout's non_linearity for correct activation.
        # e3nn wraps the gate with normalize2mom, so we retrieve the underlying function.
        _acts = old_readout.non_linearity.acts
        if len(_acts) > 0 and hasattr(_acts[0], "f"):
            _gate_fn = _acts[0].f  # unwrap normalize2mom -> raw gate function
        elif len(_acts) > 0 and callable(_acts[0]):
            _gate_fn = _acts[0]
        else:
            _gate_fn = None
        new_in = _prune_irreps(old_readout.linear_1.irreps_in, K)
        new_ro = NonLinearReadoutBlock(
            irreps_in=new_in,
            MLP_irreps=old_readout.hidden_irreps,
            gate=_gate_fn,
            irrep_out=old_readout.linear_2.irreps_out,
            num_heads=getattr(old_readout, "num_heads", 1),
        )
        # Slice linear_1 input; copy linear_2 exactly
        w = _slice_linear(old_readout.linear_1, kept_idx, slice_in=True, slice_out=False)
        _copy_linear_weight(new_ro.linear_1, w)
        with torch.no_grad():
            new_ro.linear_2.weight.copy_(old_readout.linear_2.weight)
        return new_ro
    # Fallback: deepcopy and try to slice first linear input
    new_ro = copy.deepcopy(old_readout)
    first_linear = getattr(new_ro, "linear_1", getattr(new_ro, "linear", None))
    old_first = getattr(old_readout, "linear_1", getattr(old_readout, "linear", None))
    if first_linear is not None and old_first is not None:
        w = _slice_linear(old_first, kept_idx, slice_in=True, slice_out=False)
        _copy_linear_weight(first_linear, w)
    return new_ro


def rebuild(student: nn.Module) -> nn.Module:
    """Construct a fresh pruned MACE from a gated student with binary masks.

    Returns a plain model (same class as student, no channel_gates) with
    hidden_irreps reduced to the number of kept channels.

    Design note: for the first interaction layer, node_feats_irreps is kept at
    full width (the embedding is not pruned). This ensures the skip connection (sc)
    from the first interaction exactly reproduces the gated model's sc at kept channels.
    From the second layer onwards, all irreps are pruned to K channels.
    """
    if student.channel_gates is None:
        raise ValueError("Student has no channel_gates. Call make_gated() first.")

    # Extract kept indices per layer from binary gates
    kept_idx_list = []
    for gate in student.channel_gates:
        kept = (gate.g > 0.5).nonzero(as_tuple=False).squeeze(1)
        kept_idx_list.append(kept)

    pruned = copy.deepcopy(student)
    pruned.channel_gates = None

    # Node embedding: kept at full width for first layer equivalence.
    # (The first interaction block uses the full embedding as input.)
    # No rebuild needed — just keep the deepcopy's node_embedding.

    # Rebuild interaction and product blocks
    n = len(student.interactions)
    new_interactions = []
    new_products = []
    for i in range(n):
        kept_idx = kept_idx_list[i]
        new_inter = _rebuild_interaction(
            student.interactions[i], kept_idx, first_layer=(i == 0)
        )
        new_prod = _rebuild_product(student.products[i], kept_idx)
        new_interactions.append(new_inter)
        new_products.append(new_prod)

    pruned.interactions = nn.ModuleList(new_interactions)
    pruned.products = nn.ModuleList(new_products)

    # Rebuild readouts
    new_readouts = []
    for i, readout in enumerate(student.readouts):
        layer_idx = min(i, n - 1)
        new_readouts.append(_rebuild_readout(readout, kept_idx_list[layer_idx]))
    pruned.readouts = nn.ModuleList(new_readouts)

    return pruned


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
