###########################################################################################
# Utilities for teacher→student distillation in MACE fine-tuning
# Authors: mace-tandem contributors
# This program is distributed under the MIT License (see MIT.md)
###########################################################################################

"""
distill_utils.py
================
Helpers for the integrated knowledge-distillation feature (``--distill`` flag).

Public API
----------
STUDENT_PRESETS : dict
    Named size presets for the student model.

resolve_student_config(args, teacher_model_config, teacher_scale_shift) -> dict
    Merge size preset + explicit CLI overrides into a kwargs dict ready
    to be passed as ``ScaleShiftMACE(**student_config)``.

rattle_batch(batch, rattle_std, strain_std, n_augment, device) -> Batch
    Generate perturbed copies of each structure in a ``torch_geometric.Batch``
    for on-the-fly augmentation.
"""

import logging
from typing import Any, Dict

import torch
from e3nn import o3

from mace.tools import torch_geometric

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Student size presets
# ---------------------------------------------------------------------------

STUDENT_PRESETS: Dict[str, Dict[str, Any]] = {
    "xs": dict(hidden_irreps="64x0e+64x1o", num_interactions=1, correlation=2),
    "small": dict(hidden_irreps="128x0e+128x1o", num_interactions=1, correlation=2),
    "medium": dict(hidden_irreps="256x0e+256x1o", num_interactions=2, correlation=3),
}

# ---------------------------------------------------------------------------
# resolve_student_config
# ---------------------------------------------------------------------------

# Keys from teacher_model_config that ScaleShiftMACE accepts directly (via
# MACE.__init__).  Keys NOT in this set are specific to the teacher instance
# (e.g. atomic_inter_scale / atomic_inter_shift which come from teacher_scale_shift)
# and must not be forwarded blindly.
_MACE_INIT_KEYS = {
    "r_max",
    "num_bessel",
    "num_polynomial_cutoff",
    "max_ell",
    "interaction_cls",
    "interaction_cls_first",
    "num_interactions",
    "num_elements",
    "hidden_irreps",
    "MLP_irreps",
    "atomic_energies",
    "avg_num_neighbors",
    "atomic_numbers",
    "correlation",
    "gate",
    "pair_repulsion",
    "apply_cutoff",
    "use_reduced_cg",
    "use_so3",
    "use_agnostic_product",
    "use_last_readout_only",
    "use_embedding_readout",
    "distance_transform",
    "edge_irreps",
    "use_edge_irreps_first",
    "radial_MLP",
    "radial_type",
    "heads",
    "cueq_config",
    "embedding_specs",
    "readout_cls",
}


def resolve_student_config(
    args: Any,
    teacher_model_config: Dict[str, Any],
    teacher_scale_shift: Dict[str, Any],
) -> Dict[str, Any]:
    """Return a kwargs dict suitable for ``ScaleShiftMACE(**student_config)``.

    The student inherits all compatible structural hyperparameters from the
    teacher (cutoff, radial basis, element table, atomic energies, …) and then
    has its *size* overridden by a preset and/or explicit CLI flags.

    Parameters
    ----------
    args:
        Parsed CLI namespace.  Relevant attributes:
        ``distill_size`` (str | None), ``distill_hidden_irreps`` (str | None),
        ``distill_num_interactions`` (int | None), ``distill_correlation``
        (int | None), ``distill_r_max`` (float | None).
    teacher_model_config:
        Dict returned by ``extract_config_mace_model(teacher)``.  Must contain
        at least the keys listed in ``_MACE_INIT_KEYS``.
    teacher_scale_shift:
        Dict with keys ``"atomic_inter_scale"`` and ``"atomic_inter_shift"``
        taken from the teacher's ``scale_shift`` block.  These are copied
        verbatim so the student uses the same energy/force scale.

    Returns
    -------
    dict
        Ready to unpack as ``ScaleShiftMACE(**student_config)``.
    """

    # --- Step 1: start from a filtered copy of the teacher config ----------
    student_config: Dict[str, Any] = {
        k: v
        for k, v in teacher_model_config.items()
        if k in _MACE_INIT_KEYS
    }

    # Inherit scale/shift from the dedicated argument (not teacher_model_config,
    # which may already contain per-head lists or numpy arrays).
    student_config["atomic_inter_scale"] = teacher_scale_shift["atomic_inter_scale"]
    student_config["atomic_inter_shift"] = teacher_scale_shift["atomic_inter_shift"]

    # --- Step 2: apply named size preset ------------------------------------
    distill_size = getattr(args, "distill_size", None)
    if distill_size is None and not getattr(args, "distill_hidden_irreps", None):
        # Default to smallest preset when nothing is specified.
        distill_size = "xs"

    if distill_size is not None:
        if distill_size not in STUDENT_PRESETS:
            raise ValueError(
                f"Unknown distill_size '{distill_size}'. "
                f"Valid choices: {list(STUDENT_PRESETS)}"
            )
        preset = STUDENT_PRESETS[distill_size]
        log.info(
            "Applying student size preset '%s': %s", distill_size, preset
        )
        student_config.update(preset)

    # --- Step 3: apply explicit CLI overrides (highest priority) -----------
    if getattr(args, "distill_hidden_irreps", None):
        student_config["hidden_irreps"] = args.distill_hidden_irreps
        log.info("Overriding student hidden_irreps: %s", args.distill_hidden_irreps)

    if getattr(args, "distill_num_interactions", None) is not None:
        student_config["num_interactions"] = args.distill_num_interactions
        log.info(
            "Overriding student num_interactions: %d", args.distill_num_interactions
        )

    if getattr(args, "distill_correlation", None) is not None:
        student_config["correlation"] = args.distill_correlation
        log.info("Overriding student correlation: %d", args.distill_correlation)

    if getattr(args, "distill_r_max", None) is not None:
        student_config["r_max"] = args.distill_r_max
        log.info("Overriding student r_max: %g", args.distill_r_max)

    # --- Step 4: convert hidden_irreps string → o3.Irreps ------------------
    if isinstance(student_config.get("hidden_irreps"), str):
        student_config["hidden_irreps"] = o3.Irreps(student_config["hidden_irreps"])

    log.info(
        "Student config resolved: hidden_irreps=%s, num_interactions=%s, "
        "correlation=%s, r_max=%s",
        student_config.get("hidden_irreps"),
        student_config.get("num_interactions"),
        student_config.get("correlation"),
        student_config.get("r_max"),
    )

    return student_config


# ---------------------------------------------------------------------------
# rattle_batch
# ---------------------------------------------------------------------------


def rattle_batch(
    batch: "torch_geometric.Batch",
    rattle_std: float,
    strain_std: float,
    n_augment: int,
    device: torch.device,
) -> "torch_geometric.Batch":
    """Return one rattled/strained copy of *batch*, operating on batched tensors.

    Avoids ``to_data_list()`` to sidestep the ``AtomicData`` required-argument
    constructor issue in MACE's custom torch_geometric.  Uses ``batch.clone()``
    instead.

    For each of *n_augment* augmented copies:

    * Per-atom Gaussian position displacements ~ N(0, rattle_std).
    * Per-structure symmetric 3×3 strain ~ N(0, strain_std) applied as
      ``pos_new = pos @ (I + ε)`` and ``cell_new = cell @ (I + ε)``.
      ``shifts`` is recomputed from ``unit_shifts`` and the new cell.

    When ``n_augment > 1``, a single randomly chosen copy is returned (so the
    output batch has the same number of graphs as the input).

    Parameters
    ----------
    batch:
        Input MACE ``torch_geometric.Batch`` with ``positions``, ``batch``
        (node→graph map), ``cell`` (``[n_graphs*3, 3]`` — PyG format), and
        ``unit_shifts``.
    rattle_std:
        Standard deviation (Å) for per-atom position noise.
    strain_std:
        Standard deviation for the symmetric strain matrix entries.
    n_augment:
        Number of augmented copies to generate (one is returned).
    device:
        Torch device on which to generate random tensors.

    Returns
    -------
    torch_geometric.Batch
        A cloned batch with perturbed positions (and cells/shifts).
    """
    if n_augment < 1:
        raise ValueError(f"n_augment must be >= 1, got {n_augment}")

    n_graphs = batch.num_graphs
    pos = batch.positions.to(device)      # [n_atoms, 3]
    dtype = pos.dtype
    graph_idx = batch.batch.to(device)    # [n_atoms] node→graph map

    # When n_augment > 1 pick one copy at random; caller can call multiple times.
    chosen = int(torch.randint(n_augment, (1,)).item()) if n_augment > 1 else 0

    result = None
    for copy_idx in range(n_augment):
        # Generate per-structure symmetric strain (always, to keep RNG consistent)
        A = torch.randn(n_graphs, 3, 3, dtype=dtype, device=device) * strain_std
        strain = (A + A.transpose(-1, -2)) * 0.5             # [n_graphs, 3, 3]
        I = torch.eye(3, dtype=dtype, device=device).unsqueeze(0)
        I_plus_strain = I + strain                            # [n_graphs, 3, 3]

        if copy_idx != chosen:
            continue  # discard non-chosen copy; RNG already advanced above

        aug = batch.clone()

        # Per-atom rattle
        rattle = torch.randn_like(pos) * rattle_std

        # Apply per-structure strain to each atom: I_plus_strain[graph_idx] @ pos
        Ipe_per_atom = I_plus_strain[graph_idx]               # [n_atoms, 3, 3]
        new_pos = torch.bmm(pos.unsqueeze(1), Ipe_per_atom).squeeze(1) + rattle
        aug.positions = new_pos.detach()

        # Cell and shifts
        # batch.cell is stored as [n_graphs*3, 3] (PyG concatenates along dim 0)
        has_cell = hasattr(batch, "cell") and batch.cell is not None
        if has_cell:
            cell = batch.cell.to(device).reshape(n_graphs, 3, 3)  # [n_graphs, 3, 3]
            new_cell = torch.bmm(cell, I_plus_strain)              # [n_graphs, 3, 3]
            aug.cell = new_cell.reshape(-1, 3).detach()            # [n_graphs*3, 3]

            has_unit_shifts = (
                hasattr(batch, "unit_shifts") and batch.unit_shifts is not None
            )
            if has_unit_shifts:
                unit_shifts = batch.unit_shifts.to(device)   # [n_edges, 3]
                edge_graph = graph_idx[batch.edge_index[0]]  # [n_edges]
                new_cell_per_edge = new_cell[edge_graph]     # [n_edges, 3, 3]
                new_shifts = torch.bmm(
                    unit_shifts.float().unsqueeze(1),
                    new_cell_per_edge.float(),
                ).squeeze(1).to(dtype)
                aug.shifts = new_shifts.detach()

        result = aug

    return result
