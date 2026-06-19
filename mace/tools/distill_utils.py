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
from copy import deepcopy
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
    """Generate perturbed copies of each structure in *batch*.

    For every structure in *batch* this function creates *n_augment* rattled
    copies, each with:

    * Independent per-atom Gaussian position displacements:
      ``Δpos ~ N(0, rattle_std)``.
    * A per-copy symmetric 3×3 strain matrix ``ε ~ N(0, strain_std)``
      applied as ``pos_new = pos @ (I + ε)`` and
      ``cell_new = cell @ (I + ε)`` (if ``cell`` is present).

    Parameters
    ----------
    batch:
        Input ``torch_geometric.Batch``.  Expected to have at least
        ``pos``, ``batch`` (node→graph map), and ``ptr`` attributes.
        ``cell`` (shape ``[n_graphs, 3, 3]``) is updated when present.
    rattle_std:
        Standard deviation (Å) for per-atom position noise.
    strain_std:
        Standard deviation for the symmetric strain matrix entries.
    n_augment:
        Number of augmented copies per input structure.
    device:
        Torch device on which to generate random tensors.

    Returns
    -------
    torch_geometric.Batch
        A new Batch containing ``n_structures * n_augment`` structures with
        perturbed positions (and cells).  All other per-node/per-edge
        attributes are shallow-copied from the originals.
    """

    # Split the batch into individual Data objects.
    data_list = batch.to_data_list()

    augmented: list = []
    for data in data_list:
        pos = data.pos  # [n_atoms, 3]
        n_atoms = pos.shape[0]

        has_cell = hasattr(data, "cell") and data.cell is not None

        for _ in range(n_augment):
            aug = deepcopy(data)

            # -- Symmetric strain -------------------------------------------
            # Sample a full 3×3 matrix and symmetrise: ε = (A + Aᵀ) / 2
            A = torch.randn(3, 3, dtype=pos.dtype, device=device) * strain_std
            strain = (A + A.T) * 0.5  # [3, 3]
            I_plus_strain = torch.eye(3, dtype=pos.dtype, device=device) + strain

            # -- Position perturbation ---------------------------------------
            rattle = torch.randn(n_atoms, 3, dtype=pos.dtype, device=device) * rattle_std

            # Apply: new_pos = pos @ (I + ε) + rattle
            new_pos = pos.to(device) @ I_plus_strain + rattle
            aug.pos = new_pos.detach().requires_grad_(False)

            # -- Cell update (if present) ------------------------------------
            if has_cell:
                cell = data.cell  # [1, 3, 3] or [3, 3]
                orig_shape = cell.shape
                cell_3x3 = cell.to(device).reshape(3, 3)
                new_cell = (cell_3x3 @ I_plus_strain).reshape(orig_shape)
                aug.cell = new_cell.detach().requires_grad_(False)

            augmented.append(aug)

    new_batch = torch_geometric.Batch.from_data_list(augmented)
    return new_batch
