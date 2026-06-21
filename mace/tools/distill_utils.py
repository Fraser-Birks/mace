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

filter_batch_by_head(batch, target_head_idx) -> Batch | None
    Return a sub-batch containing only graphs whose head == target_head_idx.
    Works at the tensor level (no AtomicData constructor calls).
    Returns None when no graphs match.

rattle_batch(batch, rattle_std, strain_std, n_augment, device) -> Batch
    Generate perturbed copies of each structure in a ``torch_geometric.Batch``
    for on-the-fly augmentation.

save_augmented_xyz(aug_batch, teacher_out, path, z_list) -> int
    Append augmented structures with teacher-predicted labels to an extxyz file.
    Returns the number of structures written.
"""

import logging
from typing import Any, Dict, Optional

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
    target_head_name: str = "Default",
    target_head_teacher_idx: int = 0,
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

    # --- Single-head student: always use the target head only ---------------
    # The student is system-specific and should have exactly one head.
    # If the teacher is multihead, atomic_energies is 2-D [n_elements, n_heads].
    # Extract only the target head's column so the student uses the correct E0s
    # regardless of what head index appears in any given batch.
    student_config["heads"] = [target_head_name]
    # atomic_energies from the teacher config has shape (n_heads, n_elements).
    # For a single-head student we keep shape (1, n_elements) with only the
    # target head's row — so the student uses the correct E0 reference.
    ae = student_config.get("atomic_energies")
    if ae is not None and hasattr(ae, "ndim") and ae.ndim == 2 and ae.shape[0] > 1:
        student_config["atomic_energies"] = ae[[target_head_teacher_idx], :]
        log.info(
            "Multihead teacher detected: extracting E0s for head '%s' (idx=%d) "
            "for single-head student",
            target_head_name,
            target_head_teacher_idx,
        )
    log.info("Student will have a single head: '%s'", target_head_name)

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
# filter_batch_by_head
# ---------------------------------------------------------------------------


def filter_batch_by_head(
    batch: "torch_geometric.Batch",
    target_head_idx: int,
) -> "Optional[torch_geometric.Batch]":
    """Return a sub-batch containing only graphs where head == *target_head_idx*.

    Works entirely at the tensor level, avoiding ``batch.get_example()`` /
    ``Batch.from_data_list()`` which both call ``AtomicData()`` with no
    arguments and crash because ``AtomicData.__init__`` requires positional args.

    Parameters
    ----------
    batch:
        A ``torch_geometric.Batch`` (or similar) produced by the training
        dataloader.  Must have a ``head`` attribute of shape ``[n_graphs]``.
    target_head_idx:
        Integer head index to keep.

    Returns
    -------
    torch_geometric.Batch | None
        Filtered batch, or ``None`` if no graphs in *batch* match
        *target_head_idx*.  Returns *batch* unchanged if all graphs match.
    """
    if not (hasattr(batch, "head") and batch.head is not None):
        return batch

    graph_mask = batch.head == target_head_idx  # [n_graphs] bool
    if graph_mask.all():
        return batch
    if not graph_mask.any():
        return None

    keep = graph_mask.nonzero(as_tuple=True)[0]  # LongTensor [n_keep]
    n_keep = int(keep.numel())
    n_graphs_orig = int(batch.num_graphs)
    dev = keep.device

    # ── Atom-level bookkeeping ──────────────────────────────────────────────
    graph_per_atom = batch.batch  # [n_atoms] atom→graph index
    atom_mask = graph_mask[graph_per_atom]  # [n_atoms] bool

    # Remap: old graph index → new graph index in [0, n_keep)
    remap = torch.full((n_graphs_orig,), -1, dtype=torch.long, device=dev)
    remap[keep] = torch.arange(n_keep, dtype=torch.long, device=dev)
    new_graph_per_atom = remap[graph_per_atom[atom_mask]]  # [n_keep_atoms]

    # Old atom index → new atom index in filtered batch
    kept_atoms = atom_mask.nonzero(as_tuple=False).squeeze(1)  # [n_keep_atoms]
    n_keep_atoms = int(kept_atoms.numel())
    old_to_new_atom = torch.full((int(batch.num_nodes),), -1, dtype=torch.long, device=dev)
    old_to_new_atom[kept_atoms] = torch.arange(n_keep_atoms, dtype=torch.long, device=dev)

    # ── Edge-level bookkeeping ──────────────────────────────────────────────
    # MACE edges are intra-graph, so filtering by source atom is sufficient.
    src_atoms = batch.edge_index[0]
    edge_mask = atom_mask[src_atoms]  # [n_edges] bool
    new_edge_index = old_to_new_atom[batch.edge_index[:, edge_mask]]  # [2, n_keep_edges]

    # ── Build filtered clone ────────────────────────────────────────────────
    aug = batch.clone()
    aug.__num_graphs__ = n_keep

    # Atom-level
    aug.batch = new_graph_per_atom
    aug.positions = batch.positions[atom_mask]
    aug.node_attrs = batch.node_attrs[atom_mask]

    # Edge-level
    aug.edge_index = new_edge_index
    aug.shifts = batch.shifts[edge_mask]
    aug.unit_shifts = batch.unit_shifts[edge_mask]
    for _attr in ("edge_vectors", "edge_lengths"):
        _v = getattr(batch, _attr, None)
        if _v is not None:
            setattr(aug, _attr, _v[edge_mask])

    # Per-atom optional
    for _attr in ("forces", "charges", "density_coefficients"):
        _v = getattr(batch, _attr, None)
        if _v is not None:
            setattr(aug, _attr, _v[atom_mask])

    # Graph-level scalars [n_graphs] → [n_keep]
    for _attr in (
        "head", "weight", "energy",
        "energy_weight", "forces_weight", "stress_weight", "virials_weight",
        "dipole_weight", "charges_weight", "polarizability_weight",
        "elec_temp", "total_charge", "total_spin", "volume", "fermi_level",
    ):
        _v = getattr(batch, _attr, None)
        if _v is not None:
            setattr(aug, _attr, _v[keep])

    # Graph-level multi-dim tensors [n_graphs, ...] → [n_keep, ...]
    for _attr in ("stress", "virials", "dipole", "polarizability"):
        _v = getattr(batch, _attr, None)
        if _v is not None:
            setattr(aug, _attr, _v[keep])

    # Cell: stored as [n_graphs*3, 3] (PyG concatenates row-wise)
    if hasattr(batch, "cell") and batch.cell is not None:
        aug.cell = torch.cat(
            [batch.cell[int(g) * 3 : int(g) * 3 + 3] for g in keep.tolist()],
            dim=0,
        )

    # Recompute ptr if present
    if hasattr(batch, "ptr") and batch.ptr is not None:
        atoms_per_new_graph = torch.zeros(n_keep, dtype=torch.long, device=dev)
        atoms_per_new_graph.scatter_add_(
            0, new_graph_per_atom, torch.ones(n_keep_atoms, dtype=torch.long, device=dev)
        )
        aug.ptr = torch.cat(
            [torch.zeros(1, dtype=torch.long, device=dev),
             torch.cumsum(atoms_per_new_graph, dim=0)]
        )

    return aug


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


# ---------------------------------------------------------------------------
# Augmented config dump (debugging)
# ---------------------------------------------------------------------------


def save_augmented_xyz(
    aug_batch: "torch_geometric.Batch",
    teacher_out: dict,
    path: str,
    z_list: list,
) -> int:
    """Append augmented structures with EMA-teacher labels to an extxyz file.

    Each structure in ``aug_batch`` is written as an extended-XYZ entry with
    ``teacher_energy`` (eV) and ``teacher_forces`` (eV/Å) as properties so you
    can inspect the augmented data in OVITO, ASE, or any XYZ viewer.

    Parameters
    ----------
    aug_batch:
        The rattled/strained batch returned by ``rattle_batch``.
    teacher_out:
        Detached EMA-teacher output dict (keys ``"energy"``, ``"forces"``).
    path:
        File path to write/append to.  Will be created on first call.
    z_list:
        Ordered list of atomic numbers corresponding to the one-hot element
        axis in ``aug_batch.node_attrs``.  Pass ``list(z_table.zs)`` or
        ``model.atomic_numbers.tolist()``.

    Returns
    -------
    int
        Number of structures written in this call.
    """
    import numpy as np

    try:
        import ase.io
        from ase.atoms import Atoms
    except ImportError as exc:
        raise ImportError("ase is required for save_augmented_xyz") from exc

    n_graphs = aug_batch.num_graphs
    positions = aug_batch.positions.detach().cpu().numpy()   # [n_atoms, 3]
    cell_flat = aug_batch.cell.detach().cpu().numpy()        # [n_graphs*3, 3]
    cell_3d = cell_flat.reshape(n_graphs, 3, 3)
    graph_idx = aug_batch.batch.cpu().numpy()                # [n_atoms]
    # node_attrs is one-hot [n_atoms, n_elements]; argmax gives element table index
    elem_indices = aug_batch.node_attrs.cpu().argmax(dim=-1).numpy()
    atomic_nums = np.array([z_list[i] for i in elem_indices])

    energies = teacher_out.get("energy")
    forces = teacher_out.get("forces")

    atoms_list = []
    for g in range(n_graphs):
        mask = graph_idx == g
        at = Atoms(
            numbers=atomic_nums[mask],
            positions=positions[mask],
            cell=cell_3d[g],
            pbc=True,
        )
        if energies is not None:
            at.info["teacher_energy"] = float(energies[g].cpu().item())
        if forces is not None:
            at.arrays["teacher_forces"] = forces[mask].detach().cpu().numpy()
        atoms_list.append(at)

    ase.io.write(path, atoms_list, format="extxyz", append=True)
    return n_graphs
