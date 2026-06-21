"""mace_prune CLI: channel-pruning distillation for MACE models."""
import argparse
import logging

import torch

from mace.tools.prune_utils import ChannelPruner, PruneConfig


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prune a MACE model via gradual channel-pruning distillation."
    )
    parser.add_argument("--teacher", required=True, help="Path to teacher .model file")
    parser.add_argument("--config", required=True, help="Path to XYZ config file for distillation")
    parser.add_argument("--target-channels", type=int, required=True)
    parser.add_argument("--start-step", type=int, default=0)
    parser.add_argument("--n-steps", type=int, default=100)
    parser.add_argument("--dt", type=int, default=10)
    parser.add_argument("--w-energy", type=float, default=1.0)
    parser.add_argument("--w-forces", type=float, default=100.0)
    parser.add_argument("--fine-tune-steps", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--output", default="pruned.model")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=4)
    return parser


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = build_parser().parse_args()

    device = torch.device(args.device)
    teacher = torch.load(args.teacher, map_location=device, weights_only=False)
    teacher.eval()

    # Build data loader from XYZ file
    from mace import data, tools
    from mace.tools import torch_geometric

    atoms_list = data.load_from_xyz(
        file_path=args.config,
        config_type_weights={"Default": 1.0},
        energy_key="energy",
        forces_key="forces",
        extract_atomic_energies=False,
    )[1]
    z_table = tools.AtomicNumberTable(
        sorted(set(int(z) for a in atoms_list for z in a.atomic_numbers))
    )
    r_max = float(teacher.r_max)
    atomic_data = [
        data.AtomicData.from_config(c, z_table=z_table, cutoff=r_max)
        for c in atoms_list
    ]
    loader = torch_geometric.dataloader.DataLoader(
        atomic_data, batch_size=args.batch_size, shuffle=True
    )

    config = PruneConfig(
        target_channels=args.target_channels,
        start_step=args.start_step,
        n_steps=args.n_steps,
        dt=args.dt,
        w_energy=args.w_energy,
        w_forces=args.w_forces,
        fine_tune_steps=args.fine_tune_steps,
        lr=args.lr,
    )

    pruner = ChannelPruner(teacher=teacher, data_loader=loader, config=config)
    pruned = pruner.run()

    torch.save(pruned, args.output)
    logging.info("Pruned model saved to %s", args.output)


if __name__ == "__main__":
    main()
