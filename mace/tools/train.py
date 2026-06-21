###########################################################################################
# Training script
# Authors: Ilyes Batatia, Gregor Simm, David Kovacs
# This program is distributed under the MIT License (see MIT.md)
###########################################################################################

import dataclasses
import logging
import time
from collections import defaultdict
from contextlib import contextmanager, nullcontext
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.distributed
from torch.nn.parallel import DistributedDataParallel
from torch.optim import LBFGS
from torch.optim.swa_utils import SWALR, AveragedModel
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch_ema import ExponentialMovingAverage
from torchmetrics import Metric

from mace.cli.visualise_train import TrainingPlotter

from . import torch_geometric
from .checkpoint import CheckpointHandler, CheckpointState
from .distill_utils import filter_batch_by_head
from .torch_tools import to_numpy
from .utils import (
    MetricsLogger,
    compute_mae,
    compute_q95,
    compute_rel_mae,
    compute_rel_rmse,
    compute_rmse,
    filter_nonzero_weight,
)


# Sentinel: printed once per process lifetime when --distill_debug is active
_DISTILL_DEBUG_BANNER_PRINTED: bool = False


@dataclasses.dataclass
class SWAContainer:
    model: AveragedModel
    scheduler: SWALR
    start: int
    loss_fn: torch.nn.Module


def valid_err_log(
    valid_loss,
    eval_metrics,
    logger,
    log_errors,
    epoch=None,
    valid_loader_name="Default",
):
    eval_metrics["mode"] = "eval"
    eval_metrics["epoch"] = epoch
    eval_metrics["head"] = valid_loader_name
    logger.log(eval_metrics)
    if epoch is None:
        inintial_phrase = "Initial"
    else:
        inintial_phrase = f"Epoch {epoch}"
    if log_errors == "PerAtomRMSE":
        error_e = eval_metrics["rmse_e_per_atom"] * 1e3
        error_f = eval_metrics["rmse_f"] * 1e3
        logging.info(
            f"{inintial_phrase}: head: {valid_loader_name}, loss={valid_loss:8.8f}, RMSE_E_per_atom={error_e:8.2f} meV, RMSE_F={error_f:8.2f} meV / A"
        )
    elif (
        log_errors == "PerAtomRMSEstressvirials"
        and eval_metrics["rmse_stress"] is not None
    ):
        error_e = eval_metrics["rmse_e_per_atom"] * 1e3
        error_f = eval_metrics["rmse_f"] * 1e3
        error_stress = eval_metrics["rmse_stress"] * 1e3
        logging.info(
            f"{inintial_phrase}: head: {valid_loader_name}, loss={valid_loss:8.8f}, RMSE_E_per_atom={error_e:8.2f} meV, RMSE_F={error_f:8.2f} meV / A, RMSE_stress={error_stress:8.2f} meV / A^3",
        )
    elif (
        log_errors == "PerAtomRMSEstressvirials"
        and eval_metrics["rmse_virials_per_atom"] is not None
    ):
        error_e = eval_metrics["rmse_e_per_atom"] * 1e3
        error_f = eval_metrics["rmse_f"] * 1e3
        error_virials = eval_metrics["rmse_virials_per_atom"] * 1e3
        logging.info(
            f"{inintial_phrase}: head: {valid_loader_name}, loss={valid_loss:8.8f}, RMSE_E_per_atom={error_e:8.2f} meV, RMSE_F={error_f:8.2f} meV / A, RMSE_virials_per_atom={error_virials:8.2f} meV",
        )
    elif (
        log_errors == "PerAtomMAEstressvirials"
        and eval_metrics["mae_stress_per_atom"] is not None
    ):
        error_e = eval_metrics["mae_e_per_atom"] * 1e3
        error_f = eval_metrics["mae_f"] * 1e3
        error_stress = eval_metrics["mae_stress"] * 1e3
        logging.info(
            f"{inintial_phrase}: head: {valid_loader_name}, loss={valid_loss:8.8f}, MAE_E_per_atom={error_e:8.2f} meV, MAE_F={error_f:8.2f} meV / A, MAE_stress={error_stress:8.2f} meV / A^3"
        )
    elif (
        log_errors == "PerAtomMAEstressvirials"
        and eval_metrics["mae_virials_per_atom"] is not None
    ):
        error_e = eval_metrics["mae_e_per_atom"] * 1e3
        error_f = eval_metrics["mae_f"] * 1e3
        error_virials = eval_metrics["mae_virials"] * 1e3
        logging.info(
            f"{inintial_phrase}: head: {valid_loader_name}, loss={valid_loss:8.8f}, MAE_E_per_atom={error_e:8.2f} meV, MAE_F={error_f:8.2f} meV / A, MAE_virials={error_virials:8.2f} meV"
        )
    elif log_errors == "TotalRMSE":
        error_e = eval_metrics["rmse_e"] * 1e3
        error_f = eval_metrics["rmse_f"] * 1e3
        logging.info(
            f"{inintial_phrase}: head: {valid_loader_name}, loss={valid_loss:8.8f}, RMSE_E={error_e:8.2f} meV, RMSE_F={error_f:8.2f} meV / A",
        )
    elif log_errors == "PerAtomMAE":
        error_e = eval_metrics["mae_e_per_atom"] * 1e3
        error_f = eval_metrics["mae_f"] * 1e3
        logging.info(
            f"{inintial_phrase}: head: {valid_loader_name}, loss={valid_loss:8.8f}, MAE_E_per_atom={error_e:8.2f} meV, MAE_F={error_f:8.2f} meV / A",
        )
    elif log_errors == "TotalMAE":
        error_e = eval_metrics["mae_e"] * 1e3
        error_f = eval_metrics["mae_f"] * 1e3
        logging.info(
            f"{inintial_phrase}: head: {valid_loader_name}, loss={valid_loss:8.8f}, MAE_E={error_e:8.2f} meV, MAE_F={error_f:8.2f} meV / A",
        )
    elif log_errors == "DipoleRMSE":
        error_mu = eval_metrics["rmse_mu_per_atom"] * 1e3
        logging.info(
            f"{inintial_phrase}: head: {valid_loader_name}, loss={valid_loss:8.8f}, RMSE_MU_per_atom={error_mu:8.2f} mDebye",
        )
    elif log_errors == "DipolePolarRMSE":
        error_mu = eval_metrics["rmse_mu_per_atom"] * 1e3
        error_polarizability = eval_metrics["rmse_polarizability_per_atom"] * 1e3
        logging.info(
            f"{inintial_phrase}: head: {valid_loader_name}, loss={valid_loss:.4f}, RMSE_MU_per_atom={error_mu:.2f} me A, RMSE_polarizability_per_atom={error_polarizability:.2f} me A^2 / V",
        )
    elif log_errors == "EnergyDipoleRMSE":
        error_e = eval_metrics["rmse_e_per_atom"] * 1e3
        error_f = eval_metrics["rmse_f"] * 1e3
        error_mu = eval_metrics["rmse_mu_per_atom"] * 1e3
        logging.info(
            f"{inintial_phrase}: head: {valid_loader_name}, loss={valid_loss:8.8f}, RMSE_E_per_atom={error_e:8.2f} meV, RMSE_F={error_f:8.2f} meV / A, RMSE_Mu_per_atom={error_mu:8.2f} mDebye",
        )


def train(
    model: torch.nn.Module,
    loss_fn: torch.nn.Module,
    train_loader: DataLoader,
    valid_loaders: Dict[str, DataLoader],
    optimizer: torch.optim.Optimizer,
    lr_scheduler: torch.optim.lr_scheduler.ExponentialLR,
    start_epoch: int,
    max_num_epochs: int,
    patience: int,
    checkpoint_handler: CheckpointHandler,
    logger: MetricsLogger,
    eval_interval: int,
    output_args: Dict[str, bool],
    device: torch.device,
    log_errors: str,
    swa: Optional[SWAContainer] = None,
    ema: Optional[ExponentialMovingAverage] = None,
    max_grad_norm: Optional[float] = 10.0,
    log_wandb: bool = False,
    distributed: bool = False,
    save_all_checkpoints: bool = False,
    plotter: TrainingPlotter = None,
    distributed_model: Optional[DistributedDataParallel] = None,
    train_sampler: Optional[DistributedSampler] = None,
    rank: Optional[int] = 0,
    *,
    student: Optional[torch.nn.Module] = None,
    student_loss_fn: Optional[torch.nn.Module] = None,
    student_optimizer: Optional[torch.optim.Optimizer] = None,
    student_ema: Optional[ExponentialMovingAverage] = None,
    student_lr_scheduler: Optional[torch.optim.lr_scheduler.ExponentialLR] = None,
    student_checkpoint_handler: Optional[CheckpointHandler] = None,
    student_swa: Optional[SWAContainer] = None,
    distill_warmup_epochs: int = 5,
    rattle_fn=None,
    augment_ratio: int = 1,
    dump_augmented_fn=None,
    distill_debug: bool = False,
    distill_num_heads: int = 1,
    distill_target_head_teacher_idx: int = 0,
    distill_target_head_name: str = "Default",
):
    lowest_loss = np.inf
    valid_loss = np.inf
    patience_counter = 0
    swa_start = True
    keep_last = False
    if log_wandb:
        import wandb

    if max_grad_norm is not None:
        logging.info(f"Using gradient clipping with tolerance={max_grad_norm:.3f}")

    logging.info("")
    logging.info("===========TRAINING===========")
    logging.info("Started training, reporting errors on validation set")
    logging.info("Loss metrics on validation set")
    if student is not None:
        logging.info("")
        logging.info("=========== DISTILLATION MODE ===========")
        if distill_warmup_epochs > 0:
            logging.info(
                f"  Student will begin training at epoch {distill_warmup_epochs} "
                f"(warmup: {distill_warmup_epochs} epochs)"
            )
        else:
            logging.info("  Student training active from epoch 0 (no warmup)")
        if distill_num_heads > 1:
            logging.info(
                f"  Multihead teacher ({distill_num_heads} heads): student is "
                f"single-head ('{distill_target_head_name}'). "
                "Only batches from the target head are used for distillation; "
                "pt_head / replay batches are skipped for the student."
            )
        if distill_debug:
            logging.info(
                "  Debug mode ON — per-step timings will be logged for every batch."
            )
        logging.info("=========================================")
        logging.info("")
    epoch = start_epoch

    # log validation loss before _any_ training
    for valid_loader_name, valid_loader in valid_loaders.items():
        valid_loss_head, eval_metrics = evaluate(
            model=model,
            loss_fn=loss_fn,
            data_loader=valid_loader,
            output_args=output_args,
            device=device,
        )
        valid_err_log(
            valid_loss_head, eval_metrics, logger, log_errors, None, valid_loader_name
        )
    valid_loss = valid_loss_head  # consider only the last head for the checkpoint

    # variable used for broadcast by rank == 0 if epoch loop is exited early, e.g. patience
    exit_now = torch.zeros(1, device=device) if distributed else None
    _prev_distill_enabled = False
    while epoch < max_num_epochs:
        # Determine whether distillation is active this epoch (warmup gate)
        distill_enabled = (student is not None) and (epoch >= distill_warmup_epochs)
        if distill_enabled and not _prev_distill_enabled:
            logging.info(
                f"[Distillation] Warmup complete — student training active from epoch {epoch}"
            )
        _prev_distill_enabled = distill_enabled

        # LR scheduler and SWA update
        if swa is None or epoch < swa.start:
            if epoch > start_epoch:
                lr_scheduler.step(
                    metrics=valid_loss
                )  # Can break if exponential LR, TODO fix that!
                if student_lr_scheduler is not None:
                    student_lr_scheduler.step(metrics=valid_loss)
        else:
            if swa_start:
                logging.info("Changing loss based on Stage Two Weights")
                lowest_loss = np.inf
                swa_start = False
                keep_last = True
                if student_swa is not None:
                    logging.info(
                        "[Distillation] Switching to Stage Two distillation loss"
                    )
            loss_fn = swa.loss_fn
            swa.model.update_parameters(model)
            if epoch > start_epoch:
                swa.scheduler.step()
            if student_swa is not None and distill_enabled:
                student_loss_fn = student_swa.loss_fn
                student_swa.model.update_parameters(student)
                if epoch > start_epoch:
                    student_swa.scheduler.step()

        # Train
        if distributed:
            train_sampler.set_epoch(epoch)
        if "ScheduleFree" in type(optimizer).__name__:
            optimizer.train()
        train_one_epoch(
            model=model,
            loss_fn=loss_fn,
            data_loader=train_loader,
            optimizer=optimizer,
            epoch=epoch,
            output_args=output_args,
            max_grad_norm=max_grad_norm,
            ema=ema,
            logger=logger,
            device=device,
            distributed=distributed,
            distributed_model=distributed_model,
            rank=rank,
            student=student,
            student_loss_fn=student_loss_fn,
            student_optimizer=student_optimizer,
            student_ema=student_ema,
            distill_enabled=distill_enabled,
            rattle_fn=rattle_fn,
            augment_ratio=augment_ratio,
            dump_augmented_fn=dump_augmented_fn,
            distill_debug=distill_debug,
            distill_target_head_teacher_idx=distill_target_head_teacher_idx,
        )
        if distributed:
            torch.distributed.barrier()

        # Validate
        if epoch % eval_interval == 0:
            model_to_evaluate = (
                model if distributed_model is None else distributed_model
            )
            param_context = (
                ema.average_parameters() if ema is not None else nullcontext()
            )
            if "ScheduleFree" in type(optimizer).__name__:
                optimizer.eval()
            with param_context:
                wandb_log_dict = {}
                for valid_loader_name, valid_loader in valid_loaders.items():
                    valid_loss_head, eval_metrics = evaluate(
                        model=model_to_evaluate,
                        loss_fn=loss_fn,
                        data_loader=valid_loader,
                        output_args=output_args,
                        device=device,
                    )
                    if rank == 0:
                        valid_err_log(
                            valid_loss_head,
                            eval_metrics,
                            logger,
                            log_errors,
                            epoch,
                            valid_loader_name,
                        )
                        if log_wandb:
                            wandb_log_dict[valid_loader_name] = {
                                "epoch": epoch,
                                "valid_loss": valid_loss_head,
                                "valid_rmse_e_per_atom": eval_metrics[
                                    "rmse_e_per_atom"
                                ],
                                "valid_rmse_f": eval_metrics["rmse_f"],
                            }
                if plotter and epoch % plotter.plot_frequency == 0:
                    try:
                        plotter.plot(epoch, model_to_evaluate, rank)
                    except Exception as e:  # pylint: disable=broad-except
                        logging.debug(f"Plotting failed: {e}")
                valid_loss = (
                    valid_loss_head  # consider only the last head for the checkpoint
                )
            if log_wandb:
                wandb.log(wandb_log_dict)

            # Student validation against DFT labels (only when distillation is active).
            # The student is single-head; only evaluate on the target head's valid_loader.
            # Batch head indices are remapped to 0 (student's only head).
            if student is not None and distill_enabled and rank == 0:
                if distill_debug:
                    _t_sval = time.time()
                    logging.info(
                        f"[Distill DEBUG E{epoch}] Student validation start "
                        f"(target head: '{distill_target_head_name}')"
                    )
                student_param_ctx = (
                    student_ema.average_parameters()
                    if student_ema is not None
                    else nullcontext()
                )
                with student_param_ctx:
                    # Only validate on the loader that matches the student's target head.
                    _student_val_loaders = {
                        name: loader
                        for name, loader in valid_loaders.items()
                        if name == distill_target_head_name
                    } or valid_loaders  # fallback: use all if name not found
                    for valid_loader_name, valid_loader in _student_val_loaders.items():
                        s_valid_loss, s_metrics = evaluate(
                            model=student,
                            loss_fn=loss_fn,
                            data_loader=valid_loader,
                            output_args=output_args,
                            device=device,
                            head_remap=0,
                        )
                        rmse_e = s_metrics.get("rmse_e_per_atom")
                        rmse_f = s_metrics.get("rmse_f")
                        parts = [f"Student Epoch {epoch} [{valid_loader_name}]: loss={s_valid_loss:.6f}"]
                        if rmse_e is not None:
                            parts.append(f"RMSE_E_per_atom={rmse_e * 1e3:.2f} meV")
                        if rmse_f is not None:
                            parts.append(f"RMSE_F={rmse_f * 1e3:.2f} meV/A")
                        logging.info(", ".join(parts))
                        s_metrics["mode"] = "student_eval"
                        s_metrics["epoch"] = epoch
                        s_metrics["head"] = valid_loader_name
                        logger.log(s_metrics)
                if distill_debug:
                    logging.info(
                        f"[Distill DEBUG E{epoch}] Student validation done "
                        f"({time.time()-_t_sval:.2f}s)"
                    )

            if rank == 0:
                if valid_loss >= lowest_loss:
                    patience_counter += 1
                    if patience_counter >= patience:
                        if swa is not None and epoch < swa.start:
                            logging.info(
                                f"Stopping optimization after {patience_counter} epochs without improvement and starting Stage Two"
                            )
                            epoch = swa.start
                        else:
                            logging.info(
                                f"Stopping optimization after {patience_counter} epochs without improvement"
                            )
                            if exit_now is not None:
                                exit_now.fill_(1)
                    if save_all_checkpoints:
                        param_context = (
                            ema.average_parameters()
                            if ema is not None
                            else nullcontext()
                        )
                        with param_context:
                            checkpoint_handler.save(
                                state=CheckpointState(model, optimizer, lr_scheduler),
                                epochs=epoch,
                                keep_last=True,
                            )
                        if student_checkpoint_handler is not None:
                            student_param_context = (
                                student_ema.average_parameters()
                                if student_ema is not None
                                else nullcontext()
                            )
                            with student_param_context:
                                student_checkpoint_handler.save(
                                    state=CheckpointState(
                                        student, student_optimizer, student_lr_scheduler
                                    ),
                                    epochs=epoch,
                                    keep_last=True,
                                )
                else:
                    lowest_loss = valid_loss
                    patience_counter = 0
                    param_context = (
                        ema.average_parameters() if ema is not None else nullcontext()
                    )
                    with param_context:
                        checkpoint_handler.save(
                            state=CheckpointState(model, optimizer, lr_scheduler),
                            epochs=epoch,
                            keep_last=keep_last,
                        )
                        keep_last = False or save_all_checkpoints
                    if student_checkpoint_handler is not None:
                        student_param_context = (
                            student_ema.average_parameters()
                            if student_ema is not None
                            else nullcontext()
                        )
                        with student_param_context:
                            student_checkpoint_handler.save(
                                state=CheckpointState(
                                    student, student_optimizer, student_lr_scheduler
                                ),
                                epochs=epoch,
                                keep_last=keep_last,
                            )
        if distributed:
            torch.distributed.barrier()
        if exit_now is not None:
            torch.distributed.broadcast(exit_now, src=0)
            if exit_now == 1:
                break

        epoch += 1

    logging.info("Training complete")


def train_one_epoch(
    model: torch.nn.Module,
    loss_fn: torch.nn.Module,
    data_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    output_args: Dict[str, bool],
    max_grad_norm: Optional[float],
    ema: Optional[ExponentialMovingAverage],
    logger: MetricsLogger,
    device: torch.device,
    distributed: bool,
    distributed_model: Optional[DistributedDataParallel] = None,
    rank: Optional[int] = 0,
    *,
    student: Optional[torch.nn.Module] = None,
    student_loss_fn: Optional[torch.nn.Module] = None,
    student_optimizer: Optional[torch.optim.Optimizer] = None,
    student_ema: Optional[ExponentialMovingAverage] = None,
    distill_enabled: bool = False,
    rattle_fn=None,
    augment_ratio: int = 1,
    dump_augmented_fn=None,
    distill_debug: bool = False,
    distill_target_head_teacher_idx: int = 0,
) -> None:
    model_to_train = model if distributed_model is None else distributed_model

    if isinstance(optimizer, LBFGS):
        if distill_enabled:
            logging.warning(
                "Distillation is not supported with LBFGS optimizer. "
                "Student will not be updated this epoch."
            )
        _, opt_metrics = take_step_lbfgs(
            model=model_to_train,
            loss_fn=loss_fn,
            data_loader=data_loader,
            optimizer=optimizer,
            ema=ema,
            output_args=output_args,
            max_grad_norm=max_grad_norm,
            device=device,
            distributed=distributed,
            rank=rank,
        )
        opt_metrics["mode"] = "opt"
        opt_metrics["epoch"] = epoch
        if rank == 0:
            logger.log(opt_metrics)
    else:
        if distill_debug:
            global _DISTILL_DEBUG_BANNER_PRINTED
            _dbg_epoch_start = time.time()
            _dbg_step_data: List[Dict[str, Any]] = []
            if distill_enabled:
                if not _DISTILL_DEBUG_BANNER_PRINTED:
                    _DISTILL_DEBUG_BANNER_PRINTED = True
                    logging.info(
                        "[Distill DEBUG] Pipeline phases logged per step:\n"
                        "  dft        — teacher forward+backward on the labelled batch "
                        "(DFT loss; teacher weights updated here)\n"
                        "  rattle     — random displacements + strain applied to batch "
                        "geometries to create augmented configs\n"
                        "  ema_fwd    — EMA teacher labels the augmented batch "
                        "(outputs detached: no gradient flows back to teacher)\n"
                        "  s_fwd      — student forward pass on augmented batch\n"
                        "  s_bwd+step — student distillation loss, backward, gradient "
                        "clip, optimizer step, student EMA update\n"
                        "  E_loss / F_loss — per-step distillation MSE (eV² and "
                        "(eV/Å)²) before loss weighting"
                    )
                logging.info(
                    f"[Distill DEBUG E{epoch}] Epoch start — distillation ACTIVE"
                )
            else:
                logging.info(
                    f"[Distill DEBUG E{epoch}] Epoch start — distillation WARMUP "
                    f"(student inactive)"
                )

        for _step_idx, batch in enumerate(data_loader):
            _, opt_metrics = take_step(
                model=model_to_train,
                loss_fn=loss_fn,
                batch=batch,
                optimizer=optimizer,
                ema=ema,
                output_args=output_args,
                max_grad_norm=max_grad_norm,
                device=device,
                student=student,
                student_loss_fn=student_loss_fn,
                student_optimizer=student_optimizer,
                student_ema=student_ema,
                distill_enabled=distill_enabled,
                rattle_fn=rattle_fn,
                augment_ratio=augment_ratio,
                dump_augmented_fn=dump_augmented_fn,
                distill_debug=distill_debug,
                distill_target_head_teacher_idx=distill_target_head_teacher_idx,
            )

            # --- Verbose debug logging ---
            if distill_debug and rank == 0:
                if distill_enabled and "_dbg_t_rattle_ms" in opt_metrics:
                    # First step of epoch: log aug batch stats to sanity-check teacher
                    if _step_idx == 0:
                        e_min = opt_metrics.get("_dbg_teacher_e_min")
                        e_max = opt_metrics.get("_dbg_teacher_e_max")
                        f_rms = opt_metrics.get("_dbg_teacher_f_rms")
                        stats = (
                            f"{opt_metrics['_dbg_n_aug_graphs']} graphs / "
                            f"{opt_metrics['_dbg_n_aug_atoms']} atoms"
                        )
                        if e_min is not None:
                            stats += f" | teacher E∈[{e_min:.3f}, {e_max:.3f}] eV"
                        if f_rms is not None:
                            stats += f" | teacher F_rms={f_rms:.3f} eV/Å"
                        logging.info(
                            f"[Distill DEBUG E{epoch} step 0 aug] {stats}"
                        )
                    # Per-step timing one-liner
                    logging.info(
                        f"[Distill DEBUG E{epoch} step {_step_idx}] "
                        f"dft={opt_metrics['_dbg_t_teacher_ms']:.0f}ms "
                        f"rattle={opt_metrics['_dbg_t_rattle_ms']:.0f}ms "
                        f"ema_fwd={opt_metrics['_dbg_t_ema_fwd_ms']:.0f}ms "
                        f"s_fwd={opt_metrics['_dbg_t_s_fwd_ms']:.0f}ms "
                        f"s_bwd+step={opt_metrics['_dbg_t_s_bwd_ms']:.0f}ms "
                        f"| E_loss={opt_metrics.get('distill_energy', 0):.5f} "
                        f"F_loss={opt_metrics.get('distill_forces', 0):.5f} "
                        f"| step_total={opt_metrics['time']*1000:.0f}ms"
                    )
                    # Snapshot before keys are stripped below
                    _dbg_step_data.append(dict(opt_metrics))
                elif "_dbg_t_teacher_ms" in opt_metrics:
                    # Warmup: only teacher is running
                    logging.info(
                        f"[Distill DEBUG E{epoch} step {_step_idx} WARMUP] "
                        f"teacher_dft={opt_metrics['_dbg_t_teacher_ms']:.0f}ms "
                        f"| step_total={opt_metrics['time']*1000:.0f}ms"
                    )

            # Strip debug keys before writing to JSON log (keeps log file clean)
            if distill_debug:
                for k in [k for k in opt_metrics if k.startswith("_dbg_")]:
                    del opt_metrics[k]

            opt_metrics["mode"] = "opt"
            opt_metrics["epoch"] = epoch
            if rank == 0:
                logger.log(opt_metrics)

        # --- Epoch-level summary ---
        if distill_debug and rank == 0 and distill_enabled and _dbg_step_data:
            n = len(_dbg_step_data)
            def _avg(key):
                vals = [m[key] for m in _dbg_step_data if key in m]
                return sum(vals) / len(vals) if vals else 0.0
            epoch_wall = time.time() - _dbg_epoch_start
            logging.info(
                f"[Distill DEBUG E{epoch} SUMMARY] {n} steps in {epoch_wall:.1f}s | "
                f"avg: dft={_avg('_dbg_t_teacher_ms'):.0f}ms "
                f"rattle={_avg('_dbg_t_rattle_ms'):.0f}ms "
                f"ema_fwd={_avg('_dbg_t_ema_fwd_ms'):.0f}ms "
                f"s_fwd={_avg('_dbg_t_s_fwd_ms'):.0f}ms "
                f"s_bwd+step={_avg('_dbg_t_s_bwd_ms'):.0f}ms "
                f"step_total={_avg('time')*1000:.0f}ms | "
                f"avg E_loss={_avg('distill_energy'):.5f} "
                f"F_loss={_avg('distill_forces'):.5f}"
            )


def take_step(
    model: torch.nn.Module,
    loss_fn: torch.nn.Module,
    batch: torch_geometric.batch.Batch,
    optimizer: torch.optim.Optimizer,
    ema: Optional[ExponentialMovingAverage],
    output_args: Dict[str, bool],
    max_grad_norm: Optional[float],
    device: torch.device,
    *,
    student: Optional[torch.nn.Module] = None,
    student_loss_fn: Optional[torch.nn.Module] = None,
    student_optimizer: Optional[torch.optim.Optimizer] = None,
    student_ema: Optional[ExponentialMovingAverage] = None,
    distill_enabled: bool = False,
    rattle_fn=None,
    augment_ratio: int = 1,
    dump_augmented_fn=None,
    distill_debug: bool = False,
    distill_target_head_teacher_idx: int = 0,
) -> Tuple[float, Dict[str, Any]]:
    start_time = time.time()
    batch = batch.to(device)
    batch_dict = batch.to_dict()

    def closure():
        optimizer.zero_grad(set_to_none=True)
        output = model(
            batch_dict,
            training=True,
            compute_force=output_args["forces"],
            compute_virials=output_args["virials"],
            compute_stress=output_args["stress"],
        )
        loss = loss_fn(pred=output, ref=batch)
        loss.backward()
        if max_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)

        return loss

    loss = closure()
    optimizer.step()

    if ema is not None:
        ema.update()

    loss_dict = {
        "loss": to_numpy(loss),
        "time": time.time() - start_time,
    }

    # Record teacher-DFT time before distillation block (debug only)
    if distill_debug:
        loss_dict["_dbg_t_teacher_ms"] = (time.time() - start_time) * 1000

    # --- Student distillation block (passenger: never touches the teacher graph) ---
    if distill_enabled and student is not None:
        # --- Phase 1: Filter to target-head graphs, then rattle / augment ---
        # The student is single-head; skip any graphs that belong to pt_head or
        # other non-target heads so the student only ever sees target-head geometry.
        if distill_debug:
            _t1 = time.time()
        _target_batch = filter_batch_by_head(batch, distill_target_head_teacher_idx)
        if _target_batch is None:
            # Entire batch is non-target (e.g. all pt_head). Skip student this step.
            if distill_debug:
                loss_dict["_dbg_t_rattle_ms"] = 0.0
                loss_dict["_dbg_n_aug_graphs"] = 0
                loss_dict["_dbg_n_aug_atoms"] = 0
            return to_numpy(loss), loss_dict
        aug_batch = rattle_fn(_target_batch)
        if distill_debug:
            loss_dict["_dbg_t_rattle_ms"] = (time.time() - _t1) * 1000
            loss_dict["_dbg_n_aug_graphs"] = aug_batch.num_graphs
            loss_dict["_dbg_n_aug_atoms"] = aug_batch.num_nodes

        # --- Phase 2: EMA teacher forward on augmented batch ---
        # Forces are computed via autograd.grad(energy, positions), so we cannot
        # use torch.no_grad() here.  Detach all outputs immediately after to break
        # the graph before the student loss.
        # The aug_batch is already filtered to target-head graphs only, but we
        # explicitly remap head indices to distill_target_head_teacher_idx so the
        # teacher always uses the user's head E0s/readout (not pt_head).
        if distill_debug:
            _t2 = time.time()
        _compute_stress = student_loss_fn.stress_weight.item() > 0
        _aug_batch_dev = aug_batch.to(device)
        _aug_dict_teacher = _aug_batch_dev.to_dict()
        # Remap to target head in teacher space
        _aug_dict_teacher["head"] = torch.full(
            (aug_batch.num_graphs,),
            distill_target_head_teacher_idx,
            dtype=torch.long,
            device=device,
        )
        if ema is not None:
            with ema.average_parameters():
                teacher_out = model(
                    _aug_dict_teacher,
                    training=False,
                    compute_force=True,
                    compute_virials=False,
                    compute_stress=_compute_stress,
                )
        else:
            teacher_out = model(
                _aug_dict_teacher,
                training=False,
                compute_force=True,
                compute_virials=False,
                compute_stress=_compute_stress,
            )
        if distill_debug:
            loss_dict["_dbg_t_ema_fwd_ms"] = (time.time() - _t2) * 1000

        # Detach all teacher outputs — breaks the graph so student.backward() cannot
        # reach teacher parameters.
        teacher_out = {
            k: v.detach() if isinstance(v, torch.Tensor) else v
            for k, v in teacher_out.items()
        }

        if distill_debug:
            if "energy" in teacher_out and teacher_out["energy"] is not None:
                loss_dict["_dbg_teacher_e_min"] = teacher_out["energy"].min().item()
                loss_dict["_dbg_teacher_e_max"] = teacher_out["energy"].max().item()
            if "forces" in teacher_out and teacher_out["forces"] is not None:
                loss_dict["_dbg_teacher_f_rms"] = (
                    teacher_out["forces"].pow(2).mean().sqrt().item()
                )

        # Optional: save augmented batch + teacher labels to extxyz for debugging
        if dump_augmented_fn is not None:
            dump_augmented_fn(aug_batch, teacher_out)

        # --- Phase 3: Student forward ---
        # Student is single-head (head index 0), so remap the head field.
        if distill_debug:
            _t3 = time.time()
        student_optimizer.zero_grad()
        _aug_dict_student = {**_aug_dict_teacher}  # shallow copy; shares tensors
        _aug_dict_student["head"] = torch.zeros(
            aug_batch.num_graphs, dtype=torch.long, device=device
        )
        student_out = student(
            _aug_dict_student,
            training=True,
            compute_force=True,
            compute_virials=False,
            compute_stress=_compute_stress,
        )
        if distill_debug:
            loss_dict["_dbg_t_s_fwd_ms"] = (time.time() - _t3) * 1000

        # --- Phase 4: Distillation loss + backward + student step ---
        if distill_debug:
            _t4 = time.time()
        s_loss, s_loss_dict = student_loss_fn(student_out, teacher_out, _aug_batch_dev)
        s_loss.backward()
        if max_grad_norm:
            torch.nn.utils.clip_grad_norm_(student.parameters(), max_grad_norm)
        student_optimizer.step()
        if student_ema is not None:
            student_ema.update()
        if distill_debug:
            loss_dict["_dbg_t_s_bwd_ms"] = (time.time() - _t4) * 1000

        # Merge student loss dict (keys: distill_energy, distill_forces, distill_stress)
        for k, v in s_loss_dict.items():
            loss_dict[k] = v
    # --- End student distillation block ---

    return loss, loss_dict


def take_step_lbfgs(
    model: torch.nn.Module,
    loss_fn: torch.nn.Module,
    data_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    ema: Optional[ExponentialMovingAverage],
    output_args: Dict[str, bool],
    max_grad_norm: Optional[float],
    device: torch.device,
    distributed: bool,
    rank: int,
) -> Tuple[float, Dict[str, Any]]:
    start_time = time.time()
    logging.debug(
        f"Max Allocated: {torch.cuda.max_memory_allocated() / 1024**2:.2f} MB"
    )

    total_sample_count = 0
    for batch in data_loader:
        total_sample_count += batch.num_graphs

    if distributed:
        global_sample_count = torch.tensor(total_sample_count, device=device)
        torch.distributed.all_reduce(
            global_sample_count, op=torch.distributed.ReduceOp.SUM
        )
        total_sample_count = global_sample_count.item()

    signal = torch.zeros(1, device=device) if distributed else None

    def closure():
        if distributed:
            if rank == 0:
                signal.fill_(1)
                torch.distributed.broadcast(signal, src=0)

            for param in model.parameters():
                torch.distributed.broadcast(param.data, src=0)

        optimizer.zero_grad(set_to_none=True)
        total_loss = torch.tensor(0.0, device=device)

        # Process each batch and then collect the results we pass to the optimizer
        for batch in data_loader:
            batch = batch.to(device)
            batch_dict = batch.to_dict()
            output = model(
                batch_dict,
                training=True,
                compute_force=output_args["forces"],
                compute_virials=output_args["virials"],
                compute_stress=output_args["stress"],
            )
            batch_loss = loss_fn(pred=output, ref=batch)
            batch_loss = batch_loss * (batch.num_graphs / total_sample_count)

            batch_loss.backward()
            total_loss += batch_loss

        if max_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)

        if distributed:
            torch.distributed.all_reduce(total_loss, op=torch.distributed.ReduceOp.SUM)
        return total_loss

    if distributed:
        if rank == 0:
            loss = optimizer.step(closure)
            signal.fill_(0)
            torch.distributed.broadcast(signal, src=0)
        else:
            while True:
                # Other ranks wait for signals from rank 0
                torch.distributed.broadcast(signal, src=0)
                if signal.item() == 0:
                    break
                if signal.item() == 1:
                    loss = closure()

        for param in model.parameters():
            torch.distributed.broadcast(param.data, src=0)
    else:
        loss = optimizer.step(closure)

    if ema is not None:
        ema.update()

    loss_dict = {
        "loss": to_numpy(loss),
        "time": time.time() - start_time,
    }

    return loss, loss_dict


# Keep parameters frozen/active after evaluation
@contextmanager
def preserve_grad_state(model):
    # save the original requires_grad state for all parameters
    requires_grad_backup = {param: param.requires_grad for param in model.parameters()}
    try:
        # temporarily disable gradients for all parameters
        for param in model.parameters():
            param.requires_grad = False
        yield  # perform evaluation here
    finally:
        # restore the original requires_grad states
        for param, requires_grad in requires_grad_backup.items():
            param.requires_grad = requires_grad


def evaluate(
    model: torch.nn.Module,
    loss_fn: torch.nn.Module,
    data_loader: DataLoader,
    output_args: Dict[str, bool],
    device: torch.device,
    head_remap: Optional[int] = None,
) -> Tuple[float, Dict[str, Any]]:

    metrics = MACELoss(loss_fn=loss_fn).to(device)

    start_time = time.time()

    with preserve_grad_state(model):
        for batch in data_loader:
            batch = batch.to(device)
            batch_dict = batch.to_dict()
            if head_remap is not None:
                batch_dict["head"] = torch.full(
                    (batch.num_graphs,), head_remap, dtype=torch.long, device=device
                )
            output = model(
                batch_dict,
                training=False,
                compute_force=output_args["forces"],
                compute_virials=output_args["virials"],
                compute_stress=output_args["stress"],
            )
            avg_loss, aux = metrics(batch, output)
    avg_loss, aux = metrics.compute()
    aux["time"] = time.time() - start_time
    metrics.reset()

    return avg_loss, aux


class MACELoss(Metric):
    def __init__(self, loss_fn: torch.nn.Module):
        super().__init__()
        self.loss_fn = loss_fn
        self.add_state("total_loss", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("num_data", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("E_computed", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("delta_es", default=[], dist_reduce_fx="cat")
        self.add_state("delta_es_per_atom", default=[], dist_reduce_fx="cat")
        self.add_state("Fs_computed", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("fs", default=[], dist_reduce_fx="cat")
        self.add_state("delta_fs", default=[], dist_reduce_fx="cat")
        self.add_state(
            "stress_computed", default=torch.tensor(0.0), dist_reduce_fx="sum"
        )
        self.add_state("delta_stress", default=[], dist_reduce_fx="cat")
        self.add_state(
            "virials_computed", default=torch.tensor(0.0), dist_reduce_fx="sum"
        )
        self.add_state("delta_virials", default=[], dist_reduce_fx="cat")
        self.add_state("delta_virials_per_atom", default=[], dist_reduce_fx="cat")
        self.add_state("Mus_computed", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("mus", default=[], dist_reduce_fx="cat")
        self.add_state("delta_mus", default=[], dist_reduce_fx="cat")
        self.add_state("delta_mus_per_atom", default=[], dist_reduce_fx="cat")
        self.add_state(
            "polarizability_computed", default=torch.tensor(0.0), dist_reduce_fx="sum"
        )
        self.add_state("delta_polarizability", default=[], dist_reduce_fx="cat")
        self.add_state(
            "delta_polarizability_per_atom", default=[], dist_reduce_fx="cat"
        )

    def update(self, batch, output):  # pylint: disable=arguments-differ
        loss = self.loss_fn(pred=output, ref=batch)
        self.total_loss += loss
        self.num_data += batch.num_graphs

        if output.get("energy") is not None and batch.energy is not None:
            self.delta_es.append(batch.energy - output["energy"])
            self.delta_es_per_atom.append(
                (batch.energy - output["energy"]) / (batch.ptr[1:] - batch.ptr[:-1])
            )
            self.E_computed += filter_nonzero_weight(
                batch, self.delta_es, batch.weight, batch.energy_weight
            )
        if output.get("forces") is not None and batch.forces is not None:
            self.fs.append(batch.forces)
            self.delta_fs.append(batch.forces - output["forces"])
            self.Fs_computed += filter_nonzero_weight(
                batch,
                self.delta_fs,
                batch.weight,
                batch.forces_weight,
                spread_atoms=True,
            )
        if output.get("stress") is not None and batch.stress is not None:
            self.delta_stress.append(batch.stress - output["stress"])
            self.stress_computed += filter_nonzero_weight(
                batch, self.delta_stress, batch.weight, batch.stress_weight
            )
        if output.get("virials") is not None and batch.virials is not None:
            self.delta_virials.append(batch.virials - output["virials"])
            self.delta_virials_per_atom.append(
                (batch.virials - output["virials"])
                / (batch.ptr[1:] - batch.ptr[:-1]).view(-1, 1, 1)
            )
            self.virials_computed += filter_nonzero_weight(
                batch, self.delta_virials, batch.weight, batch.virials_weight
            )
        if output.get("dipole") is not None and batch.dipole is not None:
            self.mus.append(batch.dipole)
            self.delta_mus.append(batch.dipole - output["dipole"])
            self.delta_mus_per_atom.append(
                (batch.dipole - output["dipole"])
                / (batch.ptr[1:] - batch.ptr[:-1]).unsqueeze(-1)
            )
            self.Mus_computed += filter_nonzero_weight(
                batch,
                self.delta_mus,
                batch.weight,
                batch.dipole_weight,
                spread_quantity_vector=False,
            )
        if (
            output.get("polarizability") is not None
            and batch.polarizability is not None
        ):
            self.delta_polarizability.append(
                batch.polarizability - output["polarizability"]
            )
            self.delta_polarizability_per_atom.append(
                (batch.polarizability - output["polarizability"])
                / (batch.ptr[1:] - batch.ptr[:-1]).unsqueeze(-1).unsqueeze(-1)
            )
            self.polarizability_computed += filter_nonzero_weight(
                batch,
                self.delta_polarizability,
                batch.weight,
                batch.polarizability_weight,
                spread_quantity_vector=False,
            )

    def convert(self, delta: Union[torch.Tensor, List[torch.Tensor]]) -> np.ndarray:
        if isinstance(delta, list):
            delta = torch.cat(delta)
        return to_numpy(delta)

    def compute(self):

        class NoneMultiply:
            def __mul__(self, other):
                return NoneMultiply()

            def __rmul__(self, other):
                return NoneMultiply()

            def __imul__(self, other):
                return NoneMultiply()

            def __format__(self, format_spec):
                return str(None)

        aux = defaultdict(NoneMultiply)
        aux["loss"] = to_numpy(self.total_loss / self.num_data).item()
        if self.E_computed:
            delta_es = self.convert(self.delta_es)
            delta_es_per_atom = self.convert(self.delta_es_per_atom)
            aux["mae_e"] = compute_mae(delta_es)
            aux["mae_e_per_atom"] = compute_mae(delta_es_per_atom)
            aux["rmse_e"] = compute_rmse(delta_es)
            aux["rmse_e_per_atom"] = compute_rmse(delta_es_per_atom)
            aux["q95_e"] = compute_q95(delta_es)
        if self.Fs_computed:
            fs = self.convert(self.fs)
            delta_fs = self.convert(self.delta_fs)
            aux["mae_f"] = compute_mae(delta_fs)
            aux["rel_mae_f"] = compute_rel_mae(delta_fs, fs)
            aux["rmse_f"] = compute_rmse(delta_fs)
            aux["rel_rmse_f"] = compute_rel_rmse(delta_fs, fs)
            aux["q95_f"] = compute_q95(delta_fs)
        if self.stress_computed:
            delta_stress = self.convert(self.delta_stress)
            aux["mae_stress"] = compute_mae(delta_stress)
            aux["rmse_stress"] = compute_rmse(delta_stress)
            aux["q95_stress"] = compute_q95(delta_stress)
        if self.virials_computed:
            delta_virials = self.convert(self.delta_virials)
            delta_virials_per_atom = self.convert(self.delta_virials_per_atom)
            aux["mae_virials"] = compute_mae(delta_virials)
            aux["rmse_virials"] = compute_rmse(delta_virials)
            aux["rmse_virials_per_atom"] = compute_rmse(delta_virials_per_atom)
            aux["q95_virials"] = compute_q95(delta_virials)
        if self.Mus_computed:
            mus = self.convert(self.mus)
            delta_mus = self.convert(self.delta_mus)
            delta_mus_per_atom = self.convert(self.delta_mus_per_atom)
            aux["mae_mu"] = compute_mae(delta_mus)
            aux["mae_mu_per_atom"] = compute_mae(delta_mus_per_atom)
            aux["rel_mae_mu"] = compute_rel_mae(delta_mus, mus)
            aux["rmse_mu"] = compute_rmse(delta_mus)
            aux["rmse_mu_per_atom"] = compute_rmse(delta_mus_per_atom)
            aux["rel_rmse_mu"] = compute_rel_rmse(delta_mus, mus)
            aux["q95_mu"] = compute_q95(delta_mus)
        if self.polarizability_computed:
            delta_polarizability = self.convert(self.delta_polarizability)
            delta_polarizability_per_atom = self.convert(
                self.delta_polarizability_per_atom
            )
            aux["mae_polarizability"] = compute_mae(delta_polarizability)
            aux["mae_polarizability_per_atom"] = compute_mae(
                delta_polarizability_per_atom
            )
            aux["rmse_polarizability"] = compute_rmse(delta_polarizability)
            aux["rmse_polarizability_per_atom"] = compute_rmse(
                delta_polarizability_per_atom
            )
            aux["q95_polarizability"] = compute_q95(delta_polarizability)

        return aux["loss"], aux
