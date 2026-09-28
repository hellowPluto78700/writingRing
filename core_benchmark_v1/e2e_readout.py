"""End-to-end LIF readout control for Core Benchmark 04.

This run changes only the final readout realization relative to O0:
the same 234->234 hidden backbone and bias-free W are trained end-to-end
through a beta=0.5 output LIF with WCCE on valid mean output spikes.
The selected checkpoint is then evaluated both through the trained LIF and
through a counterfactual analog accumulator using the exact same hidden
network and W.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from .data import loader
from .model import BenchmarkNet, mean_logits, spike_readout, valid_sum
from .protocol import Protocol, Run, SPLITS
from .storage import (
    checkpoint_metadata,
    load_torch,
    save_json,
    save_npz,
    save_torch,
    state_hash,
    validate_checkpoint,
)
from .training import cpu_state, metrics

E2E_LIF_BETA = 0.5


@torch.no_grad()
def evaluate_validation_lif(
    model: BenchmarkNet,
    arrays: dict[str, np.ndarray],
    p: Protocol,
    run: Run,
) -> dict[str, float]:
    model.eval()
    targets: list[np.ndarray] = []
    predictions: list[np.ndarray] = []
    losses: list[float] = []
    for x, y, lengths in loader(arrays, "val", p, run.seed):
        evidence = model(x, lengths)["evidence"]
        output_spikes = spike_readout(
            evidence,
            lengths,
            beta=E2E_LIF_BETA,
            threshold=p.threshold,
            slope=p.surrogate_slope,
        )
        mean_scores = mean_logits(output_spikes, lengths)
        targets.append(y.numpy())
        predictions.append(mean_scores.argmax(1).numpy())
        losses.append(float(F.cross_entropy(mean_scores, y, reduction="sum")))
    result = metrics(np.concatenate(targets), np.concatenate(predictions))
    result["mean_logit_ce"] = sum(losses) / sum(len(y) for y in targets)
    return result


def train_e2e_lif(
    root: Path,
    run: Run,
    p: Protocol,
    lock: dict[str, Any],
    arrays: dict[str, np.ndarray],
) -> BenchmarkNet:
    directory = root / "runs" / run.key
    directory.mkdir(parents=True, exist_ok=True)
    checkpoint_path = directory / "checkpoint.pt"
    if checkpoint_path.exists():
        checkpoint = load_torch(checkpoint_path)
        validate_checkpoint(checkpoint, run, lock)
        if not checkpoint.get("training_complete"):
            raise ValueError(f"Checkpoint is not a completed training run: {run.key}")
        model = BenchmarkNet(run, p)
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        return model

    model = BenchmarkNet(run, p)
    reference = BenchmarkNet(Run("O0", run.seed, "01_objective"), p)
    paired_initialization_verified = all(
        torch.equal(value, reference.state_dict()[name])
        for name, value in model.state_dict().items()
    )
    if not paired_initialization_verified:
        raise AssertionError("E2E LIF initialization is not paired with O0")

    initial = cpu_state(model)
    save_torch(
        directory / "initial.pt",
        {**checkpoint_metadata(run, lock), "model_state_dict": initial},
    )
    parameter_hashes = {
        name: state_hash({name: tensor}) for name, tensor in model.named_parameters()
    }
    optimizer = torch.optim.Adam(
        model.parameters(), lr=p.learning_rate, weight_decay=p.weight_decay
    )
    train_batches = loader(arrays, "train", p, run.seed, shuffle=True)

    best = evaluate_validation_lif(model, arrays, p, run)
    best_state, best_epoch = initial, 0
    history: list[dict[str, Any]] = [
        {
            "epoch": 0,
            "train_loss": None,
            "val_ba": best["ba"],
            "val_mean_logit_ce": best["mean_logit_ce"],
        }
    ]

    for epoch in range(1, p.max_epochs + 1):
        model.train()
        total_loss, count = 0.0, 0
        for x, y, lengths in train_batches:
            optimizer.zero_grad(set_to_none=True)
            evidence = model(x, lengths)["evidence"]
            output_spikes = spike_readout(
                evidence,
                lengths,
                beta=E2E_LIF_BETA,
                threshold=p.threshold,
                slope=p.surrogate_slope,
            )
            # Same WCCE reduction scale as O0: CE on valid mean logits.
            loss = F.cross_entropy(mean_logits(output_spikes, lengths), y)
            if not torch.isfinite(loss):
                raise FloatingPointError(
                    f"{run.key}: nonfinite E2E LIF loss at epoch {epoch}"
                )
            loss.backward()
            if any(
                parameter.grad is not None
                and not torch.isfinite(parameter.grad).all()
                for parameter in model.parameters()
            ):
                raise FloatingPointError(f"{run.key}: nonfinite E2E LIF gradient")
            optimizer.step()
            total_loss += float(loss.detach()) * len(y)
            count += len(y)

        val = evaluate_validation_lif(model, arrays, p, run)
        history.append(
            {
                "epoch": epoch,
                "train_loss": total_loss / count,
                "val_ba": val["ba"],
                "val_mean_logit_ce": val["mean_logit_ce"],
            }
        )
        improved = val["ba"] > best["ba"] + 1e-12 or (
            abs(val["ba"] - best["ba"]) <= 1e-12
            and val["mean_logit_ce"] < best["mean_logit_ce"] - 1e-12
        )
        if improved:
            best_state, best_epoch, best = cpu_state(model), epoch, val
        if epoch == 1 or epoch % 10 == 0:
            print(
                f'{run.key} epoch={epoch} train_loss={total_loss/count:.5f} '
                f'val_ba={val["ba"]:.5f}',
                flush=True,
            )
            save_json(directory / "progress.json", history[-1])
        if epoch >= p.min_epochs and epoch - best_epoch >= p.patience:
            break

    model.load_state_dict(best_state)
    save_json(directory / "history.json", {"rows": history})
    save_torch(
        checkpoint_path,
        {
            **checkpoint_metadata(run, lock),
            "model_state_dict": best_state,
            "training_complete": True,
            "best_epoch": best_epoch,
            "stopped_epoch": epoch,
            "best_val": best,
            "initial_parameter_hashes": parameter_hashes,
            "trainable_parameters": sum(v.numel() for v in model.parameters()),
            "paired_o0_initialization_verified": True,
            "output_beta": E2E_LIF_BETA,
            "output_threshold": p.threshold,
            "output_synaptic_alpha": 0.0,
            "selection_rule": (
                "native validation LIF BA, then valid-mean output-spike CE, "
                "then earliest epoch"
            ),
        },
    )
    return model


@torch.no_grad()
def evaluate_e2e_lif(
    model: BenchmarkNet,
    arrays: dict[str, np.ndarray],
    p: Protocol,
    run: Run,
) -> tuple[list[dict[str, Any]], dict[str, np.ndarray]]:
    model.eval()
    modes = ("R3_e2e_LIF", "R4_e2e_accumulator_swap")
    accumulators: dict[str, dict[str, list[np.ndarray] | float | int]] = {
        mode: {"targets": [], "predictions": [], "scores": [], "loss": 0.0, "count": 0}
        for mode in modes
    }
    predictions: dict[str, np.ndarray] = {
        f"{split}__ids": arrays[f"{split}_ids"] for split in SPLITS
    }

    for split in SPLITS:
        for x, y, lengths in loader(arrays, split, p, run.seed):
            evidence = model(x, lengths)["evidence"]
            lif_spikes = spike_readout(
                evidence,
                lengths,
                beta=E2E_LIF_BETA,
                threshold=p.threshold,
                slope=p.surrogate_slope,
            )
            mode_values = {
                "R3_e2e_LIF": lif_spikes,
                "R4_e2e_accumulator_swap": evidence,
            }
            for mode, values in mode_values.items():
                sum_scores = valid_sum(values, lengths)
                mean_scores = mean_logits(values, lengths)
                slot = accumulators[mode]
                slot["targets"].append(y.numpy())
                slot["predictions"].append(sum_scores.argmax(1).numpy())
                slot["scores"].append(sum_scores.numpy())
                slot["loss"] += float(
                    F.cross_entropy(mean_scores, y, reduction="sum")
                )
                slot["count"] += len(y)

        for mode in modes:
            slot = accumulators[mode]
            y_all = np.concatenate(slot["targets"])
            pred_all = np.concatenate(slot["predictions"])
            scores_all = np.concatenate(slot["scores"])
            split_metrics = metrics(y_all, pred_all)
            split_metrics.update(
                {
                    "mean_logit_ce": float(slot["loss"]) / int(slot["count"]),
                    "tie_fraction": float(
                        ((scores_all == scores_all.max(1, keepdims=True)).sum(1) > 1).mean()
                    ),
                    "all_zero_fraction": float(
                        (np.abs(scores_all).sum(1) == 0).mean()
                    ),
                }
            )
            slot[f"{split}_metrics"] = split_metrics
            predictions[f"{mode}__{split}"] = pred_all
            # Clear sample-wise buffers before the next split.
            slot["targets"] = []
            slot["predictions"] = []
            slot["scores"] = []
            slot["loss"] = 0.0
            slot["count"] = 0

    rows: list[dict[str, Any]] = []
    for mode in modes:
        row: dict[str, Any] = {
            "run_key": run.key,
            "case": run.case,
            "seed": run.seed,
            "block": run.block,
            "mode": mode,
            "beta": E2E_LIF_BETA,
            "threshold": p.threshold,
        }
        for split in SPLITS:
            row.update(
                {
                    f"{split}_{name}": value
                    for name, value in accumulators[mode][f"{split}_metrics"].items()
                }
            )
        row["train_test_gap"] = row["train_ba"] - row["test_ba"]
        rows.append(row)
    return rows, predictions


def run_e2e_readout(
    directory: Path,
    root: Path,
    run: Run,
    p: Protocol,
    lock: dict[str, Any],
    arrays: dict[str, np.ndarray],
    *,
    evaluation_only: bool = False,
) -> None:
    checkpoint_path = directory / "checkpoint.pt"
    if evaluation_only:
        if not checkpoint_path.exists():
            raise FileNotFoundError(checkpoint_path)
        checkpoint = load_torch(checkpoint_path)
        validate_checkpoint(checkpoint, run, lock)
        if not checkpoint.get("training_complete"):
            raise ValueError(f"Checkpoint is not a completed training run: {run.key}")
        model = BenchmarkNet(run, p)
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    else:
        model = train_e2e_lif(root, run, p, lock, arrays)

    rows, predictions = evaluate_e2e_lif(model, arrays, p, run)
    save_json(
        directory / "readout.json",
        {
            "rows": rows,
            "training_scope": "full_end_to_end",
            "paired_reference": "O0",
            "output_beta": E2E_LIF_BETA,
            "output_threshold": p.threshold,
            "output_synaptic_alpha": 0.0,
            "output_spike_cap": 1,
            "native_mode": "R3_e2e_LIF",
            "counterfactual_mode": "R4_e2e_accumulator_swap",
            "counterfactual_contract": (
                "same selected hidden network and same W; only bypass output LIF "
                "and accumulate analog Wz_t evidence"
            ),
        },
    )
    save_npz(directory / "predictions.npz", predictions)
