from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any, Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from scripts import experiment_3_0_1_single_tau_objectives as exp3
from scripts import experiment_4_0_1_multispike_macro_lif as exp401
from scripts import experiment_7_2_two_layer_tau_training as exp72
from scripts import experiment_7_2_6_output_readout_loss_shaping as exp726
from scripts import experiment_7_2_6_1_sum_vs_mean_ce as exp7261


EXPERIMENT_ID = "experiment_7_2_6_3_linear_vs_lif_frozen_l2"
PROTOCOL_VERSION = "linear_vs_lif_frozen_l2_v1"
ARCHITECTURE = exp726.ARCHITECTURE
REGULARIZATION = exp72.TASK_ONLY
SEEDS = tuple(exp726.SEEDS)
HIDDEN_WIDTH = exp726.HIDDEN_WIDTH
N_CLASSES = exp726.N_CLASSES
THRESHOLD = float(exp726.THRESHOLD)
OUTPUT_CAP = 1
OUTPUT_ALPHA = 0.0
LIF_BETA = 0.5
CE_LOGIT_GAIN = 1.0
MAX_EPOCHS = exp726.MAX_EPOCHS
MIN_EPOCHS = exp726.MIN_EPOCHS
PATIENCE = exp726.PATIENCE
BETA_SWEEP = tuple(exp726.BETA_SWEEP)

HEAD_LINEAR = "linear"
HEAD_LIF = "lif"
HEAD_MODES = (HEAD_LINEAR, HEAD_LIF)


@dataclass(frozen=True)
class SourceSpec:
    seed: int

    @property
    def baseline_spec(self) -> exp7261.RunSpec:
        return exp7261.RunSpec(self.seed)

    @property
    def key(self) -> str:
        return f"{ARCHITECTURE}__{REGULARIZATION}__seed{self.seed}"


@dataclass(frozen=True)
class HeadSpec:
    seed: int
    mode: str

    @property
    def source(self) -> SourceSpec:
        return SourceSpec(self.seed)

    @property
    def key(self) -> str:
        return f"{self.source.key}__head_{self.mode}"


@dataclass(frozen=True)
class Config:
    repo_root: Path
    results_dir: Path
    device: str = "cpu"
    batch_size: int = exp72.BATCH_SIZE
    threads: int = 1
    max_epochs: int = MAX_EPOCHS


def find_repo_root(start: Path | None = None) -> Path:
    return exp726.find_repo_root(start)


def results_dir(repo_root: Path) -> Path:
    return repo_root / "notebooks" / "artifacts" / EXPERIMENT_ID / PROTOCOL_VERSION


def source_specs() -> list[SourceSpec]:
    return [SourceSpec(seed) for seed in SEEDS]


def head_specs() -> list[HeadSpec]:
    return [HeadSpec(seed, mode) for seed in SEEDS for mode in HEAD_MODES]


def _save_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _path(root: Path, kind: str, key: str, suffix: str) -> Path:
    return root / kind / f"{key}{suffix}"


def _source_config(config: Config) -> exp7261.Config:
    return exp7261.Config(
        config.repo_root,
        exp7261.results_dir(config.repo_root),
        config.device,
        config.batch_size,
        config.threads,
        config.max_epochs,
    )


def _head_pair_seed(seed: int, role: str) -> int:
    # Head mode is deliberately excluded so Linear/LIF see identical W init and sample order.
    return exp3.dseed(seed, EXPERIMENT_ID, "paired_frozen_head", role)


def validate_source(spec: SourceSpec) -> None:
    if spec.seed not in SEEDS:
        raise ValueError(spec.seed)


def validate_head(spec: HeadSpec) -> None:
    validate_source(spec.source)
    if spec.mode not in HEAD_MODES:
        raise ValueError(spec.mode)


def _cache_path(config: Config, spec: SourceSpec) -> Path:
    return _path(config.results_dir, "frozen_l2_cache", spec.key, ".npz")


def _cache_meta_path(config: Config, spec: SourceSpec) -> Path:
    return _path(config.results_dir, "frozen_l2_cache", spec.key, ".json")


def prepare_cache(spec: SourceSpec, config: Config, force: bool = False) -> Path:
    validate_source(spec)
    destination = _cache_path(config, spec)
    metadata_path = _cache_meta_path(config, spec)
    if destination.exists() and metadata_path.exists() and not force:
        return destination

    source_checkpoint = exp7261._path(
        exp7261.results_dir(config.repo_root), "checkpoints", spec.baseline_spec, ".pt"
    )
    source_eval = exp7261._path(
        exp7261.results_dir(config.repo_root), "evaluations", spec.baseline_spec, ".json"
    )
    missing = [str(p) for p in (source_checkpoint, source_eval) if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "Exp7.2.6.3 requires Exp7.2.6.1 Mean-CE bias-free artifacts. "
            f"Missing for seed {spec.seed}: {missing}"
        )

    data = exp3.prepare_data(config.repo_root)
    device = torch.device(config.device)
    torch.set_num_threads(config.threads)
    model, checkpoint = exp7261._load_mean_model(spec.baseline_spec, data, _source_config(config))
    if model.output_linear.bias is not None:
        raise RuntimeError("Exp7.2.6.3 source must be the bias-free Exp7.2.6.1 model")
    loaders = exp726._e2e_loaders(data, spec.baseline_spec.c0_spec, config.batch_size, False)

    arrays: dict[str, np.ndarray] = {
        "source_w": model.output_linear.weight.detach().cpu().numpy().astype(np.float32)
    }
    split_meta: dict[str, Any] = {}
    for split, loader in loaders.items():
        l2, y, lengths = exp726._extract_l2_generic(model, loader, device)
        l2_np = l2.numpy()
        if not np.all((l2_np == 0) | (l2_np == 1)):
            raise RuntimeError(f"{spec.key}/{split}: L2 cache must be binary")
        arrays[f"{split}_l2"] = l2_np.astype(np.uint8, copy=False)
        arrays[f"{split}_y"] = y.astype(np.int64, copy=False)
        arrays[f"{split}_lengths"] = lengths.astype(np.int64, copy=False)
        split_meta[split] = {
            "shape": list(l2_np.shape),
            "n_samples": int(len(y)),
            "mean_firing_fraction_full": float(l2_np.mean()),
        }

    destination.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(destination, **arrays)
    _save_json(
        metadata_path,
        {
            "experiment_id": EXPERIMENT_ID,
            "protocol_version": PROTOCOL_VERSION,
            "spec": asdict(spec),
            "source_experiment": exp7261.EXPERIMENT_ID,
            "source_protocol": exp7261.PROTOCOL_VERSION,
            "source_checkpoint": str(source_checkpoint.relative_to(config.repo_root)),
            "source_best_epoch": int(checkpoint["best_epoch"]),
            "source_contract": "Exp7.2.6.1 Mean-CE, bias=False, task_only",
            "ce_logit_gain": CE_LOGIT_GAIN,
            "splits": split_meta,
        },
    )
    return destination


def _load_cache(spec: SourceSpec, config: Config) -> dict[str, Any]:
    path = _cache_path(config, spec)
    if not path.exists():
        raise FileNotFoundError(f"Missing Exp7.2.6.3 cache: {path}")
    with np.load(path, allow_pickle=False) as z:
        return {
            "source_w": z["source_w"].astype(np.float64),
            **{
                split: (
                    z[f"{split}_l2"].copy(),
                    z[f"{split}_y"].copy(),
                    z[f"{split}_lengths"].copy(),
                )
                for split in ("train", "val", "test")
            },
        }


def _metrics(y: np.ndarray, scores: np.ndarray) -> dict[str, float]:
    return exp72._metrics(y, scores.argmax(axis=1))


# -----------------------------------------------------------------------------
# Part A: same frozen L2 + same frozen W_linear, change only readout dynamics.
# -----------------------------------------------------------------------------

def run_mechanism(spec: SourceSpec, config: Config, force: bool = False) -> dict[str, Any]:
    validate_source(spec)
    out_path = _path(config.results_dir, "mechanism_evaluations", spec.key, ".json")
    if out_path.exists() and not force:
        return json.loads(out_path.read_text(encoding="utf-8"))

    cache = _load_cache(spec, config)
    W = cache["source_w"]
    val_l2, val_y, val_lengths = cache["val"]
    test_l2, test_y, test_lengths = cache["test"]

    # Primary experiment: input gain=1 throughout. No CE is involved here.
    analog = exp726._analog_scores(test_l2, test_lengths, W, scale=1.0)
    if1 = exp726._simulate_unipolar(test_l2, test_lengths, W, beta=1.0, cap=1, scale=1.0)
    lif05 = exp726._simulate_unipolar(test_l2, test_lengths, W, beta=LIF_BETA, cap=1, scale=1.0)

    ladder = {
        "A0_linear_analog": _metrics(test_y, analog),
        "A1_if_beta1_charge": _metrics(test_y, if1["charge_scores"]),
        "A2_if_beta1_spike_count": _metrics(test_y, if1["counts"]),
        "A3_lif_beta05_spike_count": _metrics(test_y, lif05["counts"]),
    }

    beta_rows: list[dict[str, Any]] = []
    firing: dict[str, dict[str, float]] = {}
    for beta in BETA_SWEEP:
        sim = exp726._simulate_unipolar(test_l2, test_lengths, W, beta=beta, cap=1, scale=1.0)
        beta_rows.append({"beta": float(beta), **_metrics(test_y, sim["counts"])})
        firing[f"source_w_beta_{beta:g}_cap1_gain1"] = sim["diagnostics"]

    # Secondary diagnostics. These never replace the gain=1 primary result.
    selected_gain, gain_table = exp726.calibrate_gain(val_l2, val_y, val_lengths, W)
    gain_path = _path(config.results_dir, "secondary_input_gain", spec.key, ".csv")
    gain_path.parent.mkdir(parents=True, exist_ok=True)
    gain_table.to_csv(gain_path, index=False)
    if_gain = exp726._simulate_unipolar(test_l2, test_lengths, W, beta=1.0, cap=1, scale=selected_gain)
    lif_gain = exp726._simulate_unipolar(test_l2, test_lengths, W, beta=LIF_BETA, cap=1, scale=selected_gain)
    cap31_if = exp726._simulate_unipolar(test_l2, test_lengths, W, beta=1.0, cap=31, scale=1.0)
    cap31_lif = exp726._simulate_unipolar(test_l2, test_lengths, W, beta=LIF_BETA, cap=31, scale=1.0)
    bipolar_if = exp726._simulate_bipolar(test_l2, test_lengths, W, beta=1.0, cap=1, scale=1.0)
    bipolar_lif = exp726._simulate_bipolar(test_l2, test_lengths, W, beta=LIF_BETA, cap=1, scale=1.0)

    secondary = {
        "input_gain_selected_on_validation": float(selected_gain),
        "if_beta1_gain_calibrated": _metrics(test_y, if_gain["counts"]),
        "lif_beta05_gain_calibrated": _metrics(test_y, lif_gain["counts"]),
        "if_beta1_cap31_gain1": _metrics(test_y, cap31_if["counts"]),
        "lif_beta05_cap31_gain1": _metrics(test_y, cap31_lif["counts"]),
        "if_beta1_bipolar_gain1": _metrics(test_y, bipolar_if["scores"]),
        "lif_beta05_bipolar_gain1": _metrics(test_y, bipolar_lif["scores"]),
    }
    firing.update(
        {
            "source_w_if_beta1_gain1": if1["diagnostics"],
            "source_w_lif_beta05_gain1": lif05["diagnostics"],
            "source_w_if_beta1_gain_calibrated": if_gain["diagnostics"],
            "source_w_lif_beta05_gain_calibrated": lif_gain["diagnostics"],
            "source_w_if_beta1_cap31_gain1": cap31_if["diagnostics"],
            "source_w_lif_beta05_cap31_gain1": cap31_lif["diagnostics"],
            "source_w_if_beta1_bipolar_gain1": bipolar_if["diagnostics"],
            "source_w_lif_beta05_bipolar_gain1": bipolar_lif["diagnostics"],
        }
    )

    a0_ba = ladder["A0_linear_analog"]["balanced_accuracy"]
    a1_ba = ladder["A1_if_beta1_charge"]["balanced_accuracy"]
    a2_ba = ladder["A2_if_beta1_spike_count"]["balanced_accuracy"]
    a3_ba = ladder["A3_lif_beta05_spike_count"]["balanced_accuracy"]
    payload = {
        "experiment_id": EXPERIMENT_ID,
        "protocol_version": PROTOCOL_VERSION,
        "spec": asdict(spec),
        "primary_contract": {
            "frozen_l2": True,
            "frozen_source_w_linear": True,
            "input_gain": 1.0,
            "ce_logit_gain": CE_LOGIT_GAIN,
            "bias": False,
            "output_alpha": OUTPUT_ALPHA,
            "lif_beta": LIF_BETA,
            "cap": OUTPUT_CAP,
        },
        "mechanism_ladder": ladder,
        "beta_sweep": beta_rows,
        "firing_diagnostics": firing,
        "secondary_controls": secondary,
        "decomposition_pp": {
            "charge_sanity_delta_pp": 100.0 * (a0_ba - a1_ba),
            "spike_conversion_penalty_pp": 100.0 * (a1_ba - a2_ba),
            "leakage_penalty_pp": 100.0 * (a2_ba - a3_ba),
            "total_linear_to_lif_pp": 100.0 * (a0_ba - a3_ba),
        },
        "sanity": {
            "A0_A1_prediction_equal": bool(np.array_equal(analog.argmax(1), if1["charge_scores"].argmax(1))),
            "A0_A1_max_score_error": float(np.max(np.abs(analog - if1["charge_scores"]))),
            "if_beta1_charge_identity_max_error": float(if1["diagnostics"]["max_abs_charge_identity_error"]),
        },
    }
    _save_json(out_path, payload)
    return payload


# -----------------------------------------------------------------------------
# Part B: same frozen L2, paired-train W_linear and W_lif with gain=1 Mean-CE.
# -----------------------------------------------------------------------------

class FrozenReadoutHead(nn.Module):
    def __init__(self, mode: str, n_classes: int = N_CLASSES) -> None:
        super().__init__()
        if mode not in HEAD_MODES:
            raise ValueError(mode)
        self.mode = mode
        self.output_linear = nn.Linear(HIDDEN_WIDTH, n_classes, bias=False)
        self.output_lif = (
            exp401.MacroMultiSpikeLIF(
                beta=LIF_BETA,
                threshold=THRESHOLD,
                max_spikes_per_dt=OUTPUT_CAP,
                surrogate_slope=exp72.SURROGATE_SLOPE,
            )
            if mode == HEAD_LIF
            else None
        )

    def forward_trajectory(self, l2: torch.Tensor) -> dict[str, torch.Tensor]:
        evidence = self.output_linear(l2)
        payload = {"evidence": evidence}
        if self.output_lif is None:
            return payload
        batch, steps, classes = evidence.shape
        mem = torch.zeros(batch, classes, device=evidence.device, dtype=evidence.dtype)
        spikes: list[torch.Tensor] = []
        for t in range(steps):
            spk, mem, _ = self.output_lif(evidence[:, t], mem)
            spikes.append(spk)
        payload["spikes"] = torch.stack(spikes, dim=1)
        return payload


def _valid_mean(values: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
    return exp726._valid_sum(values, lengths) / lengths.clamp_min(1).to(values.dtype).unsqueeze(1)


def _head_ce_logits(model: FrozenReadoutHead, l2: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
    tr = model.forward_trajectory(l2)
    values = tr["evidence"] if model.mode == HEAD_LINEAR else tr["spikes"]
    # Explicit gain=1.0. Do not call exp50.deployment_ce_logits (which uses gain=5).
    return CE_LOGIT_GAIN * _valid_mean(values, lengths)


def _cached_loader(
    split: tuple[np.ndarray, np.ndarray, np.ndarray],
    batch_size: int,
    shuffle: bool,
    seed: int,
) -> DataLoader:
    l2, y, lengths = split
    generator = torch.Generator().manual_seed(int(seed))
    return DataLoader(
        TensorDataset(torch.from_numpy(l2), torch.from_numpy(y), torch.from_numpy(lengths)),
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator,
        num_workers=0,
    )


def _cached_loaders(cache: dict[str, Any], seed: int, batch_size: int, shuffle_train: bool) -> dict[str, DataLoader]:
    return {
        split: _cached_loader(
            cache[split],
            batch_size,
            shuffle_train if split == "train" else False,
            _head_pair_seed(seed, f"{split}_loader"),
        )
        for split in ("train", "val", "test")
    }


def _evaluate_head(model: FrozenReadoutHead, loader: Iterable, device: torch.device) -> dict[str, float]:
    ys: list[np.ndarray] = []
    preds: list[np.ndarray] = []
    loss_sum = 0.0
    n = 0
    model.eval()
    with torch.no_grad():
        for l2, y, lengths in loader:
            l2 = l2.to(device=device, dtype=torch.float32)
            y = y.to(device)
            lengths = lengths.to(device)
            logits = _head_ce_logits(model, l2, lengths)
            loss = F.cross_entropy(logits, y)
            ys.append(y.cpu().numpy())
            preds.append(logits.argmax(1).cpu().numpy())
            loss_sum += float(loss) * len(y)
            n += len(y)
    metrics = exp72._metrics(np.concatenate(ys), np.concatenate(preds))
    return {**metrics, "objective_loss": loss_sum / max(n, 1)}


def _plot_history(frame: pd.DataFrame, path: Path, title: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(frame.epoch, frame.train_ba, label="train BA")
    ax.plot(frame.epoch, frame.val_ba, label="val BA")
    ax.set_ylim(0, 1)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Balanced accuracy")
    ax2 = ax.twinx()
    ax2.plot(frame.epoch, frame.train_loss, linestyle="--", label="train loss")
    ax2.plot(frame.epoch, frame.val_loss, linestyle=":", label="val loss")
    ax.legend(loc="upper left")
    ax2.legend(loc="upper right")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def run_head(spec: HeadSpec, config: Config, force: bool = False) -> dict[str, Any]:
    validate_head(spec)
    eval_path = _path(config.results_dir, "head_evaluations", spec.key, ".json")
    checkpoint_path = _path(config.results_dir, "head_checkpoints", spec.key, ".pt")
    if eval_path.exists() and checkpoint_path.exists() and not force:
        return json.loads(eval_path.read_text(encoding="utf-8"))

    cache = _load_cache(spec.source, config)
    device = torch.device(config.device)
    torch.set_num_threads(config.threads)
    exp3.seed_all(_head_pair_seed(spec.seed, "model_init"))
    model = FrozenReadoutHead(spec.mode).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=exp72.LR, weight_decay=exp72.WEIGHT_DECAY)
    train_loader = _cached_loaders(cache, spec.seed, config.batch_size, True)["train"]
    eval_loaders = _cached_loaders(cache, spec.seed, config.batch_size, False)

    best_state: dict[str, torch.Tensor] | None = None
    best_epoch = -1
    best_ba = -1.0
    best_loss = float("inf")
    stopped_epoch = config.max_epochs
    history: list[dict[str, float]] = []

    for epoch in range(1, config.max_epochs + 1):
        model.train()
        train_loss_sum = 0.0
        n = 0
        for l2, y, lengths in train_loader:
            l2 = l2.to(device=device, dtype=torch.float32)
            y = y.to(device)
            lengths = lengths.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = _head_ce_logits(model, l2, lengths)
            loss = F.cross_entropy(logits, y)
            loss.backward()
            optimizer.step()
            train_loss_sum += float(loss.detach()) * len(y)
            n += len(y)

        train_metrics = _evaluate_head(model, eval_loaders["train"], device)
        val_metrics = _evaluate_head(model, eval_loaders["val"], device)
        history.append(
            {
                "epoch": float(epoch),
                "train_ba": float(train_metrics["balanced_accuracy"]),
                "val_ba": float(val_metrics["balanced_accuracy"]),
                "train_loss": train_loss_sum / max(n, 1),
                "val_loss": float(val_metrics["objective_loss"]),
            }
        )
        improved = val_metrics["balanced_accuracy"] > best_ba + 1e-12 or (
            abs(val_metrics["balanced_accuracy"] - best_ba) <= 1e-12
            and val_metrics["objective_loss"] < best_loss
        )
        if improved:
            best_ba = float(val_metrics["balanced_accuracy"])
            best_loss = float(val_metrics["objective_loss"])
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        if epoch >= MIN_EPOCHS and best_epoch > 0 and epoch - best_epoch >= PATIENCE:
            stopped_epoch = epoch
            break

    if best_state is None:
        raise RuntimeError(f"No checkpoint selected for {spec.key}")

    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "experiment_id": EXPERIMENT_ID,
            "protocol_version": PROTOCOL_VERSION,
            "spec": asdict(spec),
            "best_epoch": best_epoch,
            "stopped_epoch": stopped_epoch,
            "best_val_ba": best_ba,
            "best_val_objective_loss": best_loss,
            "ce_logit_gain": CE_LOGIT_GAIN,
            "model_state_dict": best_state,
            "trainable_parameters": HIDDEN_WIDTH * N_CLASSES,
            "pairing": "head mode excluded from model-init and loader seeds",
        },
        checkpoint_path,
    )
    model.load_state_dict(best_state, strict=True)
    final_metrics = {
        split: _evaluate_head(model, loader, device)
        for split, loader in eval_loaders.items()
    }
    payload = {
        "experiment_id": EXPERIMENT_ID,
        "protocol_version": PROTOCOL_VERSION,
        "spec": asdict(spec),
        "ce_logit_gain": CE_LOGIT_GAIN,
        "loss_contract": "CE(valid_mean(head_output), y), gain=1",
        "best_epoch": best_epoch,
        "stopped_epoch": stopped_epoch,
        "metrics": final_metrics,
    }
    _save_json(eval_path, payload)

    hist = pd.DataFrame(history)
    hist_path = _path(config.results_dir, "head_histories", spec.key, ".csv")
    hist_path.parent.mkdir(parents=True, exist_ok=True)
    hist.to_csv(hist_path, index=False)
    _plot_history(hist, _path(config.results_dir, "head_training_curves", spec.key, ".png"), spec.key)
    return payload


def _load_head_weight(spec: HeadSpec, config: Config) -> np.ndarray:
    path = _path(config.results_dir, "head_checkpoints", spec.key, ".pt")
    if not path.exists():
        raise FileNotFoundError(path)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    return payload["model_state_dict"]["output_linear.weight"].numpy().astype(np.float64)


def run_crosscheck(spec: SourceSpec, config: Config, force: bool = False) -> dict[str, Any]:
    validate_source(spec)
    out_path = _path(config.results_dir, "crosscheck_evaluations", spec.key, ".json")
    if out_path.exists() and not force:
        return json.loads(out_path.read_text(encoding="utf-8"))

    cache = _load_cache(spec, config)
    test_l2, test_y, test_lengths = cache["test"]
    Wlin = _load_head_weight(HeadSpec(spec.seed, HEAD_LINEAR), config)
    Wlif = _load_head_weight(HeadSpec(spec.seed, HEAD_LIF), config)

    x1 = exp726._analog_scores(test_l2, test_lengths, Wlin, scale=1.0)
    x2_sim = exp726._simulate_unipolar(test_l2, test_lengths, Wlin, beta=LIF_BETA, cap=1, scale=1.0)
    x3 = exp726._analog_scores(test_l2, test_lengths, Wlif, scale=1.0)
    x4_sim = exp726._simulate_unipolar(test_l2, test_lengths, Wlif, beta=LIF_BETA, cap=1, scale=1.0)

    cross = {
        "X1_wlin_analog": _metrics(test_y, x1),
        "X2_wlin_lif_beta05": _metrics(test_y, x2_sim["counts"]),
        "X3_wlif_analog": _metrics(test_y, x3),
        "X4_wlif_lif_beta05": _metrics(test_y, x4_sim["counts"]),
    }
    x1_ba = cross["X1_wlin_analog"]["balanced_accuracy"]
    x2_ba = cross["X2_wlin_lif_beta05"]["balanced_accuracy"]
    x3_ba = cross["X3_wlif_analog"]["balanced_accuracy"]
    x4_ba = cross["X4_wlif_lif_beta05"]["balanced_accuracy"]
    payload = {
        "experiment_id": EXPERIMENT_ID,
        "protocol_version": PROTOCOL_VERSION,
        "spec": asdict(spec),
        "crosscheck": cross,
        "paired_head_training": {
            "ce_logit_gain": CE_LOGIT_GAIN,
            "bias": False,
            "linear_loss": "CE(valid_mean(Wz), y)",
            "lif_loss": "CE(valid_mean(S), y)",
            "same_initial_w": True,
            "same_loader_order": True,
        },
        "decomposition_pp": {
            "projection_quality_gap_x1_minus_x3_pp": 100.0 * (x1_ba - x3_ba),
            "wlin_dynamics_gap_x1_minus_x2_pp": 100.0 * (x1_ba - x2_ba),
            "wlif_dynamics_gap_x3_minus_x4_pp": 100.0 * (x3_ba - x4_ba),
            "native_head_gap_x1_minus_x4_pp": 100.0 * (x1_ba - x4_ba),
        },
        "firing_diagnostics": {
            "X2_wlin_lif_beta05": x2_sim["diagnostics"],
            "X4_wlif_lif_beta05": x4_sim["diagnostics"],
        },
    }
    _save_json(out_path, payload)
    return payload


# -----------------------------------------------------------------------------
# Aggregation-only finalizer.
# -----------------------------------------------------------------------------

def _aggregate(frame: pd.DataFrame, groups: list[str], values: list[str]) -> pd.DataFrame:
    grouped = frame.groupby(groups, dropna=False)[values]
    return pd.concat(
        [
            grouped.size().rename("n"),
            grouped.mean().add_suffix("_mean"),
            grouped.std(ddof=1).fillna(0).add_suffix("_std"),
        ],
        axis=1,
    ).reset_index()


def finalize(config: Config) -> dict[str, Any]:
    missing: list[str] = []
    for spec in source_specs():
        for kind in ("mechanism_evaluations", "crosscheck_evaluations"):
            p = _path(config.results_dir, kind, spec.key, ".json")
            if not p.exists():
                missing.append(str(p))
    for spec in head_specs():
        p = _path(config.results_dir, "head_evaluations", spec.key, ".json")
        if not p.exists():
            missing.append(str(p))
    if missing:
        raise FileNotFoundError("Exp7.2.6.3 incomplete; missing:\n" + "\n".join(missing))

    mechanism_rows: list[dict[str, Any]] = []
    beta_rows: list[dict[str, Any]] = []
    secondary_rows: list[dict[str, Any]] = []
    firing_rows: list[dict[str, Any]] = []
    decomposition_rows: list[dict[str, Any]] = []
    for spec in source_specs():
        payload = json.loads(_path(config.results_dir, "mechanism_evaluations", spec.key, ".json").read_text())
        for condition, metrics in payload["mechanism_ladder"].items():
            mechanism_rows.append({"seed": spec.seed, "condition": condition, **metrics})
        for row in payload["beta_sweep"]:
            beta_rows.append({"seed": spec.seed, **row})
        sec = payload["secondary_controls"]
        for condition, value in sec.items():
            if isinstance(value, dict):
                secondary_rows.append({"seed": spec.seed, "condition": condition, **value})
        for condition, diag in payload["firing_diagnostics"].items():
            firing_rows.append({"seed": spec.seed, "condition": condition, **diag})
        decomposition_rows.append({"seed": spec.seed, **payload["decomposition_pp"]})

    head_rows: list[dict[str, Any]] = []
    for spec in head_specs():
        payload = json.loads(_path(config.results_dir, "head_evaluations", spec.key, ".json").read_text())
        for split, metrics in payload["metrics"].items():
            head_rows.append({"seed": spec.seed, "mode": spec.mode, "split": split, **metrics})

    cross_rows: list[dict[str, Any]] = []
    cross_decomp_rows: list[dict[str, Any]] = []
    cross_firing_rows: list[dict[str, Any]] = []
    for spec in source_specs():
        payload = json.loads(_path(config.results_dir, "crosscheck_evaluations", spec.key, ".json").read_text())
        for condition, metrics in payload["crosscheck"].items():
            cross_rows.append({"seed": spec.seed, "condition": condition, **metrics})
        cross_decomp_rows.append({"seed": spec.seed, **payload["decomposition_pp"]})
        for condition, diag in payload["firing_diagnostics"].items():
            cross_firing_rows.append({"seed": spec.seed, "condition": condition, **diag})

    root = config.results_dir
    frames = {
        "mechanism_runs.csv": pd.DataFrame(mechanism_rows),
        "beta_sweep_runs.csv": pd.DataFrame(beta_rows),
        "secondary_controls_runs.csv": pd.DataFrame(secondary_rows),
        "firing_diagnostics_runs.csv": pd.DataFrame(firing_rows),
        "mechanism_decomposition_runs.csv": pd.DataFrame(decomposition_rows),
        "head_runs.csv": pd.DataFrame(head_rows),
        "crosscheck_runs.csv": pd.DataFrame(cross_rows),
        "crosscheck_decomposition_runs.csv": pd.DataFrame(cross_decomp_rows),
        "crosscheck_firing_runs.csv": pd.DataFrame(cross_firing_rows),
    }
    for name, frame in frames.items():
        frame.to_csv(root / name, index=False)

    metric_cols = ["accuracy", "balanced_accuracy", "macro_f1"]
    mechanism_summary = _aggregate(frames["mechanism_runs.csv"], ["condition"], metric_cols)
    beta_summary = _aggregate(frames["beta_sweep_runs.csv"], ["beta"], metric_cols)
    secondary_summary = _aggregate(frames["secondary_controls_runs.csv"], ["condition"], metric_cols)
    head_summary = _aggregate(frames["head_runs.csv"], ["mode", "split"], metric_cols + ["objective_loss"])
    cross_summary = _aggregate(frames["crosscheck_runs.csv"], ["condition"], metric_cols)
    mechanism_summary.to_csv(root / "mechanism_summary.csv", index=False)
    beta_summary.to_csv(root / "beta_sweep_summary.csv", index=False)
    secondary_summary.to_csv(root / "secondary_controls_summary.csv", index=False)
    head_summary.to_csv(root / "head_summary.csv", index=False)
    cross_summary.to_csv(root / "crosscheck_summary.csv", index=False)

    manifest = {
        "experiment_id": EXPERIMENT_ID,
        "protocol_version": PROTOCOL_VERSION,
        "architecture": ARCHITECTURE,
        "regularization": REGULARIZATION,
        "seeds": list(SEEDS),
        "source": "Exp7.2.6.1 Mean-CE bias=False task_only",
        "ce_logit_gain": CE_LOGIT_GAIN,
        "primary_input_gain": 1.0,
        "no_e2e_retraining": True,
        "cache_tasks": len(source_specs()),
        "mechanism_tasks": len(source_specs()),
        "head_training_tasks": len(head_specs()),
        "crosscheck_tasks": len(source_specs()),
        "notebook_contract": "analysis-only; reads finalized CSV/JSON artifacts only",
        "primary_question": "Why do Linear and LIF readouts differ on the same frozen L2 sequence?",
    }
    _save_json(root / "manifest.json", manifest)
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Exp7.2.6.3 frozen-L2 Linear-vs-LIF decomposition")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=exp72.BATCH_SIZE)
    parser.add_argument("--max-epochs", type=int, default=MAX_EPOCHS)
    sub = parser.add_subparsers(dest="command", required=True)
    for name, n in (("prepare-cache", len(source_specs())), ("mechanism", len(source_specs())), ("train-head", len(head_specs())), ("crosscheck", len(source_specs()))):
        p = sub.add_parser(name)
        p.add_argument("--array-task-id", type=int, required=True, choices=range(n))
        p.add_argument("--force", action="store_true")
    sub.add_parser("finalize")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    repo_root = find_repo_root(Path.cwd())
    config = Config(
        repo_root=repo_root,
        results_dir=results_dir(repo_root),
        device=args.device,
        batch_size=args.batch_size,
        threads=args.threads,
        max_epochs=args.max_epochs,
    )
    config.results_dir.mkdir(parents=True, exist_ok=True)

    if args.command == "prepare-cache":
        path = prepare_cache(source_specs()[args.array_task_id], config, args.force)
        print(path)
    elif args.command == "mechanism":
        payload = run_mechanism(source_specs()[args.array_task_id], config, args.force)
        print(json.dumps(payload["decomposition_pp"], indent=2, sort_keys=True))
    elif args.command == "train-head":
        payload = run_head(head_specs()[args.array_task_id], config, args.force)
        print(json.dumps(payload["metrics"]["test"], indent=2, sort_keys=True))
    elif args.command == "crosscheck":
        payload = run_crosscheck(source_specs()[args.array_task_id], config, args.force)
        print(json.dumps(payload["decomposition_pp"], indent=2, sort_keys=True))
    elif args.command == "finalize":
        print(json.dumps(finalize(config), indent=2, sort_keys=True))
    else:
        raise ValueError(args.command)


if __name__ == "__main__":
    main()
