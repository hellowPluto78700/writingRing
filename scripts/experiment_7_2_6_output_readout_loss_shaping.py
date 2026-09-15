from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
from typing import Any, Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from scripts import experiment_0_1_general_comparison as exp01
from scripts import experiment_3_0_1_single_tau_objectives as exp3
from scripts import experiment_4_0_1_multispike_macro_lif as exp401
from scripts import experiment_5_0_local_evidence_objectives as exp50
from scripts import experiment_7_2_two_layer_tau_training as exp72
from scripts import experiment_7_2_3_objective_temporal_dynamics as exp723
from scripts import experiment_7_2_5_output_synaptic_alpha as exp725


EXPERIMENT_ID = "experiment_7_2_6_output_readout_loss_shaping"
PROTOCOL_VERSION = "output_readout_loss_shaping_v1"

ARCHITECTURE = "234x234"
ARCHITECTURES = {ARCHITECTURE: exp723.ARCHITECTURES[ARCHITECTURE]}
REGULARIZATIONS = exp725.REGULARIZATIONS
SEEDS = exp725.SEEDS
OBJECTIVE = "whole_count_ce"
HIDDEN_WIDTH = exp725.HIDDEN_WIDTH
N_CLASSES = 12
THRESHOLD = float(exp72.THRESHOLD)
OUTPUT_CAP = 1
SOURCE_ALPHA = 0.0
SOURCE_BETA = 0.5
MAX_EPOCHS = 100
MIN_EPOCHS = 20
PATIENCE = 30

HEAD_ANALOG = "analog"
HEAD_LIF = "lif"
HEAD_MODES = (HEAD_ANALOG, HEAD_LIF)

E2E_ANALOG = "c0_analog"
E2E_IF = "c1_if_beta10"
E2E_LIF = "c2_lif_beta05_reuse"
E2E_TRAIN_CONDITIONS = (E2E_ANALOG, E2E_IF)
E2E_ALL_CONDITIONS = (E2E_ANALOG, E2E_IF, E2E_LIF)

PROBE_MATCHED_WC = "l2_wholecount_matched_biasfree"
PROBE_STD_WC = exp725.VALID_WC_PROBE
PROBE_F250 = exp725.VALID_F250_PROBE
PROBE_SOURCES = (PROBE_MATCHED_WC, PROBE_STD_WC, PROBE_F250)

BETA_SWEEP = (1.0, 0.95, 0.9, 0.8, 0.7, 0.6, 0.5)


@dataclass(frozen=True)
class SourceSpec:
    regularization: str
    seed: int

    @property
    def key(self) -> str:
        return f"{ARCHITECTURE}__{self.regularization}__seed{self.seed}"

    @property
    def exp725_spec(self) -> exp725.E2ESpec:
        return exp725.E2ESpec(
            ARCHITECTURE,
            self.regularization,
            self.seed,
            OBJECTIVE,
            SOURCE_ALPHA,
        )


@dataclass(frozen=True)
class FrozenHeadSpec:
    regularization: str
    seed: int
    mode: str

    @property
    def source(self) -> SourceSpec:
        return SourceSpec(self.regularization, self.seed)

    @property
    def key(self) -> str:
        return f"{self.source.key}__head_{self.mode}"


@dataclass(frozen=True)
class E2ESpec:
    regularization: str
    seed: int
    condition: str

    @property
    def source(self) -> SourceSpec:
        return SourceSpec(self.regularization, self.seed)

    @property
    def key(self) -> str:
        return f"{ARCHITECTURE}__{self.regularization}__seed{self.seed}__{self.condition}"


@dataclass(frozen=True)
class Config:
    repo_root: Path
    results_dir: Path
    device: str = "cpu"
    batch_size: int = exp72.BATCH_SIZE
    threads: int = 1
    max_epochs: int = MAX_EPOCHS


def find_repo_root(start: Path | None = None) -> Path:
    return exp725.find_repo_root(start)


def results_dir(repo_root: Path) -> Path:
    return repo_root / "notebooks" / "artifacts" / EXPERIMENT_ID / PROTOCOL_VERSION


def source_specs() -> list[SourceSpec]:
    return [SourceSpec(r, s) for r in REGULARIZATIONS for s in SEEDS]


def frozen_head_specs() -> list[FrozenHeadSpec]:
    return [FrozenHeadSpec(src.regularization, src.seed, mode) for src in source_specs() for mode in HEAD_MODES]


def e2e_train_specs() -> list[E2ESpec]:
    return [E2ESpec(r, s, c) for r in REGULARIZATIONS for s in SEEDS for c in E2E_TRAIN_CONDITIONS]


def probe_specs() -> list[E2ESpec]:
    return [E2ESpec(r, s, c) for r in REGULARIZATIONS for s in SEEDS for c in E2E_ALL_CONDITIONS]


def validate_source_spec(spec: SourceSpec) -> None:
    if spec.regularization not in REGULARIZATIONS:
        raise ValueError(spec.regularization)
    if spec.seed not in SEEDS:
        raise ValueError(spec.seed)


def validate_head_spec(spec: FrozenHeadSpec) -> None:
    validate_source_spec(spec.source)
    if spec.mode not in HEAD_MODES:
        raise ValueError(spec.mode)


def validate_e2e_spec(spec: E2ESpec, allow_reuse: bool = True) -> None:
    validate_source_spec(spec.source)
    allowed = E2E_ALL_CONDITIONS if allow_reuse else E2E_TRAIN_CONDITIONS
    if spec.condition not in allowed:
        raise ValueError(spec.condition)


def _save_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _source_path(root: Path, kind: str, spec: SourceSpec, suffix: str) -> Path:
    return root / kind / f"{spec.key}{suffix}"


def _head_path(root: Path, kind: str, spec: FrozenHeadSpec, suffix: str) -> Path:
    return root / kind / "frozen_heads" / f"{spec.key}{suffix}"


def _mechanism_path(root: Path, kind: str, spec: SourceSpec, suffix: str) -> Path:
    return root / kind / f"{spec.key}{suffix}"


def _e2e_path(root: Path, kind: str, spec: E2ESpec, suffix: str) -> Path:
    return root / kind / "e2e" / spec.condition / f"{spec.key}{suffix}"


def _reference_seed(spec: SourceSpec, role: str) -> int:
    # Reuse the Exp7.2.5 alpha=0 seed stream exactly, so C0/C1 and reused C2
    # share the same model initialization and loader order.
    return exp725.e2e_pair_seed(spec.exp725_spec, role)


def _head_pair_seed(spec: SourceSpec, role: str) -> int:
    # Head mode is intentionally excluded.
    return exp3.dseed(spec.seed, EXPERIMENT_ID, "frozen_head_pair", spec.regularization, role)


def _probe_pair_seed(spec: SourceSpec, role: str) -> int:
    # E2E condition is intentionally excluded.
    return exp3.dseed(spec.seed, EXPERIMENT_ID, "probe_pair", spec.regularization, role)


def _valid_mask(lengths: torch.Tensor, steps: int, dtype: torch.dtype) -> torch.Tensor:
    positions = torch.arange(steps, device=lengths.device)[None, :]
    return (positions < lengths[:, None]).to(dtype)


def _valid_sum(values: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
    mask = _valid_mask(lengths, values.shape[1], values.dtype).unsqueeze(-1)
    return (values * mask).sum(dim=1)


def _numpy_valid_mask(lengths: np.ndarray, steps: int) -> np.ndarray:
    return np.arange(steps)[None, :] < lengths[:, None]


def _whole_count_np(l2: np.ndarray, lengths: np.ndarray) -> np.ndarray:
    valid = _numpy_valid_mask(lengths, l2.shape[1]).astype(l2.dtype)[..., None]
    return (l2 * valid).sum(axis=1)


def _metrics(y: np.ndarray, scores: np.ndarray) -> dict[str, float]:
    return exp72._metrics(y, scores.argmax(axis=1))


def _aggregate(df: pd.DataFrame, groups: list[str], values: list[str]) -> pd.DataFrame:
    grouped = df.groupby(groups, dropna=False)[values]
    return pd.concat(
        [
            grouped.size().rename("n"),
            grouped.mean().add_suffix("_mean"),
            grouped.std(ddof=1).fillna(0).add_suffix("_std"),
        ],
        axis=1,
    ).reset_index()


# -----------------------------------------------------------------------------
# Exp7.2.5 source checkpoint / frozen L2 cache
# -----------------------------------------------------------------------------

def _exp725_checkpoint_path(spec: SourceSpec, repo_root: Path) -> Path:
    return exp725._e2e_path(
        exp725.results_dir(repo_root),
        "checkpoints",
        spec.exp725_spec,
        ".pt",
    )


def _exp725_evaluation_path(spec: SourceSpec, repo_root: Path) -> Path:
    return exp725._e2e_path(
        exp725.results_dir(repo_root),
        "evaluations",
        spec.exp725_spec,
        ".json",
    )


def _load_exp725_model(
    spec: SourceSpec,
    data: exp3.Data,
    repo_root: Path,
    device: torch.device,
) -> tuple[exp725.PairedE2ESNN, dict[str, Any]]:
    checkpoint_path = _exp725_checkpoint_path(spec, repo_root)
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Exp7.2.6 requires Exp7.2.5 C2 checkpoint {checkpoint_path}. "
            "Run Exp7.2.5 234x234 whole_count_ce alpha=0 beta=.5 first."
        )
    payload = torch.load(checkpoint_path, map_location=device)
    model = exp725.PairedE2ESNN(spec.exp725_spec, len(data.labels), data.fs).to(device)
    model.load_state_dict(payload["model_state_dict"], strict=True)
    model.eval()
    return model, payload


def prepare_frozen_cache(
    spec: SourceSpec,
    data: exp3.Data,
    config: Config,
    force: bool = False,
) -> Path:
    validate_source_spec(spec)
    destination = _source_path(config.results_dir, "frozen_l2_cache", spec, ".npz")
    metadata = _source_path(config.results_dir, "frozen_l2_cache", spec, ".json")
    if destination.exists() and metadata.exists() and not force:
        return destination

    torch.set_num_threads(config.threads)
    device = torch.device(config.device)
    model, checkpoint_payload = _load_exp725_model(spec, data, config.repo_root, device)
    loaders = exp725._e2e_loaders(data, spec.exp725_spec, config.batch_size, False)
    extracted = {
        name: exp725._extract_e2e_l2(model, loader, device)
        for name, loader in loaders.items()
    }

    arrays: dict[str, np.ndarray] = {}
    split_meta: dict[str, Any] = {}
    for name, (l2, y, lengths) in extracted.items():
        arr = l2.numpy()
        if not np.all((arr == 0) | (arr == 1)):
            raise RuntimeError(f"{spec.key}/{name}: L2 is not binary")
        compact = arr.astype(np.uint8, copy=False)
        arrays[f"{name}_l2"] = compact
        arrays[f"{name}_y"] = y.astype(np.int64, copy=False)
        arrays[f"{name}_lengths"] = lengths.astype(np.int64, copy=False)
        split_meta[name] = {
            "shape": list(compact.shape),
            "n_samples": int(len(y)),
            "mean_firing_fraction_full": float(compact.mean()),
        }

    destination.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(destination, **arrays)
    _save_json(
        metadata,
        {
            "experiment_id": EXPERIMENT_ID,
            "protocol_version": PROTOCOL_VERSION,
            "source_experiment": exp725.EXPERIMENT_ID,
            "source_protocol": exp725.PROTOCOL_VERSION,
            "source_spec": asdict(spec),
            "source_exp725_spec": asdict(spec.exp725_spec),
            "source_checkpoint": str(_exp725_checkpoint_path(spec, config.repo_root).relative_to(config.repo_root)),
            "source_best_epoch": int(checkpoint_payload["best_epoch"]),
            "storage_dtype": "uint8",
            "splits": split_meta,
        },
    )
    return destination


def _load_cache(spec: SourceSpec, config: Config) -> dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    path = _source_path(config.results_dir, "frozen_l2_cache", spec, ".npz")
    if not path.exists():
        raise FileNotFoundError(f"Missing frozen L2 cache: {path}")
    with np.load(path, allow_pickle=False) as z:
        return {
            name: (
                z[f"{name}_l2"].copy(),
                z[f"{name}_y"].copy(),
                z[f"{name}_lengths"].copy(),
            )
            for name in ("train", "val", "test")
        }


def _cached_loader(
    split: tuple[np.ndarray, np.ndarray, np.ndarray],
    batch_size: int,
    shuffle: bool,
    seed: int,
) -> DataLoader:
    l2, y, lengths = split
    generator = torch.Generator()
    generator.manual_seed(int(seed))
    return DataLoader(
        TensorDataset(torch.from_numpy(l2), torch.from_numpy(y), torch.from_numpy(lengths)),
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator,
        num_workers=0,
    )


def _cached_loaders(
    cache: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]],
    spec: SourceSpec,
    batch_size: int,
    shuffle_train: bool,
) -> dict[str, DataLoader]:
    return {
        name: _cached_loader(
            split,
            batch_size,
            shuffle_train if name == "train" else False,
            _head_pair_seed(spec, f"{name}_loader"),
        )
        for name, split in cache.items()
    }


# -----------------------------------------------------------------------------
# Paired frozen heads
# -----------------------------------------------------------------------------

class FrozenMatchedHead(nn.Module):
    def __init__(self, mode: str, n_classes: int) -> None:
        super().__init__()
        if mode not in HEAD_MODES:
            raise ValueError(mode)
        self.mode = mode
        self.output_linear = nn.Linear(HIDDEN_WIDTH, n_classes, bias=False)
        self.output_lif = (
            exp401.MacroMultiSpikeLIF(
                beta=SOURCE_BETA,
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
        batch, steps, _ = evidence.shape
        mem = torch.zeros(batch, evidence.shape[-1], device=evidence.device, dtype=evidence.dtype)
        spikes: list[torch.Tensor] = []
        for t in range(steps):
            spk, mem, _ = self.output_lif(evidence[:, t], mem)
            spikes.append(spk)
        payload["spikes"] = torch.stack(spikes, dim=1)
        return payload


def _head_logits(model: FrozenMatchedHead, l2: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
    tr = model.forward_trajectory(l2)
    if model.mode == HEAD_ANALOG:
        return _valid_sum(tr["evidence"], lengths)
    return _valid_sum(tr["spikes"], lengths)


def _evaluate_frozen_head(
    model: FrozenMatchedHead,
    loader: Iterable,
    device: torch.device,
) -> dict[str, Any]:
    labels: list[np.ndarray] = []
    preds: list[np.ndarray] = []
    loss_sum = 0.0
    n = 0
    model.eval()
    with torch.no_grad():
        for l2, y, lengths in loader:
            l2 = l2.to(device=device, dtype=torch.float32)
            y = y.to(device)
            lengths = lengths.to(device)
            logits = _head_logits(model, l2, lengths)
            loss = F.cross_entropy(logits, y)
            labels.append(y.cpu().numpy())
            preds.append(logits.argmax(1).cpu().numpy())
            loss_sum += float(loss) * len(y)
            n += len(y)
    y_true = np.concatenate(labels)
    metrics = exp72._metrics(y_true, np.concatenate(preds))
    return {**metrics, "objective_loss": loss_sum / max(n, 1)}


def _early_stop(epoch: int, best_epoch: int) -> bool:
    return epoch >= MIN_EPOCHS and best_epoch > 0 and epoch - best_epoch >= PATIENCE


def _plot_history(rows: list[dict[str, float]], path: Path, title: str) -> None:
    frame = pd.DataFrame(rows)
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
    ax.set_title(title)
    ax.legend(loc="upper left")
    ax2.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def run_frozen_head(
    spec: FrozenHeadSpec,
    data: exp3.Data,
    config: Config,
    force: bool = False,
) -> dict[str, Any]:
    validate_head_spec(spec)
    eval_path = _head_path(config.results_dir, "evaluations", spec, ".json")
    checkpoint_path = _head_path(config.results_dir, "checkpoints", spec, ".pt")
    if eval_path.exists() and checkpoint_path.exists() and not force:
        return json.loads(eval_path.read_text(encoding="utf-8"))

    cache = _load_cache(spec.source, config)
    device = torch.device(config.device)
    torch.set_num_threads(config.threads)
    exp3.seed_all(_head_pair_seed(spec.source, "model_init"))
    model = FrozenMatchedHead(spec.mode, len(data.labels)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=exp72.LR, weight_decay=exp72.WEIGHT_DECAY)
    train_loader = _cached_loaders(cache, spec.source, config.batch_size, True)["train"]
    eval_loaders = _cached_loaders(cache, spec.source, config.batch_size, False)

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
            logits = _head_logits(model, l2, lengths)
            loss = F.cross_entropy(logits, y)
            loss.backward()
            optimizer.step()
            train_loss_sum += float(loss.detach()) * len(y)
            n += len(y)
        train_metrics = _evaluate_frozen_head(model, eval_loaders["train"], device)
        val_metrics = _evaluate_frozen_head(model, eval_loaders["val"], device)
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
        if _early_stop(epoch, best_epoch):
            stopped_epoch = epoch
            break

    if best_state is None:
        raise RuntimeError(f"No frozen-head checkpoint selected for {spec.key}")
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "experiment_id": EXPERIMENT_ID,
            "protocol_version": PROTOCOL_VERSION,
            "block": "frozen_head",
            "spec": asdict(spec),
            "best_epoch": best_epoch,
            "stopped_epoch": stopped_epoch,
            "best_val_ba": best_ba,
            "best_val_objective_loss": best_loss,
            "model_state_dict": best_state,
            "trainable_parameters": HIDDEN_WIDTH * len(data.labels),
        },
        checkpoint_path,
    )
    model.load_state_dict(best_state, strict=True)
    final_metrics = {
        split: _evaluate_frozen_head(model, loader, device)
        for split, loader in eval_loaders.items()
    }
    payload = {
        "experiment_id": EXPERIMENT_ID,
        "protocol_version": PROTOCOL_VERSION,
        "block": "frozen_head",
        "spec": asdict(spec),
        "best_epoch": best_epoch,
        "stopped_epoch": stopped_epoch,
        "metrics": final_metrics,
    }
    _save_json(eval_path, payload)
    hist_path = _head_path(config.results_dir, "histories", spec, ".csv")
    hist_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(history).to_csv(hist_path, index=False)
    _plot_history(history, _head_path(config.results_dir, "training_curves", spec, ".png"), spec.key)
    return payload


def _load_head_weight(spec: FrozenHeadSpec, config: Config) -> np.ndarray:
    path = _head_path(config.results_dir, "checkpoints", spec, ".pt")
    if not path.exists():
        raise FileNotFoundError(path)
    payload = torch.load(path, map_location="cpu")
    return payload["model_state_dict"]["output_linear.weight"].numpy().astype(np.float64)


# -----------------------------------------------------------------------------
# Frozen mechanism simulators
# -----------------------------------------------------------------------------

def _analog_scores(l2: np.ndarray, lengths: np.ndarray, W: np.ndarray, scale: float = 1.0) -> np.ndarray:
    return float(scale) * (_whole_count_np(l2, lengths).astype(np.float64) @ W.T)


def _simulate_unipolar(
    l2: np.ndarray,
    lengths: np.ndarray,
    W: np.ndarray,
    beta: float,
    cap: int,
    scale: float = 1.0,
    record_trajectory: bool = False,
) -> dict[str, Any]:
    if not 0.0 <= beta <= 1.0:
        raise ValueError(beta)
    if cap < 1:
        raise ValueError(cap)
    n, steps, _ = l2.shape
    k = W.shape[0]
    mem = np.zeros((n, k), dtype=np.float64)
    counts = np.zeros((n, k), dtype=np.float64)
    input_sum = np.zeros((n, k), dtype=np.float64)
    leak_sum = np.zeros((n, k), dtype=np.float64)
    total_spikes = 0.0
    valid_positions = 0
    potential_multi = 0
    negative_evidence = 0
    trajectory: dict[str, list[np.ndarray]] | None = None
    if record_trajectory:
        trajectory = {name: [] for name in ("evidence", "pre", "spikes", "membrane")}

    for t in range(steps):
        current = float(scale) * (l2[:, t].astype(np.float64) @ W.T)
        vt = t < lengths
        pre = beta * mem + current
        spikes = np.floor(np.maximum(pre, 0.0) / THRESHOLD)
        spikes = np.minimum(spikes, float(cap))
        spikes[~vt] = 0.0
        new_mem = pre - THRESHOLD * spikes
        input_sum[vt] += current[vt]
        counts += spikes
        selected = pre[vt]
        valid_positions += int(selected.size)
        potential_multi += int((selected >= 2.0 * THRESHOLD).sum())
        negative_evidence += int((current[vt] < 0.0).sum())
        total_spikes += float(spikes[vt].sum())
        mem[vt] = new_mem[vt]
        survives = t < (lengths - 1)
        if np.any(survives):
            leak_sum[survives] += (1.0 - beta) * mem[survives]
        if trajectory is not None:
            trajectory["evidence"].append(current.copy())
            trajectory["pre"].append(pre.copy())
            trajectory["spikes"].append(spikes.copy())
            trajectory["membrane"].append(mem.copy())

    residual = mem.copy()
    spike_charge = THRESHOLD * counts
    identity_error = input_sum - (spike_charge + leak_sum + residual)
    valid_steps = max(int(lengths.sum()), 1)
    diagnostics = {
        "mean_total_output_spikes_per_sample": float(counts.sum(axis=1).mean()),
        "mean_spikes_per_output_neuron_sample": float(counts.mean()),
        "silent_sample_fraction": float((counts.sum(axis=1) == 0).mean()),
        "silent_output_neuron_fraction": float((counts == 0).mean()),
        "output_spikes_per_neuron_second": float(total_spikes * 64.0 / max(k * valid_steps, 1)),
        "potential_multi_crossing_fraction": potential_multi / max(valid_positions, 1),
        "negative_evidence_fraction": negative_evidence / max(valid_positions, 1),
        "mean_abs_final_membrane": float(np.abs(residual).mean()),
        "max_abs_charge_identity_error": float(np.abs(identity_error).max()),
    }
    result: dict[str, Any] = {
        "counts": counts,
        "input_sum": input_sum,
        "spike_charge": spike_charge,
        "leak": leak_sum,
        "residual": residual,
        "charge_scores": spike_charge + residual if beta == 1.0 else spike_charge + leak_sum + residual,
        "diagnostics": diagnostics,
    }
    if trajectory is not None:
        result["trajectory"] = {name: np.stack(values, axis=1) for name, values in trajectory.items()}
    return result


def _simulate_bipolar(
    l2: np.ndarray,
    lengths: np.ndarray,
    W: np.ndarray,
    beta: float,
    cap: int,
    scale: float,
) -> dict[str, Any]:
    n, steps, _ = l2.shape
    k = W.shape[0]
    pos_mem = np.zeros((n, k), dtype=np.float64)
    neg_mem = np.zeros((n, k), dtype=np.float64)
    pos_count = np.zeros((n, k), dtype=np.float64)
    neg_count = np.zeros((n, k), dtype=np.float64)
    total = 0.0
    valid_positions = 0
    for t in range(steps):
        evidence = float(scale) * (l2[:, t].astype(np.float64) @ W.T)
        pos_current = np.maximum(evidence, 0.0)
        neg_current = np.maximum(-evidence, 0.0)
        vt = t < lengths
        pos_pre = beta * pos_mem + pos_current
        neg_pre = beta * neg_mem + neg_current
        pos_spk = np.minimum(np.floor(np.maximum(pos_pre, 0.0) / THRESHOLD), float(cap))
        neg_spk = np.minimum(np.floor(np.maximum(neg_pre, 0.0) / THRESHOLD), float(cap))
        pos_spk[~vt] = 0.0
        neg_spk[~vt] = 0.0
        pos_new = pos_pre - THRESHOLD * pos_spk
        neg_new = neg_pre - THRESHOLD * neg_spk
        pos_mem[vt] = pos_new[vt]
        neg_mem[vt] = neg_new[vt]
        pos_count += pos_spk
        neg_count += neg_spk
        total += float(pos_spk[vt].sum() + neg_spk[vt].sum())
        valid_positions += int(pos_pre[vt].size)
    return {
        "scores": pos_count - neg_count,
        "diagnostics": {
            "mean_total_output_spikes_per_sample": float((pos_count + neg_count).sum(axis=1).mean()),
            "silent_sample_fraction": float(((pos_count + neg_count).sum(axis=1) == 0).mean()),
            "output_spikes_per_neuron_second": float(total * 64.0 / max(k * int(lengths.sum()), 1)),
            "positive_event_fraction": float(pos_count.sum() / max(pos_count.sum() + neg_count.sum(), 1.0)),
            "negative_event_fraction": float(neg_count.sum() / max(pos_count.sum() + neg_count.sum(), 1.0)),
            "mean_abs_final_membrane": float((np.abs(pos_mem) + np.abs(neg_mem)).mean() / 2.0),
            "valid_output_positions": float(valid_positions),
        },
    }


def _gain_key(item: tuple[float, float]) -> tuple[float, float]:
    gain, ba = item
    return (-ba, abs(math.log2(gain)), gain)


def calibrate_gain(
    val_l2: np.ndarray,
    val_y: np.ndarray,
    val_lengths: np.ndarray,
    W: np.ndarray,
) -> tuple[float, pd.DataFrame]:
    # Validation only. No test arrays are accepted by this function.
    coarse = [2.0 ** exponent for exponent in range(-4, 5)]
    coarse_rows: list[dict[str, float | str]] = []
    for gain in coarse:
        sim = _simulate_unipolar(val_l2, val_lengths, W, beta=1.0, cap=1, scale=gain)
        ba = float(_metrics(val_y, sim["counts"])["balanced_accuracy"])
        coarse_rows.append({"stage": "coarse", "gain": gain, "balanced_accuracy": ba})
    coarse_best = sorted(
        [(float(r["gain"]), float(r["balanced_accuracy"])) for r in coarse_rows],
        key=_gain_key,
    )[0][0]
    center = math.log2(coarse_best)
    fine = sorted({2.0 ** max(-4.0, min(4.0, center + offset)) for offset in (-0.5, -0.25, 0.0, 0.25, 0.5)})
    rows = coarse_rows[:]
    for gain in fine:
        if any(abs(float(r["gain"]) - gain) < 1e-12 for r in rows):
            continue
        sim = _simulate_unipolar(val_l2, val_lengths, W, beta=1.0, cap=1, scale=gain)
        ba = float(_metrics(val_y, sim["counts"])["balanced_accuracy"])
        rows.append({"stage": "fine", "gain": gain, "balanced_accuracy": ba})
    selected = sorted(
        [(float(r["gain"]), float(r["balanced_accuracy"])) for r in rows],
        key=_gain_key,
    )[0][0]
    table = pd.DataFrame(rows).sort_values(["balanced_accuracy", "gain"], ascending=[False, True])
    return selected, table


def _standardized_accessibility(cache: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]], spec: SourceSpec) -> dict[str, Any]:
    features: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for split, (l2, y, lengths) in cache.items():
        features[split] = (_whole_count_np(l2, lengths).astype(np.float32), y)
    probe = exp01._fit_linear_probe(
        features["train"][0], features["train"][1],
        features["val"][0], features["val"][1],
        features["test"][0], features["test"][1],
        _probe_pair_seed(spec, "part1_standardized_wc"),
    )
    return {
        "probe_C": float(probe["probe_C"]),
        "feature_dim": int(probe["feature_dim"]),
        "metrics": {split: probe[split] for split in ("train", "val", "test")},
    }


def _native_weight(spec: SourceSpec, data: exp3.Data, config: Config) -> np.ndarray:
    model, _ = _load_exp725_model(spec, data, config.repo_root, torch.device(config.device))
    return model.output_linear.weight.detach().cpu().numpy().astype(np.float64)


def _save_margin_decomposition(
    spec: SourceSpec,
    y: np.ndarray,
    analog_scores: np.ndarray,
    lif_sim: dict[str, Any],
    root: Path,
) -> pd.DataFrame:
    analog_pred = analog_scores.argmax(axis=1)
    lif_scores = lif_sim["counts"]
    lif_pred = lif_scores.argmax(axis=1)
    rows: list[dict[str, Any]] = []
    for i in np.where((analog_pred == y) & (lif_pred != y))[0]:
        wrong = lif_scores[i].copy()
        wrong[int(y[i])] = -np.inf
        kstar = int(np.argmax(wrong))
        true = int(y[i])
        rows.append(
            {
                "regularization": spec.regularization,
                "seed": spec.seed,
                "sample_index": int(i),
                "true_class": true,
                "competitor_class": kstar,
                "d_input": float(lif_sim["input_sum"][i, true] - lif_sim["input_sum"][i, kstar]),
                "d_spike": float(lif_sim["spike_charge"][i, true] - lif_sim["spike_charge"][i, kstar]),
                "d_leak": float(lif_sim["leak"][i, true] - lif_sim["leak"][i, kstar]),
                "d_residual": float(lif_sim["residual"][i, true] - lif_sim["residual"][i, kstar]),
                "analog_margin_unscaled": float(analog_scores[i, true] - np.max(np.delete(analog_scores[i], true))),
            }
        )
    frame = pd.DataFrame(rows)
    path = _mechanism_path(root, "margin_decomposition", spec, ".csv")
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    return frame


def _representative_trajectory(
    spec: SourceSpec,
    val_l2: np.ndarray,
    val_y: np.ndarray,
    val_lengths: np.ndarray,
    W: np.ndarray,
    gain: float,
    root: Path,
) -> None:
    analog = _analog_scores(val_l2, val_lengths, W, scale=gain)
    lif05 = _simulate_unipolar(val_l2, val_lengths, W, beta=0.5, cap=1, scale=gain)
    analog_pred = analog.argmax(axis=1)
    lif_pred = lif05["counts"].argmax(axis=1)
    candidates = np.where((analog_pred == val_y) & (lif_pred != val_y))[0]
    if len(candidates) == 0:
        return
    margins = []
    for index in candidates:
        true = int(val_y[index])
        wrong = np.delete(analog[index], true)
        margins.append(float(analog[index, true] - wrong.max()))
    selected = int(candidates[int(np.argmax(margins))])
    true = int(val_y[selected])
    wrong_scores = lif05["counts"][selected].copy()
    wrong_scores[true] = -np.inf
    competitor = int(np.argmax(wrong_scores))

    one_l2 = val_l2[selected:selected + 1]
    one_lengths = val_lengths[selected:selected + 1]
    sim1 = _simulate_unipolar(one_l2, one_lengths, W, beta=1.0, cap=1, scale=gain, record_trajectory=True)
    sim05 = _simulate_unipolar(one_l2, one_lengths, W, beta=0.5, cap=1, scale=gain, record_trajectory=True)
    T = int(one_lengths[0])
    evidence = sim1["trajectory"]["evidence"][0, :T]
    rows = pd.DataFrame(
        {
            "timestep": np.arange(T),
            "true_class": true,
            "competitor_class": competitor,
            "evidence_true": evidence[:, true],
            "evidence_competitor": evidence[:, competitor],
            "mem_beta1_true": sim1["trajectory"]["membrane"][0, :T, true],
            "mem_beta1_competitor": sim1["trajectory"]["membrane"][0, :T, competitor],
            "spike_beta1_true": sim1["trajectory"]["spikes"][0, :T, true],
            "spike_beta1_competitor": sim1["trajectory"]["spikes"][0, :T, competitor],
            "mem_beta05_true": sim05["trajectory"]["membrane"][0, :T, true],
            "mem_beta05_competitor": sim05["trajectory"]["membrane"][0, :T, competitor],
            "spike_beta05_true": sim05["trajectory"]["spikes"][0, :T, true],
            "spike_beta05_competitor": sim05["trajectory"]["spikes"][0, :T, competitor],
        }
    )
    rows["cumulative_evidence_true"] = rows.evidence_true.cumsum()
    rows["cumulative_evidence_competitor"] = rows.evidence_competitor.cumsum()
    rows["cumulative_spike_beta1_true"] = rows.spike_beta1_true.cumsum()
    rows["cumulative_spike_beta05_true"] = rows.spike_beta05_true.cumsum()
    csv_path = _mechanism_path(root, "representative_trajectories", spec, ".csv")
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    rows.to_csv(csv_path, index=False)

    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    axes[0].plot(rows.timestep, rows.evidence_true, label="evidence true")
    axes[0].plot(rows.timestep, rows.evidence_competitor, label="evidence competitor")
    axes[0].legend()
    axes[0].set_ylabel("Evidence")
    axes[1].plot(rows.timestep, rows.mem_beta1_true, label="beta=1 true")
    axes[1].plot(rows.timestep, rows.mem_beta05_true, label="beta=.5 true")
    axes[1].legend()
    axes[1].set_ylabel("Membrane")
    axes[2].plot(rows.timestep, rows.cumulative_evidence_true, label="cum evidence true")
    axes[2].plot(rows.timestep, rows.cumulative_spike_beta1_true, label="cum spike beta=1 true")
    axes[2].plot(rows.timestep, rows.cumulative_spike_beta05_true, label="cum spike beta=.5 true")
    axes[2].legend()
    axes[2].set_xlabel("Timestep")
    axes[2].set_ylabel("Cumulative")
    fig.suptitle(f"{spec.key}; sample={selected}; y={true}; competitor={competitor}")
    fig.tight_layout()
    fig.savefig(_mechanism_path(root, "representative_trajectories", spec, ".png"), dpi=160)
    plt.close(fig)


def run_mechanism(
    spec: SourceSpec,
    data: exp3.Data,
    config: Config,
    force: bool = False,
) -> dict[str, Any]:
    validate_source_spec(spec)
    out_path = _mechanism_path(config.results_dir, "mechanism_evaluations", spec, ".json")
    if out_path.exists() and not force:
        return json.loads(out_path.read_text(encoding="utf-8"))
    cache = _load_cache(spec, config)
    Wlin = _load_head_weight(FrozenHeadSpec(spec.regularization, spec.seed, HEAD_ANALOG), config)
    Wlif = _load_head_weight(FrozenHeadSpec(spec.regularization, spec.seed, HEAD_LIF), config)
    Wnative = _native_weight(spec, data, config)

    val_l2, val_y, val_lengths = cache["val"]
    test_l2, test_y, test_lengths = cache["test"]
    gain, gain_table = calibrate_gain(val_l2, val_y, val_lengths, Wlin)
    gain_path = _mechanism_path(config.results_dir, "gain_calibrations", spec, ".csv")
    gain_path.parent.mkdir(parents=True, exist_ok=True)
    gain_table.to_csv(gain_path, index=False)

    # 2x2 cross-check.
    x1 = _analog_scores(test_l2, test_lengths, Wlin)
    x2_sim = _simulate_unipolar(test_l2, test_lengths, Wlin, beta=0.5, cap=1, scale=gain)
    x3 = _analog_scores(test_l2, test_lengths, Wlif)
    x4_sim = _simulate_unipolar(test_l2, test_lengths, Wlif, beta=0.5, cap=1, scale=1.0)
    crosscheck = {
        "X1_wlin_analog": _metrics(test_y, x1),
        "X2_wlin_lif_beta05": _metrics(test_y, x2_sim["counts"]),
        "X3_wlif_analog": _metrics(test_y, x3),
        "X4_wlif_lif_beta05": _metrics(test_y, x4_sim["counts"]),
    }

    # Direction B: good linear projection -> SNN interface -> leakage.
    b1 = _simulate_unipolar(test_l2, test_lengths, Wlin, beta=1.0, cap=1, scale=gain)
    b2_cap31 = _simulate_unipolar(test_l2, test_lengths, Wlin, beta=1.0, cap=31, scale=gain)
    b2_bipolar = _simulate_bipolar(test_l2, test_lengths, Wlin, beta=1.0, cap=1, scale=gain)
    lif05_cap31 = _simulate_unipolar(test_l2, test_lengths, Wlin, beta=0.5, cap=31, scale=gain)
    lif05_bipolar = _simulate_bipolar(test_l2, test_lengths, Wlin, beta=0.5, cap=1, scale=gain)
    mechanism_ladder = {
        "B0_wlin_analog": _metrics(test_y, x1),
        "B1_if_beta1_charge": _metrics(test_y, b1["charge_scores"]),
        "B2_if_beta1_spike": _metrics(test_y, b1["counts"]),
        "B2a_if_beta1_cap31": _metrics(test_y, b2_cap31["counts"]),
        "B2b_if_beta1_bipolar": _metrics(test_y, b2_bipolar["scores"]),
        "B3_lif_beta05_spike": _metrics(test_y, x2_sim["counts"]),
        "B3a_lif_beta05_cap31": _metrics(test_y, lif05_cap31["counts"]),
        "B3b_lif_beta05_bipolar": _metrics(test_y, lif05_bipolar["scores"]),
    }

    beta_rows: list[dict[str, Any]] = []
    firing: dict[str, dict[str, float]] = {}
    for beta in BETA_SWEEP:
        sim = _simulate_unipolar(test_l2, test_lengths, Wlin, beta=beta, cap=1, scale=gain)
        metric = _metrics(test_y, sim["counts"])
        beta_rows.append({"beta": beta, **metric})
        firing[f"wlin_unipolar_beta_{beta:g}_cap1"] = sim["diagnostics"]

    # Direction A: Wlif reverse validation.
    a1 = _simulate_unipolar(test_l2, test_lengths, Wlif, beta=1.0, cap=1, scale=1.0)
    direction_a = {
        "A0_wlif_lif_beta05": _metrics(test_y, x4_sim["counts"]),
        "A1_wlif_if_beta1_spike": _metrics(test_y, a1["counts"]),
        "A2_wlif_if_beta1_charge": _metrics(test_y, a1["charge_scores"]),
        "A3_wlif_analog": _metrics(test_y, x3),
    }

    native_analog = _analog_scores(test_l2, test_lengths, Wnative)
    native_lif = _simulate_unipolar(test_l2, test_lengths, Wnative, beta=0.5, cap=1, scale=1.0)
    native_control = {
        "native_e2e_W_analog": _metrics(test_y, native_analog),
        "native_e2e_W_lif_beta05": _metrics(test_y, native_lif["counts"]),
    }

    firing.update(
        {
            "wlin_if_beta1_cap31": b2_cap31["diagnostics"],
            "wlin_lif_beta05_cap31": lif05_cap31["diagnostics"],
            "wlif_if_beta1_cap1": a1["diagnostics"],
            "wlif_lif_beta05_cap1": x4_sim["diagnostics"],
            "native_lif_beta05_cap1": native_lif["diagnostics"],
            "wlin_if_beta1_bipolar": b2_bipolar["diagnostics"],
            "wlin_lif_beta05_bipolar": lif05_bipolar["diagnostics"],
        }
    )

    margin = _save_margin_decomposition(spec, test_y, x1, x2_sim, config.results_dir)
    _representative_trajectory(spec, val_l2, val_y, val_lengths, Wlin, gain, config.results_dir)
    accessibility = _standardized_accessibility(cache, spec)

    payload = {
        "experiment_id": EXPERIMENT_ID,
        "protocol_version": PROTOCOL_VERSION,
        "spec": asdict(spec),
        "architecture": ARCHITECTURE,
        "gain": gain,
        "gain_calibration_split": "validation",
        "crosscheck": crosscheck,
        "mechanism_ladder": mechanism_ladder,
        "beta_sweep": beta_rows,
        "direction_a": direction_a,
        "native_control": native_control,
        "firing_diagnostics": firing,
        "standardized_wc_accessibility": accessibility,
        "sanity": {
            "B0_vs_B1_prediction_equal": bool(np.array_equal(x1.argmax(1), b1["charge_scores"].argmax(1))),
            "B0_vs_B1_max_score_relation_error": float(np.max(np.abs(gain * x1 - b1["charge_scores"]))),
            "A2_vs_A3_prediction_equal": bool(np.array_equal(x3.argmax(1), a1["charge_scores"].argmax(1))),
            "A2_vs_A3_max_score_error": float(np.max(np.abs(x3 - a1["charge_scores"]))),
            "margin_failure_samples": int(len(margin)),
        },
    }
    _save_json(out_path, payload)
    return payload


# -----------------------------------------------------------------------------
# Paired E2E representation-shaping models
# -----------------------------------------------------------------------------

class PairedShapingSNN(nn.Module):
    def __init__(self, condition: str, n_classes: int, fs: float) -> None:
        super().__init__()
        if condition not in E2E_TRAIN_CONDITIONS:
            raise ValueError(condition)
        self.condition = condition
        self.n_classes = int(n_classes)
        self.fs = float(fs)
        shifts = ARCHITECTURES[ARCHITECTURE]
        self.hidden_linears = nn.ModuleList(
            [
                nn.Linear(exp72.EXPECTED_CHANNELS, HIDDEN_WIDTH, bias=False),
                nn.Linear(HIDDEN_WIDTH, HIDDEN_WIDTH, bias=False),
            ]
        )
        beta_hidden = math.exp(-(1000.0 / fs) / exp72.TAU_MEM_MS)
        self.hidden_lifs = nn.ModuleList(
            [
                exp401.MacroMultiSpikeLIF(
                    beta=beta_hidden,
                    threshold=THRESHOLD,
                    max_spikes_per_dt=1,
                    surrogate_slope=exp72.SURROGATE_SLOPE,
                )
                for _ in range(2)
            ]
        )
        self.register_buffer("alpha_0", exp50.alpha_vector(HIDDEN_WIDTH, shifts[0]))
        self.register_buffer("alpha_1", exp50.alpha_vector(HIDDEN_WIDTH, shifts[1]))
        self.output_linear = nn.Linear(HIDDEN_WIDTH, self.n_classes, bias=False)
        self.output_lif = (
            exp401.MacroMultiSpikeLIF(
                beta=1.0,
                threshold=THRESHOLD,
                max_spikes_per_dt=1,
                surrogate_slope=exp72.SURROGATE_SLOPE,
            )
            if condition == E2E_IF
            else None
        )

    def forward_trajectory(self, x: torch.Tensor) -> dict[str, Any]:
        batch, steps, channels = x.shape
        if channels != exp72.EXPECTED_CHANNELS:
            raise ValueError(channels)
        syn = [torch.zeros(batch, HIDDEN_WIDTH, device=x.device, dtype=x.dtype) for _ in range(2)]
        mem = [torch.zeros_like(syn[0]), torch.zeros_like(syn[1])]
        hidden: list[list[torch.Tensor]] = [[], []]
        evidence: list[torch.Tensor] = []
        output_spikes: list[torch.Tensor] = []
        out_mem = torch.zeros(batch, self.n_classes, device=x.device, dtype=x.dtype)
        for t in range(steps):
            cur = x[:, t]
            for li in range(2):
                alpha = getattr(self, f"alpha_{li}")
                syn[li] = alpha * syn[li] + self.hidden_linears[li](cur)
                spk, mem[li], _ = self.hidden_lifs[li](syn[li], mem[li])
                hidden[li].append(spk)
                cur = spk
            e = self.output_linear(cur)
            evidence.append(e)
            if self.output_lif is not None:
                spk, out_mem, _ = self.output_lif(e, out_mem)
                output_spikes.append(spk)
        payload: dict[str, Any] = {
            "hidden_spikes": tuple(torch.stack(layer, dim=1) for layer in hidden),
            "output_evidence": torch.stack(evidence, dim=1),
        }
        if output_spikes:
            payload["output_spikes"] = torch.stack(output_spikes, dim=1)
        return payload


def _e2e_loaders(data: exp3.Data, spec: E2ESpec, batch_size: int, shuffle_train: bool) -> dict[str, Any]:
    parts = {
        "train": (data.Xtr, data.ytr, data.ltr),
        "val": (data.Xva, data.yva, data.lva),
        "test": (data.Xte, data.yte, data.lte),
    }
    return {
        split: exp3.loader(
            X,
            y,
            lengths,
            batch_size,
            shuffle_train if split == "train" else False,
            _reference_seed(spec.source, f"{split}_loader"),
        )
        for split, (X, y, lengths) in parts.items()
    }


def _e2e_task_logits(
    model: PairedShapingSNN,
    tr: dict[str, Any],
    lengths: torch.Tensor,
) -> torch.Tensor:
    if model.condition == E2E_ANALOG:
        return _valid_sum(tr["output_evidence"], lengths)
    return exp50.deployment_logits(tr["output_spikes"], lengths, OUTPUT_CAP)


def _reference_calibration(spec: E2ESpec, config: Config) -> dict[str, Any]:
    checkpoint = _exp725_checkpoint_path(spec.source, config.repo_root)
    payload = torch.load(checkpoint, map_location="cpu")
    calibration = dict(payload.get("calibration", {}))
    if spec.regularization == exp72.TASK_ONLY:
        return {
            "calibrated": False,
            "lambda_rate": 0.0,
            "lambda_persist": 0.0,
            "reference": "Exp7.2.5 C2 alpha=0 beta=.5",
        }
    if "lambda_rate" not in calibration or "lambda_persist" not in calibration:
        raise RuntimeError(f"Missing Exp7.2.5 calibration in {checkpoint}")
    calibration["reference"] = "Exp7.2.5 C2 alpha=0 beta=.5"
    return calibration


def _e2e_regularizer(
    spec: E2ESpec,
    trajectory: dict[str, Any],
    lengths: torch.Tensor,
    calibration: dict[str, Any],
    epoch: int,
) -> torch.Tensor:
    if spec.regularization == exp72.TASK_ONLY:
        return torch.zeros((), device=lengths.device)
    rate, _, _, persist = exp72.regularization_terms(trajectory["hidden_spikes"], lengths)
    scale = exp72.warmup_scale(epoch)
    return scale * (
        float(calibration["lambda_rate"]) * rate
        + float(calibration["lambda_persist"]) * persist
    )


def _evaluate_e2e(
    model: PairedShapingSNN,
    loader: Iterable,
    device: torch.device,
) -> dict[str, Any]:
    labels: list[np.ndarray] = []
    preds: list[np.ndarray] = []
    loss_sum = 0.0
    n = 0
    model.eval()
    with torch.no_grad():
        for X, y, lengths in loader:
            X, y, lengths = X.to(device), y.to(device), lengths.to(device)
            tr = model.forward_trajectory(X)
            logits = _e2e_task_logits(model, tr, lengths)
            loss = F.cross_entropy(logits, y)
            labels.append(y.cpu().numpy())
            preds.append(logits.argmax(1).cpu().numpy())
            loss_sum += float(loss) * len(y)
            n += len(y)
    y_true = np.concatenate(labels)
    metrics = exp72._metrics(y_true, np.concatenate(preds))
    return {**metrics, "objective_loss": loss_sum / max(n, 1)}


def run_e2e(
    spec: E2ESpec,
    data: exp3.Data,
    config: Config,
    force: bool = False,
) -> dict[str, Any]:
    validate_e2e_spec(spec, allow_reuse=False)
    eval_path = _e2e_path(config.results_dir, "evaluations", spec, ".json")
    checkpoint_path = _e2e_path(config.results_dir, "checkpoints", spec, ".pt")
    if eval_path.exists() and checkpoint_path.exists() and not force:
        return json.loads(eval_path.read_text(encoding="utf-8"))

    device = torch.device(config.device)
    torch.set_num_threads(config.threads)
    exp3.seed_all(_reference_seed(spec.source, "model_init"))
    model = PairedShapingSNN(spec.condition, len(data.labels), data.fs).to(device)
    calibration = _reference_calibration(spec, config)
    optimizer = torch.optim.Adam(model.parameters(), lr=exp72.LR, weight_decay=exp72.WEIGHT_DECAY)
    train_loader = _e2e_loaders(data, spec, config.batch_size, True)["train"]
    eval_loaders = _e2e_loaders(data, spec, config.batch_size, False)

    best_state: dict[str, torch.Tensor] | None = None
    best_epoch = -1
    best_ba = -1.0
    best_loss = float("inf")
    stopped_epoch = config.max_epochs
    history: list[dict[str, float]] = []
    for epoch in range(1, config.max_epochs + 1):
        model.train()
        task_sum = 0.0
        reg_sum = 0.0
        total_sum = 0.0
        n = 0
        for X, y, lengths in train_loader:
            X, y, lengths = X.to(device), y.to(device), lengths.to(device)
            optimizer.zero_grad(set_to_none=True)
            tr = model.forward_trajectory(X)
            logits = _e2e_task_logits(model, tr, lengths)
            task = F.cross_entropy(logits, y)
            reg = _e2e_regularizer(spec, tr, lengths, calibration, epoch)
            total = task + reg
            total.backward()
            optimizer.step()
            task_sum += float(task.detach()) * len(y)
            reg_sum += float(reg.detach()) * len(y)
            total_sum += float(total.detach()) * len(y)
            n += len(y)
        train_metrics = _evaluate_e2e(model, eval_loaders["train"], device)
        val_metrics = _evaluate_e2e(model, eval_loaders["val"], device)
        history.append(
            {
                "epoch": float(epoch),
                "train_ba": float(train_metrics["balanced_accuracy"]),
                "val_ba": float(val_metrics["balanced_accuracy"]),
                "train_task_loss": task_sum / max(n, 1),
                "train_reg_loss": reg_sum / max(n, 1),
                "train_total_loss": total_sum / max(n, 1),
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
        if _early_stop(epoch, best_epoch):
            stopped_epoch = epoch
            break

    if best_state is None:
        raise RuntimeError(f"No E2E checkpoint selected for {spec.key}")
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
            "model_state_dict": best_state,
            "calibration": calibration,
            "paired_reference_exp725_spec": asdict(spec.source.exp725_spec),
        },
        checkpoint_path,
    )
    model.load_state_dict(best_state, strict=True)
    metrics = {
        split: _evaluate_e2e(model, loader, device)
        for split, loader in eval_loaders.items()
    }
    payload = {
        "experiment_id": EXPERIMENT_ID,
        "protocol_version": PROTOCOL_VERSION,
        "spec": asdict(spec),
        "best_epoch": best_epoch,
        "stopped_epoch": stopped_epoch,
        "calibration": calibration,
        "metrics": metrics,
    }
    _save_json(eval_path, payload)
    hist_path = _e2e_path(config.results_dir, "histories", spec, ".csv")
    hist_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(history).to_csv(hist_path, index=False)
    _plot_history(
        [
            {
                "epoch": r["epoch"],
                "train_ba": r["train_ba"],
                "val_ba": r["val_ba"],
                "train_loss": r["train_task_loss"],
                "val_loss": r["val_loss"],
            }
            for r in history
        ],
        _e2e_path(config.results_dir, "training_curves", spec, ".png"),
        spec.key,
    )
    return payload


def _load_condition_model(
    spec: E2ESpec,
    data: exp3.Data,
    config: Config,
) -> tuple[nn.Module, dict[str, Any]]:
    device = torch.device(config.device)
    if spec.condition == E2E_LIF:
        return _load_exp725_model(spec.source, data, config.repo_root, device)
    path = _e2e_path(config.results_dir, "checkpoints", spec, ".pt")
    if not path.exists():
        raise FileNotFoundError(path)
    payload = torch.load(path, map_location=device)
    model = PairedShapingSNN(spec.condition, len(data.labels), data.fs).to(device)
    model.load_state_dict(payload["model_state_dict"], strict=True)
    model.eval()
    return model, payload


def _extract_l2_generic(model: nn.Module, loader: Iterable, device: torch.device) -> tuple[torch.Tensor, np.ndarray, np.ndarray]:
    xs: list[torch.Tensor] = []
    ys: list[np.ndarray] = []
    ls: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for X, y, lengths in loader:
            tr = model.forward_trajectory(X.to(device))
            xs.append(tr["hidden_spikes"][-1].cpu())
            ys.append(y.numpy())
            ls.append(lengths.numpy())
    return torch.cat(xs, dim=0), np.concatenate(ys), np.concatenate(ls)


def _fit_matched_count_probe(
    splits: dict[str, tuple[torch.Tensor, np.ndarray, np.ndarray]],
    spec: E2ESpec,
    n_classes: int,
) -> dict[str, Any]:
    arrays: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for split, (l2, y, lengths) in splits.items():
        arrays[split] = (_whole_count_np(l2.numpy(), lengths).astype(np.float32), y)
    exp3.seed_all(_probe_pair_seed(spec.source, "matched_wc_init"))
    model = nn.Linear(HIDDEN_WIDTH, n_classes, bias=False)
    optimizer = torch.optim.Adam(model.parameters(), lr=exp72.LR, weight_decay=exp72.WEIGHT_DECAY)
    Xtr = torch.from_numpy(arrays["train"][0])
    ytr = torch.from_numpy(arrays["train"][1])
    generator = torch.Generator().manual_seed(_probe_pair_seed(spec.source, "matched_wc_loader"))
    loader = DataLoader(TensorDataset(Xtr, ytr), batch_size=exp72.BATCH_SIZE, shuffle=True, generator=generator)
    best_state: dict[str, torch.Tensor] | None = None
    best_ba = -1.0
    best_epoch = -1
    best_loss = float("inf")
    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        for X, y in loader:
            optimizer.zero_grad(set_to_none=True)
            loss = F.cross_entropy(model(X), y)
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            val_logits = model(torch.from_numpy(arrays["val"][0]))
            val_loss = float(F.cross_entropy(val_logits, torch.from_numpy(arrays["val"][1])))
            val_pred = val_logits.argmax(1).numpy()
        val_ba = float(exp72._metrics(arrays["val"][1], val_pred)["balanced_accuracy"])
        improved = val_ba > best_ba + 1e-12 or (abs(val_ba - best_ba) <= 1e-12 and val_loss < best_loss)
        if improved:
            best_ba, best_loss, best_epoch = val_ba, val_loss, epoch
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        if _early_stop(epoch, best_epoch):
            break
    if best_state is None:
        raise RuntimeError("No matched count probe checkpoint")
    model.load_state_dict(best_state)
    metrics: dict[str, dict[str, float]] = {}
    with torch.no_grad():
        for split, (X, y) in arrays.items():
            logits = model(torch.from_numpy(X)).numpy()
            metrics[split] = _metrics(y, logits)
    return {
        "source": PROBE_MATCHED_WC,
        "feature_dim": HIDDEN_WIDTH,
        "best_epoch": best_epoch,
        "metrics": metrics,
    }


def run_probe(
    spec: E2ESpec,
    data: exp3.Data,
    config: Config,
    force: bool = False,
) -> dict[str, Any]:
    validate_e2e_spec(spec, allow_reuse=True)
    path = _e2e_path(config.results_dir, "probe_evaluations", spec, ".json")
    if path.exists() and not force:
        return json.loads(path.read_text(encoding="utf-8"))
    device = torch.device(config.device)
    model, checkpoint_payload = _load_condition_model(spec, data, config)
    if spec.condition == E2E_LIF:
        loaders = exp725._e2e_loaders(data, spec.source.exp725_spec, config.batch_size, False)
        source_eval_path = _exp725_evaluation_path(spec.source, config.repo_root)
        if not source_eval_path.exists():
            raise FileNotFoundError(source_eval_path)
        source_eval = json.loads(source_eval_path.read_text(encoding="utf-8"))
        native_metrics = {
            split: result["valid"]
            for split, result in source_eval["metrics"].items()
        }
    else:
        loaders = _e2e_loaders(data, spec, config.batch_size, False)
        eval_path = _e2e_path(config.results_dir, "evaluations", spec, ".json")
        source_eval = json.loads(eval_path.read_text(encoding="utf-8"))
        native_metrics = source_eval["metrics"]
    splits = {
        split: _extract_l2_generic(model, loader, device)
        for split, loader in loaders.items()
    }

    matched = _fit_matched_count_probe(splits, spec, len(data.labels))
    # Reuse the exact Exp7.2.5 standardized probe protocol, with the C2 reference
    # seed for every condition so probe randomness is paired.
    ref_spec = spec.source.exp725_spec
    standardized_wc = exp725._fit_e2e_probe(PROBE_STD_WC, splits, ref_spec, data.bin_steps)
    fixed250 = exp725._fit_e2e_probe(PROBE_F250, splits, ref_spec, data.bin_steps)
    payload = {
        "experiment_id": EXPERIMENT_ID,
        "protocol_version": PROTOCOL_VERSION,
        "spec": asdict(spec),
        "checkpoint_best_epoch": int(checkpoint_payload["best_epoch"]),
        "native_metrics": native_metrics,
        "probes": {
            PROBE_MATCHED_WC: matched,
            PROBE_STD_WC: standardized_wc,
            PROBE_F250: fixed250,
        },
    }
    _save_json(path, payload)
    return payload


# -----------------------------------------------------------------------------
# Final aggregation
# -----------------------------------------------------------------------------

def _flatten_metric_rows(
    spec: SourceSpec,
    mapping: dict[str, dict[str, float]],
    block: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for condition, metrics in mapping.items():
        rows.append(
            {
                "architecture": ARCHITECTURE,
                "regularization": spec.regularization,
                "seed": spec.seed,
                "block": block,
                "condition": condition,
                "accuracy": float(metrics["accuracy"]),
                "balanced_accuracy": float(metrics["balanced_accuracy"]),
                "macro_f1": float(metrics["macro_f1"]),
            }
        )
    return rows


def finalize(repo_root: Path) -> dict[str, Any]:
    root = results_dir(repo_root)
    cross_rows: list[dict[str, Any]] = []
    ladder_rows: list[dict[str, Any]] = []
    beta_rows: list[dict[str, Any]] = []
    firing_rows: list[dict[str, Any]] = []
    source_rows: list[dict[str, Any]] = []
    margin_frames: list[pd.DataFrame] = []
    missing: list[str] = []

    for spec in source_specs():
        path = _mechanism_path(root, "mechanism_evaluations", spec, ".json")
        if not path.exists():
            missing.append(f"mechanism:{spec.key}")
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        cross_rows.extend(_flatten_metric_rows(spec, payload["crosscheck"], "crosscheck"))
        ladder_rows.extend(_flatten_metric_rows(spec, payload["mechanism_ladder"], "direction_b"))
        ladder_rows.extend(_flatten_metric_rows(spec, payload["direction_a"], "direction_a"))
        ladder_rows.extend(_flatten_metric_rows(spec, payload["native_control"], "native_control"))
        for row in payload["beta_sweep"]:
            beta_rows.append({"architecture": ARCHITECTURE, "regularization": spec.regularization, "seed": spec.seed, **row})
        for condition, diag in payload["firing_diagnostics"].items():
            firing_rows.append({"architecture": ARCHITECTURE, "regularization": spec.regularization, "seed": spec.seed, "condition": condition, **diag})
        source_rows.append(
            {
                "architecture": ARCHITECTURE,
                "regularization": spec.regularization,
                "seed": spec.seed,
                "gain": float(payload["gain"]),
                "standardized_wc_test_ba": float(payload["standardized_wc_accessibility"]["metrics"]["test"]["balanced_accuracy"]),
                **payload["sanity"],
            }
        )
        margin_path = _mechanism_path(root, "margin_decomposition", spec, ".csv")
        if margin_path.exists():
            frame = pd.read_csv(margin_path)
            if not frame.empty:
                margin_frames.append(frame)

    probe_rows: list[dict[str, Any]] = []
    e2e_rows: list[dict[str, Any]] = []
    for spec in probe_specs():
        path = _e2e_path(root, "probe_evaluations", spec, ".json")
        if not path.exists():
            missing.append(f"probe:{spec.key}")
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        for split, metrics in payload["native_metrics"].items():
            e2e_rows.append(
                {
                    "architecture": ARCHITECTURE,
                    "regularization": spec.regularization,
                    "seed": spec.seed,
                    "condition": spec.condition,
                    "split": split,
                    "accuracy": float(metrics["accuracy"]),
                    "balanced_accuracy": float(metrics["balanced_accuracy"]),
                    "macro_f1": float(metrics["macro_f1"]),
                }
            )
        for probe_name, probe in payload["probes"].items():
            for split, metrics in probe["metrics"].items():
                probe_rows.append(
                    {
                        "architecture": ARCHITECTURE,
                        "regularization": spec.regularization,
                        "seed": spec.seed,
                        "condition": spec.condition,
                        "probe": probe_name,
                        "split": split,
                        "accuracy": float(metrics["accuracy"]),
                        "balanced_accuracy": float(metrics["balanced_accuracy"]),
                        "macro_f1": float(metrics["macro_f1"]),
                    }
                )

    if missing:
        raise RuntimeError(f"Missing {len(missing)} Exp7.2.6 artifacts; first={missing[:8]}")

    cross = pd.DataFrame(cross_rows)
    cross.to_csv(root / "crosscheck_2x2_runs.csv", index=False)
    _aggregate(cross, ["architecture", "regularization", "condition"], ["accuracy", "balanced_accuracy", "macro_f1"]).to_csv(root / "crosscheck_2x2_summary.csv", index=False)

    ladder = pd.DataFrame(ladder_rows)
    ladder.to_csv(root / "mechanism_ladder_runs.csv", index=False)
    _aggregate(ladder, ["architecture", "regularization", "block", "condition"], ["accuracy", "balanced_accuracy", "macro_f1"]).to_csv(root / "mechanism_ladder_summary.csv", index=False)

    beta = pd.DataFrame(beta_rows)
    beta.to_csv(root / "beta_sweep_runs.csv", index=False)
    _aggregate(beta, ["architecture", "regularization", "beta"], ["accuracy", "balanced_accuracy", "macro_f1"]).to_csv(root / "beta_sweep_summary.csv", index=False)

    firing = pd.DataFrame(firing_rows)
    firing.to_csv(root / "firing_diagnostics.csv", index=False)
    source_manifest = pd.DataFrame(source_rows)
    source_manifest.to_csv(root / "source_manifest.csv", index=False)
    margin = pd.concat(margin_frames, ignore_index=True) if margin_frames else pd.DataFrame()
    margin.to_csv(root / "margin_decomposition.csv", index=False)

    e2e = pd.DataFrame(e2e_rows)
    e2e.to_csv(root / "e2e_performance_runs.csv", index=False)
    _aggregate(e2e, ["architecture", "regularization", "condition", "split"], ["accuracy", "balanced_accuracy", "macro_f1"]).to_csv(root / "e2e_performance_summary.csv", index=False)

    probes = pd.DataFrame(probe_rows)
    probes.to_csv(root / "e2e_l2_probe_runs.csv", index=False)
    _aggregate(probes, ["architecture", "regularization", "condition", "probe", "split"], ["accuracy", "balanced_accuracy", "macro_f1"]).to_csv(root / "e2e_l2_probe_summary.csv", index=False)

    delta_rows: list[dict[str, Any]] = []
    test = probes[probes.split == "test"]
    for (regularization, seed, probe), group in test.groupby(["regularization", "seed", "probe"]):
        indexed = group.set_index("condition")
        for condition in E2E_ALL_CONDITIONS:
            if condition not in indexed.index:
                raise RuntimeError(f"Missing {condition} for {regularization}/{seed}/{probe}")
        values = {c: float(indexed.loc[c, "balanced_accuracy"]) for c in E2E_ALL_CONDITIONS}
        delta_rows.extend(
            [
                {"regularization": regularization, "seed": seed, "probe": probe, "contrast": "analog_minus_if", "delta": values[E2E_ANALOG] - values[E2E_IF]},
                {"regularization": regularization, "seed": seed, "probe": probe, "contrast": "if_minus_lif", "delta": values[E2E_IF] - values[E2E_LIF]},
                {"regularization": regularization, "seed": seed, "probe": probe, "contrast": "analog_minus_lif", "delta": values[E2E_ANALOG] - values[E2E_LIF]},
            ]
        )
    deltas = pd.DataFrame(delta_rows)
    deltas.to_csv(root / "representation_shaping_delta.csv", index=False)
    _aggregate(deltas, ["regularization", "probe", "contrast"], ["delta"]).to_csv(root / "representation_shaping_delta_summary.csv", index=False)

    controls = ladder[ladder.condition.isin(["B2_if_beta1_spike", "B2a_if_beta1_cap31", "B2b_if_beta1_bipolar", "B3_lif_beta05_spike", "B3a_lif_beta05_cap31", "B3b_lif_beta05_bipolar"])].copy()
    controls.to_csv(root / "cap_polarity_controls.csv", index=False)

    report_files: list[str] = []
    for regularization in REGULARIZATIONS:
        for name, frame in (
            ("crosscheck", cross[cross.regularization == regularization]),
            ("mechanism", ladder[ladder.regularization == regularization]),
            ("e2e", probes[(probes.regularization == regularization) & (probes.split == "test")]),
        ):
            out = f"report_{regularization}_{name}.csv"
            frame.to_csv(root / out, index=False)
            report_files.append(out)

    manifest = {
        "experiment_id": EXPERIMENT_ID,
        "protocol_version": PROTOCOL_VERSION,
        "architecture": ARCHITECTURE,
        "shift_profile": [list(x) for x in ARCHITECTURES[ARCHITECTURE]],
        "regularizations": list(REGULARIZATIONS),
        "seeds": list(SEEDS),
        "objective": OBJECTIVE,
        "source_condition": {"output_alpha": SOURCE_ALPHA, "output_beta": SOURCE_BETA},
        "source_cache_tasks": len(source_specs()),
        "frozen_head_training_runs": len(frozen_head_specs()),
        "mechanism_tasks": len(source_specs()),
        "new_e2e_training_runs": len(e2e_train_specs()),
        "probe_tasks_including_reused_c2": len(probe_specs()),
        "checkpoint_selection": "validation valid-length WholeCount BA; validation loss tie-break",
        "gain_calibration": "validation-only beta=1 cap=1 IF; fixed thereafter",
        "e2e_pairing": "C0/C1 reuse exact Exp7.2.5 C2 model-init and loader seed stream",
        "notebook_inputs": [
            "crosscheck_2x2_summary.csv",
            "mechanism_ladder_summary.csv",
            "beta_sweep_summary.csv",
            "cap_polarity_controls.csv",
            "e2e_performance_summary.csv",
            "e2e_l2_probe_summary.csv",
            "representation_shaping_delta_summary.csv",
            *report_files,
        ],
    }
    _save_json(root / "manifest.json", manifest)
    return manifest


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def _select_source(index: int | None, args: argparse.Namespace) -> SourceSpec:
    specs = source_specs()
    if index is not None:
        if not 0 <= index < len(specs):
            raise ValueError(index)
        return specs[index]
    if args.regularization is None or args.seed is None:
        raise ValueError("Specify --array-task-id or --regularization and --seed")
    return SourceSpec(args.regularization, args.seed)


def _select_head(index: int | None, args: argparse.Namespace) -> FrozenHeadSpec:
    specs = frozen_head_specs()
    if index is not None:
        if not 0 <= index < len(specs):
            raise ValueError(index)
        return specs[index]
    if args.regularization is None or args.seed is None or args.mode is None:
        raise ValueError("Specify --array-task-id or all frozen-head fields")
    return FrozenHeadSpec(args.regularization, args.seed, args.mode)


def _select_e2e(index: int | None, args: argparse.Namespace, probes: bool) -> E2ESpec:
    specs = probe_specs() if probes else e2e_train_specs()
    if index is not None:
        if not 0 <= index < len(specs):
            raise ValueError(index)
        return specs[index]
    allowed = E2E_ALL_CONDITIONS if probes else E2E_TRAIN_CONDITIONS
    if args.regularization is None or args.seed is None or args.condition is None:
        raise ValueError("Specify --array-task-id or all E2E fields")
    if args.condition not in allowed:
        raise ValueError(args.condition)
    return E2ESpec(args.regularization, args.seed, args.condition)


def _add_source_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--array-task-id", type=int)
    parser.add_argument("--regularization", choices=REGULARIZATIONS)
    parser.add_argument("--seed", type=int, choices=SEEDS)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=exp72.BATCH_SIZE)
    parser.add_argument("--max-epochs", type=int, default=MAX_EPOCHS)
    sub = parser.add_subparsers(dest="command", required=True)

    prep = sub.add_parser("prepare-frozen")
    _add_source_args(prep)
    prep.add_argument("--force", action="store_true")

    head = sub.add_parser("run-frozen-head")
    _add_source_args(head)
    head.add_argument("--mode", choices=HEAD_MODES)
    head.add_argument("--force", action="store_true")

    mechanism = sub.add_parser("run-mechanism")
    _add_source_args(mechanism)
    mechanism.add_argument("--force", action="store_true")

    e2e = sub.add_parser("run-e2e")
    _add_source_args(e2e)
    e2e.add_argument("--condition", choices=E2E_TRAIN_CONDITIONS)
    e2e.add_argument("--force", action="store_true")

    probe = sub.add_parser("run-probe")
    _add_source_args(probe)
    probe.add_argument("--condition", choices=E2E_ALL_CONDITIONS)
    probe.add_argument("--force", action="store_true")

    for command in ("list-source", "list-head", "list-e2e", "list-probe", "finalize"):
        sub.add_parser(command)
    return parser


def main() -> None:
    args = _parser().parse_args()
    repo_root = find_repo_root(args.repo_root)
    config = Config(repo_root, results_dir(repo_root), args.device, args.batch_size, args.threads, args.max_epochs)

    if args.command == "list-source":
        for i, spec in enumerate(source_specs()):
            print(i, spec.key)
        return
    if args.command == "list-head":
        for i, spec in enumerate(frozen_head_specs()):
            print(i, spec.key)
        return
    if args.command == "list-e2e":
        for i, spec in enumerate(e2e_train_specs()):
            print(i, spec.key)
        return
    if args.command == "list-probe":
        for i, spec in enumerate(probe_specs()):
            print(i, spec.key)
        return
    if args.command == "finalize":
        print(json.dumps(finalize(repo_root), indent=2))
        return

    data = exp72.prepare_data(repo_root)
    if args.command == "prepare-frozen":
        spec = _select_source(args.array_task_id, args)
        path = prepare_frozen_cache(spec, data, config, args.force)
        print(json.dumps({"spec": asdict(spec), "cache": str(path)}, indent=2))
        return
    if args.command == "run-frozen-head":
        spec = _select_head(args.array_task_id, args)
        payload = run_frozen_head(spec, data, config, args.force)
        print(json.dumps({"spec": payload["spec"], "best_epoch": payload["best_epoch"]}, indent=2))
        return
    if args.command == "run-mechanism":
        spec = _select_source(args.array_task_id, args)
        payload = run_mechanism(spec, data, config, args.force)
        print(json.dumps({"spec": payload["spec"], "gain": payload["gain"]}, indent=2))
        return
    if args.command == "run-e2e":
        spec = _select_e2e(args.array_task_id, args, probes=False)
        payload = run_e2e(spec, data, config, args.force)
        print(json.dumps({"spec": payload["spec"], "best_epoch": payload["best_epoch"]}, indent=2))
        return
    if args.command == "run-probe":
        spec = _select_e2e(args.array_task_id, args, probes=True)
        payload = run_probe(spec, data, config, args.force)
        print(json.dumps({"spec": payload["spec"], "checkpoint_best_epoch": payload["checkpoint_best_epoch"]}, indent=2))
        return
    raise ValueError(args.command)


if __name__ == "__main__":
    main()
