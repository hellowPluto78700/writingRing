from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
from typing import Any, Iterable

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
from scripts import experiment_7_2_6_output_readout_loss_shaping as exp726


EXPERIMENT_ID = "experiment_7_3_training_strategy_decomposition"
PROTOCOL_VERSION = "training_strategy_decomposition_v1"
ARCHITECTURE = "234x234"
SHIFTS = ((2, 3, 4), (2, 3, 4))
REGULARIZATION = exp72.TASK_ONLY
SEEDS = (11, 23, 37)
HIDDEN_WIDTH = exp72.HIDDEN_WIDTH
READOUTS = ("linear", "lif")
OBJECTIVES = ("tsce", "wcce")
LIF_BETA = 0.5
THRESHOLD = float(exp72.THRESHOLD)
OUTPUT_CAP = 1
CE_GAIN = 1.0
MAX_EPOCHS = 100
MIN_EPOCHS = 20
PATIENCE = 30
PROBE_SOURCES = ("l2_wholecount_linear", "l2_fixed250_linear")

E2E_CASES = (
    ("A1_e2e_linear_tsce", "linear", "tsce"),
    ("A2_e2e_linear_wcce", "linear", "wcce"),
    ("A3_e2e_lif_tsce", "lif", "tsce"),
    ("A4_e2e_lif_wcce", "lif", "wcce"),
)

STAGE2_CASES = (
    ("B1_tsbackbone_linear_tsce", "tsce", "linear", "tsce"),
    ("B2_tsbackbone_linear_wcce", "tsce", "linear", "wcce"),
    ("B3_tsbackbone_lif_tsce", "tsce", "lif", "tsce"),
    ("B4_tsbackbone_lif_wcce", "tsce", "lif", "wcce"),
    ("B5_wcbackbone_linear_tsce", "wcce", "linear", "tsce"),
    ("B6_wcbackbone_linear_wcce", "wcce", "linear", "wcce"),
    ("B7_wcbackbone_lif_tsce", "wcce", "lif", "tsce"),
    ("B8_wcbackbone_lif_wcce", "wcce", "lif", "wcce"),
)


@dataclass(frozen=True)
class E2ESpec:
    seed: int
    readout: str
    objective: str

    @property
    def method(self) -> str:
        for name, readout, objective in E2E_CASES:
            if (self.readout, self.objective) == (readout, objective):
                return name
        raise ValueError((self.readout, self.objective))

    @property
    def key(self) -> str:
        return f"{ARCHITECTURE}__{self.method}__{REGULARIZATION}__seed{self.seed}"


@dataclass(frozen=True)
class BackboneSpec:
    seed: int
    objective: str

    @property
    def source(self) -> E2ESpec:
        return E2ESpec(self.seed, "linear", self.objective)

    @property
    def key(self) -> str:
        return f"{ARCHITECTURE}__backbone_{self.objective}__{REGULARIZATION}__seed{self.seed}"


@dataclass(frozen=True)
class Stage2Spec:
    seed: int
    backbone_objective: str
    readout: str
    objective: str

    @property
    def method(self) -> str:
        for name, backbone_objective, readout, objective in STAGE2_CASES:
            if (self.backbone_objective, self.readout, self.objective) == (
                backbone_objective,
                readout,
                objective,
            ):
                return name
        raise ValueError((self.backbone_objective, self.readout, self.objective))

    @property
    def backbone(self) -> BackboneSpec:
        return BackboneSpec(self.seed, self.backbone_objective)

    @property
    def key(self) -> str:
        return f"{ARCHITECTURE}__{self.method}__{REGULARIZATION}__seed{self.seed}"


@dataclass(frozen=True)
class Config:
    repo_root: Path
    results_dir: Path
    device: str = "cpu"
    batch_size: int = exp72.BATCH_SIZE
    threads: int = 1
    max_epochs: int = MAX_EPOCHS


def find_repo_root(start: Path | None = None) -> Path:
    return exp72.find_repo_root(start)


def results_dir(repo_root: Path) -> Path:
    return repo_root / "notebooks" / "artifacts" / EXPERIMENT_ID / PROTOCOL_VERSION


def e2e_specs() -> list[E2ESpec]:
    return [
        E2ESpec(seed, readout, objective)
        for seed in SEEDS
        for _, readout, objective in E2E_CASES
    ]


def backbone_specs() -> list[BackboneSpec]:
    return [BackboneSpec(seed, objective) for seed in SEEDS for objective in OBJECTIVES]


def stage2_specs() -> list[Stage2Spec]:
    return [
        Stage2Spec(seed, backbone_objective, readout, objective)
        for seed in SEEDS
        for _, backbone_objective, readout, objective in STAGE2_CASES
    ]


def validate_e2e(spec: E2ESpec) -> None:
    if spec.seed not in SEEDS:
        raise ValueError(spec.seed)
    if spec.readout not in READOUTS:
        raise ValueError(spec.readout)
    if spec.objective not in OBJECTIVES:
        raise ValueError(spec.objective)
    _ = spec.method


def validate_backbone(spec: BackboneSpec) -> None:
    if spec.seed not in SEEDS:
        raise ValueError(spec.seed)
    if spec.objective not in OBJECTIVES:
        raise ValueError(spec.objective)


def validate_stage2(spec: Stage2Spec) -> None:
    if spec.seed not in SEEDS:
        raise ValueError(spec.seed)
    if spec.backbone_objective not in OBJECTIVES:
        raise ValueError(spec.backbone_objective)
    if spec.readout not in READOUTS:
        raise ValueError(spec.readout)
    if spec.objective not in OBJECTIVES:
        raise ValueError(spec.objective)
    _ = spec.method


def _save_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _path(root: Path, kind: str, key: str, suffix: str) -> Path:
    return root / kind / f"{key}{suffix}"


def _e2e_pair_seed(seed: int, role: str) -> int:
    return exp3.dseed(seed, EXPERIMENT_ID, "e2e_paired", role)


def _stage2_pair_seed(seed: int, role: str) -> int:
    return exp3.dseed(seed, EXPERIMENT_ID, "stage2_paired", role)


def _probe_seed(seed: int, source: str) -> int:
    return exp3.dseed(seed, EXPERIMENT_ID, "paired_probe", source)


def _raw_loaders(
    data: exp3.Data,
    seed: int,
    batch_size: int,
    shuffle_train: bool,
) -> dict[str, Any]:
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
            _e2e_pair_seed(seed, f"{split}_loader"),
        )
        for split, (X, y, lengths) in parts.items()
    }


def _cached_loader(
    split: tuple[np.ndarray, np.ndarray, np.ndarray],
    batch_size: int,
    shuffle: bool,
    seed: int,
) -> DataLoader:
    l2, y, lengths = split
    generator = torch.Generator().manual_seed(int(seed))
    return DataLoader(
        TensorDataset(
            torch.from_numpy(l2),
            torch.from_numpy(y),
            torch.from_numpy(lengths),
        ),
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator,
        num_workers=0,
    )


def _cached_loaders(
    cache: dict[str, Any],
    seed: int,
    batch_size: int,
    shuffle_train: bool,
) -> dict[str, DataLoader]:
    return {
        split: _cached_loader(
            cache[split],
            batch_size,
            shuffle_train if split == "train" else False,
            _stage2_pair_seed(seed, f"{split}_loader"),
        )
        for split in ("train", "val", "test")
    }


def _valid_mask(lengths: torch.Tensor, n_steps: int) -> torch.Tensor:
    return torch.arange(n_steps, device=lengths.device)[None, :] < lengths[:, None]


def _valid_mean(values: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
    mask = _valid_mask(lengths, values.shape[1]).to(values.dtype).unsqueeze(-1)
    summed = (values * mask).sum(dim=1)
    return summed / lengths.clamp_min(1).to(values.dtype).unsqueeze(1)


def _objective_loss_scores(
    values: torch.Tensor,
    lengths: torch.Tensor,
    y: torch.Tensor,
    objective: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    scores = _valid_mean(values, lengths)
    if objective == "wcce":
        return F.cross_entropy(CE_GAIN * scores, y), scores
    if objective == "tsce":
        valid = _valid_mask(lengths, values.shape[1])
        targets = y[:, None].expand(len(y), values.shape[1])
        return F.cross_entropy((CE_GAIN * values)[valid], targets[valid]), scores
    raise ValueError(objective)


class Exp73Net(nn.Module):
    def __init__(self, readout: str, n_classes: int, fs: float) -> None:
        super().__init__()
        if readout not in READOUTS:
            raise ValueError(readout)
        self.readout = readout
        self.n_classes = int(n_classes)
        self.fs = float(fs)
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
        self.register_buffer("alpha_0", exp50.alpha_vector(HIDDEN_WIDTH, SHIFTS[0]))
        self.register_buffer("alpha_1", exp50.alpha_vector(HIDDEN_WIDTH, SHIFTS[1]))
        self.output_linear = nn.Linear(HIDDEN_WIDTH, n_classes, bias=False)
        self.output_lif = (
            exp401.MacroMultiSpikeLIF(
                beta=LIF_BETA,
                threshold=THRESHOLD,
                max_spikes_per_dt=OUTPUT_CAP,
                surrogate_slope=exp72.SURROGATE_SLOPE,
            )
            if readout == "lif"
            else None
        )

    def forward_trajectory(self, x: torch.Tensor) -> dict[str, Any]:
        batch, steps, channels = x.shape
        if channels != exp72.EXPECTED_CHANNELS:
            raise ValueError(channels)
        syn = [
            torch.zeros(batch, HIDDEN_WIDTH, device=x.device, dtype=x.dtype)
            for _ in range(2)
        ]
        mem = [torch.zeros_like(syn[0]), torch.zeros_like(syn[1])]
        hidden: list[list[torch.Tensor]] = [[], []]
        evidence_steps: list[torch.Tensor] = []
        output_spikes: list[torch.Tensor] = []
        out_mem = torch.zeros(
            batch, self.n_classes, device=x.device, dtype=x.dtype
        )
        for t in range(steps):
            cur = x[:, t]
            for li in range(2):
                alpha = getattr(self, f"alpha_{li}")
                syn[li] = alpha * syn[li] + self.hidden_linears[li](cur)
                spk, mem[li], _ = self.hidden_lifs[li](syn[li], mem[li])
                hidden[li].append(spk)
                cur = spk
            evidence = self.output_linear(cur)
            evidence_steps.append(evidence)
            if self.output_lif is not None:
                spk, out_mem, _ = self.output_lif(evidence, out_mem)
                output_spikes.append(spk)
        payload: dict[str, Any] = {
            "hidden_spikes": tuple(torch.stack(v, dim=1) for v in hidden),
            "evidence": torch.stack(evidence_steps, dim=1),
        }
        if output_spikes:
            payload["output_spikes"] = torch.stack(output_spikes, dim=1)
        return payload


class Stage2Head(nn.Module):
    def __init__(self, readout: str, n_classes: int) -> None:
        super().__init__()
        if readout not in READOUTS:
            raise ValueError(readout)
        self.readout = readout
        self.output_linear = nn.Linear(HIDDEN_WIDTH, n_classes, bias=False)
        self.output_lif = (
            exp401.MacroMultiSpikeLIF(
                beta=LIF_BETA,
                threshold=THRESHOLD,
                max_spikes_per_dt=OUTPUT_CAP,
                surrogate_slope=exp72.SURROGATE_SLOPE,
            )
            if readout == "lif"
            else None
        )

    def forward_trajectory(self, l2: torch.Tensor) -> dict[str, torch.Tensor]:
        evidence = self.output_linear(l2)
        payload = {"evidence": evidence}
        if self.output_lif is None:
            return payload
        batch, steps, n_classes = evidence.shape
        mem = torch.zeros(batch, n_classes, device=evidence.device, dtype=evidence.dtype)
        spikes: list[torch.Tensor] = []
        for t in range(steps):
            spk, mem, _ = self.output_lif(evidence[:, t], mem)
            spikes.append(spk)
        payload["output_spikes"] = torch.stack(spikes, dim=1)
        return payload


def _trajectory_values(readout: str, trajectory: dict[str, Any]) -> torch.Tensor:
    return trajectory["evidence"] if readout == "linear" else trajectory["output_spikes"]


def _metrics(y: np.ndarray, scores: np.ndarray) -> dict[str, float]:
    return exp72._metrics(y, scores.argmax(axis=1))


def _evaluate_e2e(
    spec: E2ESpec,
    model: Exp73Net,
    loader: Iterable,
    device: torch.device,
) -> dict[str, float]:
    ys: list[np.ndarray] = []
    preds: list[np.ndarray] = []
    loss_sum = 0.0
    n_total = 0
    model.eval()
    with torch.no_grad():
        for X, y, lengths in loader:
            X = X.to(device)
            y = y.to(device)
            lengths = lengths.to(device)
            tr = model.forward_trajectory(X)
            loss, scores = _objective_loss_scores(
                _trajectory_values(spec.readout, tr), lengths, y, spec.objective
            )
            ys.append(y.cpu().numpy())
            preds.append(scores.argmax(1).cpu().numpy())
            loss_sum += float(loss) * len(y)
            n_total += len(y)
    out = exp72._metrics(np.concatenate(ys), np.concatenate(preds))
    out["objective_loss"] = loss_sum / max(n_total, 1)
    return out


def _evaluate_stage2_native(
    spec: Stage2Spec,
    model: Stage2Head,
    loader: Iterable,
    device: torch.device,
) -> dict[str, float]:
    ys: list[np.ndarray] = []
    preds: list[np.ndarray] = []
    loss_sum = 0.0
    n_total = 0
    model.eval()
    with torch.no_grad():
        for l2, y, lengths in loader:
            l2 = l2.to(device=device, dtype=torch.float32)
            y = y.to(device)
            lengths = lengths.to(device)
            tr = model.forward_trajectory(l2)
            loss, scores = _objective_loss_scores(
                _trajectory_values(spec.readout, tr), lengths, y, spec.objective
            )
            ys.append(y.cpu().numpy())
            preds.append(scores.argmax(1).cpu().numpy())
            loss_sum += float(loss) * len(y)
            n_total += len(y)
    out = exp72._metrics(np.concatenate(ys), np.concatenate(preds))
    out["objective_loss"] = loss_sum / max(n_total, 1)
    return out


def _checkpoint_improved(
    metrics: dict[str, float], best_ba: float, best_loss: float
) -> bool:
    return metrics["balanced_accuracy"] > best_ba + 1e-12 or (
        abs(metrics["balanced_accuracy"] - best_ba) <= 1e-12
        and metrics["objective_loss"] < best_loss
    )


def _extract_l2(
    model: Exp73Net,
    loader: Iterable,
    device: torch.device,
) -> tuple[torch.Tensor, np.ndarray, np.ndarray]:
    xs: list[torch.Tensor] = []
    ys: list[np.ndarray] = []
    lengths_all: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for X, y, lengths in loader:
            tr = model.forward_trajectory(X.to(device))
            xs.append(tr["hidden_spikes"][-1].cpu())
            ys.append(y.numpy())
            lengths_all.append(lengths.numpy())
    return torch.cat(xs, dim=0), np.concatenate(ys), np.concatenate(lengths_all)


def _probe_features(
    l2: torch.Tensor, lengths: np.ndarray, bin_steps: int, source: str
) -> np.ndarray:
    lt = torch.tensor(lengths, dtype=torch.long)
    if source == "l2_wholecount_linear":
        mask = exp50.valid_mask(lt, l2.shape[1]).to(l2.dtype).unsqueeze(-1)
        return (l2 * mask).sum(dim=1).numpy()
    if source == "l2_fixed250_linear":
        return exp3.fixed_counts(l2, lt, bin_steps).flatten(1).numpy()
    raise ValueError(source)


def _fit_probes(
    splits: dict[str, tuple[torch.Tensor, np.ndarray, np.ndarray]],
    seed: int,
    bin_steps: int,
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for source in PROBE_SOURCES:
        features = {
            split: (_probe_features(l2, lengths, bin_steps, source), y)
            for split, (l2, y, lengths) in splits.items()
        }
        probe = exp01._fit_linear_probe(
            features["train"][0],
            features["train"][1],
            features["val"][0],
            features["val"][1],
            features["test"][0],
            features["test"][1],
            _probe_seed(seed, source),
        )
        result[source] = {
            "feature_dim": int(probe["feature_dim"]),
            "probe_C": float(probe["probe_C"]),
            "metrics": {split: probe[split] for split in ("train", "val", "test")},
        }
    return result


def _numpy_splits(
    splits: dict[str, tuple[torch.Tensor, np.ndarray, np.ndarray]]
) -> dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    return {
        split: (l2.numpy().astype(np.uint8), y, lengths)
        for split, (l2, y, lengths) in splits.items()
    }


def _cross_evaluate_w(
    splits: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]],
    W: np.ndarray,
) -> dict[str, dict[str, dict[str, float]]]:
    result: dict[str, dict[str, dict[str, float]]] = {}
    for split, (l2, y, lengths) in splits.items():
        analog = exp726._analog_scores(l2, lengths, W, scale=1.0)
        lif = exp726._simulate_unipolar(
            l2,
            lengths,
            W,
            beta=LIF_BETA,
            cap=OUTPUT_CAP,
            scale=1.0,
        )["counts"]
        result[split] = {
            "analog": _metrics(y, analog),
            "lif_beta05": _metrics(y, lif),
        }
    return result


def _load_e2e_model(
    spec: E2ESpec, data: exp3.Data, config: Config
) -> tuple[Exp73Net, dict[str, Any]]:
    checkpoint_path = _path(config.results_dir, "e2e_checkpoints", spec.key, ".pt")
    if not checkpoint_path.exists():
        raise FileNotFoundError(checkpoint_path)
    payload = torch.load(checkpoint_path, map_location=config.device, weights_only=False)
    model = Exp73Net(spec.readout, len(data.labels), data.fs).to(config.device)
    model.load_state_dict(payload["model_state_dict"], strict=True)
    model.eval()
    return model, payload


def run_e2e(spec: E2ESpec, config: Config, force: bool = False) -> dict[str, Any]:
    validate_e2e(spec)
    eval_path = _path(config.results_dir, "e2e_evaluations", spec.key, ".json")
    checkpoint_path = _path(config.results_dir, "e2e_checkpoints", spec.key, ".pt")
    if eval_path.exists() and checkpoint_path.exists() and not force:
        return json.loads(eval_path.read_text(encoding="utf-8"))

    data = exp3.prepare_data(config.repo_root)
    device = torch.device(config.device)
    torch.set_num_threads(config.threads)
    exp3.seed_all(_e2e_pair_seed(spec.seed, "model_init"))
    model = Exp73Net(spec.readout, len(data.labels), data.fs).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=exp72.LR, weight_decay=exp72.WEIGHT_DECAY
    )
    train_loader = _raw_loaders(data, spec.seed, config.batch_size, True)["train"]
    eval_loaders = _raw_loaders(data, spec.seed, config.batch_size, False)

    best_state: dict[str, torch.Tensor] | None = None
    best_epoch = -1
    best_ba = -1.0
    best_loss = float("inf")
    stopped_epoch = config.max_epochs
    history: list[dict[str, float]] = []

    for epoch in range(1, config.max_epochs + 1):
        model.train()
        train_loss_sum = 0.0
        n_total = 0
        for X, y, lengths in train_loader:
            X = X.to(device)
            y = y.to(device)
            lengths = lengths.to(device)
            optimizer.zero_grad(set_to_none=True)
            tr = model.forward_trajectory(X)
            loss, _ = _objective_loss_scores(
                _trajectory_values(spec.readout, tr), lengths, y, spec.objective
            )
            loss.backward()
            optimizer.step()
            train_loss_sum += float(loss.detach()) * len(y)
            n_total += len(y)

        train_metrics = _evaluate_e2e(spec, model, eval_loaders["train"], device)
        val_metrics = _evaluate_e2e(spec, model, eval_loaders["val"], device)
        history.append(
            {
                "epoch": float(epoch),
                "train_ba": float(train_metrics["balanced_accuracy"]),
                "val_ba": float(val_metrics["balanced_accuracy"]),
                "train_loss": train_loss_sum / max(n_total, 1),
                "val_loss": float(val_metrics["objective_loss"]),
            }
        )
        if _checkpoint_improved(val_metrics, best_ba, best_loss):
            best_ba = float(val_metrics["balanced_accuracy"])
            best_loss = float(val_metrics["objective_loss"])
            best_epoch = epoch
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
        if epoch >= MIN_EPOCHS and best_epoch > 0 and epoch - best_epoch >= PATIENCE:
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
            "pairing": "objective/readout excluded from E2E model-init and loader seeds",
            "ce_gain": CE_GAIN,
        },
        checkpoint_path,
    )
    model.load_state_dict(best_state, strict=True)
    native_metrics = {
        split: _evaluate_e2e(spec, model, loader, device)
        for split, loader in eval_loaders.items()
    }
    l2_splits = {
        split: _extract_l2(model, loader, device)
        for split, loader in eval_loaders.items()
    }
    probes = _fit_probes(l2_splits, spec.seed, data.bin_steps)
    numpy_splits = _numpy_splits(l2_splits)
    W = model.output_linear.weight.detach().cpu().numpy().astype(np.float64)
    cross = _cross_evaluate_w(numpy_splits, W)

    lif_reference_equal: bool | None = None
    if spec.readout == "lif":
        lif_reference_equal = (
            native_metrics["test"]["balanced_accuracy"]
            == cross["test"]["lif_beta05"]["balanced_accuracy"]
        )

    payload = {
        "experiment_id": EXPERIMENT_ID,
        "protocol_version": PROTOCOL_VERSION,
        "group": "e2e",
        "method": spec.method,
        "spec": asdict(spec),
        "contract": {
            "architecture": ARCHITECTURE,
            "shifts": [list(v) for v in SHIFTS],
            "regularization": REGULARIZATION,
            "bias": False,
            "ce_gain": CE_GAIN,
            "lif_beta": LIF_BETA,
            "threshold": THRESHOLD,
            "output_cap": OUTPUT_CAP,
            "all_l1_l2_w_trainable": True,
            "checkpoint_metric": "native_validation_balanced_accuracy",
            "final_primary_readout": "lif_beta05_spike_count",
        },
        "best_epoch": best_epoch,
        "stopped_epoch": stopped_epoch,
        "native_metrics": native_metrics,
        "cross_evaluation": cross,
        "representation_probes": probes,
        "lif_native_reference_ba_equal": lif_reference_equal,
    }
    _save_json(eval_path, payload)
    history_path = _path(config.results_dir, "e2e_histories", spec.key, ".csv")
    history_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(history).to_csv(history_path, index=False)
    return payload


def _stage2_cache_path(config: Config, spec: BackboneSpec) -> Path:
    return _path(config.results_dir, "stage2_l2_cache", spec.key, ".npz")


def _stage2_cache_meta_path(config: Config, spec: BackboneSpec) -> Path:
    return _path(config.results_dir, "stage2_l2_cache", spec.key, ".json")


def prepare_stage2_cache(
    spec: BackboneSpec, config: Config, force: bool = False
) -> Path:
    validate_backbone(spec)
    destination = _stage2_cache_path(config, spec)
    metadata_path = _stage2_cache_meta_path(config, spec)
    if destination.exists() and metadata_path.exists() and not force:
        return destination

    data = exp3.prepare_data(config.repo_root)
    source_model, checkpoint = _load_e2e_model(spec.source, data, config)
    if spec.source.readout != "linear":
        raise RuntimeError("Stage-1 backbone source must use the temporary Linear head")
    device = torch.device(config.device)
    torch.set_num_threads(config.threads)
    loaders = _raw_loaders(data, spec.seed, config.batch_size, False)
    splits = {
        split: _extract_l2(source_model, loader, device)
        for split, loader in loaders.items()
    }
    probes = _fit_probes(splits, spec.seed, data.bin_steps)

    arrays: dict[str, np.ndarray] = {}
    split_meta: dict[str, Any] = {}
    for split, (l2, y, lengths) in splits.items():
        arr = l2.numpy()
        if not np.all((arr == 0) | (arr == 1)):
            raise RuntimeError(f"{spec.key}/{split}: L2 cache must be binary")
        arrays[f"{split}_l2"] = arr.astype(np.uint8, copy=False)
        arrays[f"{split}_y"] = y.astype(np.int64, copy=False)
        arrays[f"{split}_lengths"] = lengths.astype(np.int64, copy=False)
        split_meta[split] = {
            "shape": list(arr.shape),
            "n_samples": int(len(y)),
            "mean_firing_fraction_full": float(arr.mean()),
            "mean_valid_length": float(lengths.mean()),
        }

    destination.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(destination, **arrays)
    _save_json(
        metadata_path,
        {
            "experiment_id": EXPERIMENT_ID,
            "protocol_version": PROTOCOL_VERSION,
            "spec": asdict(spec),
            "source_e2e_spec": asdict(spec.source),
            "source_method": spec.source.method,
            "source_best_epoch": int(checkpoint["best_epoch"]),
            "source_checkpoint": str(
                _path(config.results_dir, "e2e_checkpoints", spec.source.key, ".pt").relative_to(
                    config.repo_root
                )
            ),
            "frozen_l1_l2": True,
            "discard_stage1_w": True,
            "representation_probes": probes,
            "splits": split_meta,
        },
    )
    return destination


def _load_stage2_cache(spec: BackboneSpec, config: Config) -> dict[str, Any]:
    path = _stage2_cache_path(config, spec)
    meta_path = _stage2_cache_meta_path(config, spec)
    if not path.exists() or not meta_path.exists():
        raise FileNotFoundError(f"Missing stage-2 cache for {spec.key}")
    with np.load(path, allow_pickle=False) as z:
        result: dict[str, Any] = {
            split: (
                z[f"{split}_l2"].copy(),
                z[f"{split}_y"].copy(),
                z[f"{split}_lengths"].copy(),
            )
            for split in ("train", "val", "test")
        }
    result["metadata"] = json.loads(meta_path.read_text(encoding="utf-8"))
    return result


def run_stage2(
    spec: Stage2Spec, config: Config, force: bool = False
) -> dict[str, Any]:
    validate_stage2(spec)
    eval_path = _path(config.results_dir, "stage2_evaluations", spec.key, ".json")
    checkpoint_path = _path(config.results_dir, "stage2_checkpoints", spec.key, ".pt")
    if eval_path.exists() and checkpoint_path.exists() and not force:
        return json.loads(eval_path.read_text(encoding="utf-8"))

    cache = _load_stage2_cache(spec.backbone, config)
    n_classes = int(np.max(cache["train"][1])) + 1
    device = torch.device(config.device)
    torch.set_num_threads(config.threads)
    exp3.seed_all(_stage2_pair_seed(spec.seed, "model_init"))
    model = Stage2Head(spec.readout, n_classes).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=exp72.LR, weight_decay=exp72.WEIGHT_DECAY
    )
    train_loader = _cached_loaders(
        cache, spec.seed, config.batch_size, True
    )["train"]
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
        n_total = 0
        for l2, y, lengths in train_loader:
            l2 = l2.to(device=device, dtype=torch.float32)
            y = y.to(device)
            lengths = lengths.to(device)
            optimizer.zero_grad(set_to_none=True)
            tr = model.forward_trajectory(l2)
            loss, _ = _objective_loss_scores(
                _trajectory_values(spec.readout, tr), lengths, y, spec.objective
            )
            loss.backward()
            optimizer.step()
            train_loss_sum += float(loss.detach()) * len(y)
            n_total += len(y)

        train_metrics = _evaluate_stage2_native(
            spec, model, eval_loaders["train"], device
        )
        val_metrics = _evaluate_stage2_native(spec, model, eval_loaders["val"], device)
        history.append(
            {
                "epoch": float(epoch),
                "train_ba": float(train_metrics["balanced_accuracy"]),
                "val_ba": float(val_metrics["balanced_accuracy"]),
                "train_loss": train_loss_sum / max(n_total, 1),
                "val_loss": float(val_metrics["objective_loss"]),
            }
        )
        if _checkpoint_improved(val_metrics, best_ba, best_loss):
            best_ba = float(val_metrics["balanced_accuracy"])
            best_loss = float(val_metrics["objective_loss"])
            best_epoch = epoch
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
        if epoch >= MIN_EPOCHS and best_epoch > 0 and epoch - best_epoch >= PATIENCE:
            stopped_epoch = epoch
            break

    if best_state is None:
        raise RuntimeError(f"No stage-2 checkpoint selected for {spec.key}")

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
            "pairing": "all stage-2 conditions share W0 and loader order within seed",
            "trainable_parameters": HIDDEN_WIDTH * n_classes,
            "ce_gain": CE_GAIN,
        },
        checkpoint_path,
    )
    model.load_state_dict(best_state, strict=True)
    native_metrics = {
        split: _evaluate_stage2_native(spec, model, loader, device)
        for split, loader in eval_loaders.items()
    }
    W = model.output_linear.weight.detach().cpu().numpy().astype(np.float64)
    split_payload = {
        split: cache[split] for split in ("train", "val", "test")
    }
    cross = _cross_evaluate_w(split_payload, W)
    lif_reference_equal: bool | None = None
    if spec.readout == "lif":
        lif_reference_equal = (
            native_metrics["test"]["balanced_accuracy"]
            == cross["test"]["lif_beta05"]["balanced_accuracy"]
        )

    payload = {
        "experiment_id": EXPERIMENT_ID,
        "protocol_version": PROTOCOL_VERSION,
        "group": "two_stage",
        "method": spec.method,
        "spec": asdict(spec),
        "contract": {
            "architecture": ARCHITECTURE,
            "stage1_readout": "linear",
            "stage1_backbone_objective": spec.backbone_objective,
            "frozen_l1_l2": True,
            "discard_stage1_w": True,
            "bias": False,
            "ce_gain": CE_GAIN,
            "lif_beta": LIF_BETA,
            "threshold": THRESHOLD,
            "output_cap": OUTPUT_CAP,
            "stage2_trainable_parameters": HIDDEN_WIDTH * n_classes,
            "checkpoint_metric": "native_validation_balanced_accuracy",
            "final_primary_readout": "lif_beta05_spike_count",
        },
        "best_epoch": best_epoch,
        "stopped_epoch": stopped_epoch,
        "native_metrics": native_metrics,
        "cross_evaluation": cross,
        "representation_probes": cache["metadata"]["representation_probes"],
        "source_backbone_best_epoch": cache["metadata"]["source_best_epoch"],
        "source_method": cache["metadata"]["source_method"],
        "lif_native_reference_ba_equal": lif_reference_equal,
    }
    _save_json(eval_path, payload)
    history_path = _path(config.results_dir, "stage2_histories", spec.key, ".csv")
    history_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(history).to_csv(history_path, index=False)
    return payload


def _probe_test_ba(payload: dict[str, Any], source: str) -> float:
    return float(
        payload["representation_probes"][source]["metrics"]["test"][
            "balanced_accuracy"
        ]
    )


def _method_row(payload: dict[str, Any]) -> dict[str, Any]:
    group = str(payload["group"])
    spec = payload["spec"]
    if group == "e2e":
        backbone_objective = spec["objective"]
        w_objective = spec["objective"]
        train_readout = spec["readout"]
        training_scope = "joint_e2e"
        source_method = payload["method"]
    elif group == "two_stage":
        backbone_objective = spec["backbone_objective"]
        w_objective = spec["objective"]
        train_readout = spec["readout"]
        training_scope = "frozen_backbone_retrain_w"
        source_method = payload["source_method"]
    else:
        raise ValueError(group)
    native = payload["native_metrics"]
    cross = payload["cross_evaluation"]
    analog_ba = float(cross["test"]["analog"]["balanced_accuracy"])
    lif_ba = float(cross["test"]["lif_beta05"]["balanced_accuracy"])
    return {
        "group": group,
        "method": payload["method"],
        "seed": int(spec["seed"]),
        "architecture": ARCHITECTURE,
        "backbone_objective": backbone_objective,
        "w_objective": w_objective,
        "w_train_readout": train_readout,
        "training_scope": training_scope,
        "source_backbone_method": source_method,
        "best_epoch": int(payload["best_epoch"]),
        "stopped_epoch": int(payload["stopped_epoch"]),
        "native_train_ba": float(native["train"]["balanced_accuracy"]),
        "native_val_ba": float(native["val"]["balanced_accuracy"]),
        "native_test_ba": float(native["test"]["balanced_accuracy"]),
        "analog_test_ba": analog_ba,
        "final_lif_test_ba": lif_ba,
        "analog_to_lif_gap_pp": 100.0 * (analog_ba - lif_ba),
        "l2_wholecount_probe_test_ba": _probe_test_ba(
            payload, "l2_wholecount_linear"
        ),
        "l2_fixed250_probe_test_ba": _probe_test_ba(
            payload, "l2_fixed250_linear"
        ),
    }


def _load_required_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def _contrast_frames(runs: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    contrasts = (
        ("A1_minus_A3_e2e_tsce_linear_vs_lif", "A1_e2e_linear_tsce", "A3_e2e_lif_tsce"),
        ("A2_minus_A4_e2e_wcce_linear_vs_lif", "A2_e2e_linear_wcce", "A4_e2e_lif_wcce"),
        ("B2_minus_B6_ts_vs_wc_backbone", "B2_tsbackbone_linear_wcce", "B6_wcbackbone_linear_wcce"),
        ("B2_minus_B1_wcce_vs_tsce_W", "B2_tsbackbone_linear_wcce", "B1_tsbackbone_linear_tsce"),
        ("B2_minus_B4_linear_vs_lif_W_training", "B2_tsbackbone_linear_wcce", "B4_tsbackbone_lif_wcce"),
        ("B2_minus_A1_two_stage_vs_e2e_linear_tsce", "B2_tsbackbone_linear_wcce", "A1_e2e_linear_tsce"),
        ("B2_minus_A2_two_stage_vs_e2e_linear_wcce", "B2_tsbackbone_linear_wcce", "A2_e2e_linear_wcce"),
        ("B2_minus_A3_two_stage_vs_e2e_lif_tsce", "B2_tsbackbone_linear_wcce", "A3_e2e_lif_tsce"),
        ("B2_minus_A4_two_stage_vs_e2e_lif_wcce", "B2_tsbackbone_linear_wcce", "A4_e2e_lif_wcce"),
    )
    rows: list[dict[str, Any]] = []
    for name, left, right in contrasts:
        left_rows = runs[runs.method == left].set_index("seed")
        right_rows = runs[runs.method == right].set_index("seed")
        if set(left_rows.index) != set(SEEDS) or set(right_rows.index) != set(SEEDS):
            raise RuntimeError(f"Incomplete contrast {name}")
        for seed in SEEDS:
            rows.append(
                {
                    "contrast": name,
                    "left_method": left,
                    "right_method": right,
                    "seed": seed,
                    "final_lif_delta_pp": 100.0
                    * (
                        float(left_rows.loc[seed, "final_lif_test_ba"])
                        - float(right_rows.loc[seed, "final_lif_test_ba"])
                    ),
                }
            )
    contrast_runs = pd.DataFrame(rows)
    contrast_summary = (
        contrast_runs.groupby(["contrast", "left_method", "right_method"], sort=False)[
            "final_lif_delta_pp"
        ]
        .agg(["count", "mean", "std"])
        .reset_index()
        .rename(
            columns={
                "count": "n",
                "mean": "final_lif_delta_pp_mean",
                "std": "final_lif_delta_pp_std",
            }
        )
    )
    return contrast_runs, contrast_summary


def finalize(config: Config) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for spec in e2e_specs():
        payload = _load_required_json(
            _path(config.results_dir, "e2e_evaluations", spec.key, ".json")
        )
        rows.append(_method_row(payload))
    for spec in backbone_specs():
        _ = _load_required_json(_stage2_cache_meta_path(config, spec))
    for spec in stage2_specs():
        payload = _load_required_json(
            _path(config.results_dir, "stage2_evaluations", spec.key, ".json")
        )
        rows.append(_method_row(payload))

    runs = pd.DataFrame(rows)
    if len(runs) != 36:
        raise RuntimeError(f"Expected 36 method/seed rows, got {len(runs)}")
    config.results_dir.mkdir(parents=True, exist_ok=True)
    runs.to_csv(config.results_dir / "method_runs.csv", index=False)

    group_cols = [
        "group",
        "method",
        "architecture",
        "backbone_objective",
        "w_objective",
        "w_train_readout",
        "training_scope",
        "source_backbone_method",
    ]
    metric_cols = [
        "native_test_ba",
        "analog_test_ba",
        "final_lif_test_ba",
        "analog_to_lif_gap_pp",
        "l2_wholecount_probe_test_ba",
        "l2_fixed250_probe_test_ba",
        "best_epoch",
    ]
    summary = runs.groupby(group_cols, sort=False)[metric_cols].agg(["mean", "std"]).reset_index()
    summary.columns = [
        "_".join(str(v) for v in col if str(v)) if isinstance(col, tuple) else str(col)
        for col in summary.columns
    ]
    summary.to_csv(config.results_dir / "method_summary.csv", index=False)

    two_stage = summary[summary["group"] == "two_stage"].copy()
    two_stage.to_csv(config.results_dir / "two_stage_matrix_summary.csv", index=False)

    contrast_runs, contrast_summary = _contrast_frames(runs)
    contrast_runs.to_csv(config.results_dir / "contrast_runs.csv", index=False)
    contrast_summary.to_csv(config.results_dir / "contrast_summary.csv", index=False)

    manifest = {
        "experiment_id": EXPERIMENT_ID,
        "protocol_version": PROTOCOL_VERSION,
        "architecture": ARCHITECTURE,
        "shifts": [list(v) for v in SHIFTS],
        "seeds": list(SEEDS),
        "regularization": REGULARIZATION,
        "ce_gain": CE_GAIN,
        "lif_beta": LIF_BETA,
        "threshold": THRESHOLD,
        "output_cap": OUTPUT_CAP,
        "e2e_methods": [name for name, _, _ in E2E_CASES],
        "two_stage_methods": [name for name, _, _, _ in STAGE2_CASES],
        "counts": {
            "e2e_runs": len(e2e_specs()),
            "stage2_backbone_caches": len(backbone_specs()),
            "stage2_runs": len(stage2_specs()),
            "final_method_seed_rows": len(runs),
        },
        "primary_metric": "final_lif_test_ba",
        "checkpoint_rule": "native validation BA primary, native objective loss tiebreak",
        "stage2_contract": "Stage 1 uses Linear TSCE/WCCE to train L1/L2, then discards W; Stage 2 freezes L1/L2 and retrains a fresh W with paired initialization/order.",
        "final_deployment_contract": f"Every method is re-evaluated through the same beta={LIF_BETA:g}, threshold={THRESHOLD:g}, cap={OUTPUT_CAP} LIF spike-count simulator.",
        "hypothesis_method": "B2_tsbackbone_linear_wcce",
    }
    _save_json(config.results_dir / "manifest.json", manifest)
    return manifest


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Exp7.3 training-strategy decomposition")
    parser.add_argument("--repo-root", type=Path, default=None)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=exp72.BATCH_SIZE)
    parser.add_argument("--max-epochs", type=int, default=MAX_EPOCHS)
    parser.add_argument("--force", action="store_true")
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("e2e", "prepare-stage2-cache", "stage2"):
        child = sub.add_parser(name)
        child.add_argument("--array-task-id", type=int, required=True)
    sub.add_parser("finalize")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    root = args.repo_root.resolve() if args.repo_root else find_repo_root()
    config = Config(
        root,
        results_dir(root),
        args.device,
        args.batch_size,
        args.threads,
        args.max_epochs,
    )
    if args.command == "e2e":
        specs = e2e_specs()
        if args.array_task_id < 0 or args.array_task_id >= len(specs):
            raise IndexError(args.array_task_id)
        run_e2e(specs[args.array_task_id], config, force=args.force)
        return
    if args.command == "prepare-stage2-cache":
        specs = backbone_specs()
        if args.array_task_id < 0 or args.array_task_id >= len(specs):
            raise IndexError(args.array_task_id)
        prepare_stage2_cache(specs[args.array_task_id], config, force=args.force)
        return
    if args.command == "stage2":
        specs = stage2_specs()
        if args.array_task_id < 0 or args.array_task_id >= len(specs):
            raise IndexError(args.array_task_id)
        run_stage2(specs[args.array_task_id], config, force=args.force)
        return
    if args.command == "finalize":
        finalize(config)
        return
    raise ValueError(args.command)


if __name__ == "__main__":
    main()
