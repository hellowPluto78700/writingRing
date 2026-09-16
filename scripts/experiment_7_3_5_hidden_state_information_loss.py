from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler

from scripts import experiment_3_0_1_single_tau_objectives as exp3
from scripts import experiment_3_0_2_hidden_multitau_architectures as exp302
from scripts import experiment_7_2_two_layer_tau_training as exp72
from scripts import experiment_7_3_training_strategy_decomposition as exp73


EXPERIMENT_ID = "experiment_7_3_5_hidden_state_information_loss"
PROTOCOL_VERSION = "hidden_state_information_loss_v1"
ARCHITECTURE = exp73.ARCHITECTURE
SOURCE_METHOD = "A2_e2e_linear_wcce"
SOURCE_BACKBONE_OBJECTIVE = "wcce"
SEEDS = exp73.SEEDS
LAYERS = ("l1", "l2")
PRIMARY_STATES = ("syn_current", "pre_reset", "spike")
SECONDARY_STATES = ("post_reset",)
STATES = PRIMARY_STATES + SECONDARY_STATES
AGGREGATIONS = ("whole_mean", "fixed250_ordered_mean")
C_GRID = tuple(float(v) for v in exp302.PROBE_C_GRID)
MAX_ITER = 5000
BATCH_SIZE = exp72.BATCH_SIZE
HIDDEN_WIDTH = exp73.HIDDEN_WIDTH
SOURCE_REPLAY_ATOL = 1e-6
EXPECTED_EXTRACTION_TASKS = len(SEEDS)
EXPECTED_PROBE_TASKS = len(SEEDS) * len(LAYERS) * len(STATES) * len(AGGREGATIONS)


@dataclass(frozen=True)
class ExtractionSpec:
    seed: int

    @property
    def key(self) -> str:
        return f"{ARCHITECTURE}__{SOURCE_METHOD}__seed{self.seed}"


@dataclass(frozen=True)
class ProbeSpec:
    seed: int
    layer: str
    state: str
    aggregation: str

    @property
    def key(self) -> str:
        return (
            f"{ARCHITECTURE}__{SOURCE_METHOD}__seed{self.seed}__"
            f"{self.layer}__{self.state}__{self.aggregation}"
        )


@dataclass(frozen=True)
class Config:
    repo_root: Path
    results_dir: Path
    device: str = "cpu"
    batch_size: int = BATCH_SIZE
    threads: int = 1


def find_repo_root(start: Path | None = None) -> Path:
    return exp73.find_repo_root(start)


def results_dir(repo_root: Path) -> Path:
    return repo_root / "notebooks" / "artifacts" / EXPERIMENT_ID / PROTOCOL_VERSION


def extraction_specs() -> list[ExtractionSpec]:
    return [ExtractionSpec(seed) for seed in SEEDS]


def probe_specs() -> list[ProbeSpec]:
    return [
        ProbeSpec(seed, layer, state, aggregation)
        for seed in SEEDS
        for layer in LAYERS
        for state in STATES
        for aggregation in AGGREGATIONS
    ]


def validate_extraction_spec(spec: ExtractionSpec) -> None:
    if spec.seed not in SEEDS:
        raise ValueError(spec.seed)


def validate_probe_spec(spec: ProbeSpec) -> None:
    if spec.seed not in SEEDS:
        raise ValueError(spec.seed)
    if spec.layer not in LAYERS:
        raise ValueError(spec.layer)
    if spec.state not in STATES:
        raise ValueError(spec.state)
    if spec.aggregation not in AGGREGATIONS:
        raise ValueError(spec.aggregation)


def _save_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _source_e2e_spec(seed: int) -> exp73.E2ESpec:
    return exp73.E2ESpec(seed, "linear", "wcce")


def _source_checkpoint_path(repo_root: Path, seed: int) -> Path:
    spec = _source_e2e_spec(seed)
    return exp73.results_dir(repo_root) / "e2e_checkpoints" / f"{spec.key}.pt"


def _source_l2_cache_paths(repo_root: Path, seed: int) -> tuple[Path, Path]:
    backbone = exp73.BackboneSpec(seed, SOURCE_BACKBONE_OBJECTIVE)
    root = exp73.results_dir(repo_root) / "stage2_l2_cache"
    return root / f"{backbone.key}.npz", root / f"{backbone.key}.json"


def _feature_cache_path(root: Path, spec: ProbeSpec) -> Path:
    return root / "feature_cache" / f"{spec.key}.npz"


def _extraction_meta_path(root: Path, seed: int) -> Path:
    return root / "extractions" / f"{ARCHITECTURE}__{SOURCE_METHOD}__seed{seed}.json"


def _evaluation_path(root: Path, spec: ProbeSpec) -> Path:
    return root / "evaluations" / f"{spec.key}.json"


def _load_source_model(
    config: Config,
    data: exp3.Data,
    seed: int,
) -> tuple[exp73.Exp73Net, dict[str, Any], Path]:
    checkpoint_path = _source_checkpoint_path(config.repo_root, seed)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Missing Exp7.3 A2 checkpoint: {checkpoint_path}")
    payload = torch.load(checkpoint_path, map_location=config.device, weights_only=False)
    if not isinstance(payload, dict):
        raise ValueError(f"Unexpected checkpoint payload: {checkpoint_path}")
    if payload.get("experiment_id") != exp73.EXPERIMENT_ID:
        raise ValueError(f"Unexpected source experiment: {payload.get('experiment_id')!r}")
    if payload.get("protocol_version") != exp73.PROTOCOL_VERSION:
        raise ValueError(f"Unexpected source protocol: {payload.get('protocol_version')!r}")
    stored_spec = payload.get("spec")
    expected_spec = asdict(_source_e2e_spec(seed))
    if stored_spec != expected_spec:
        raise ValueError(
            f"A2 checkpoint identity mismatch for seed {seed}: {stored_spec!r} != {expected_spec!r}"
        )
    state_dict = payload.get("model_state_dict")
    if not isinstance(state_dict, dict):
        raise ValueError(f"Missing model_state_dict: {checkpoint_path}")

    model = exp73.Exp73Net("linear", len(data.labels), data.fs).to(config.device)
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model, payload, checkpoint_path


def _hidden_state_trajectory(
    model: exp73.Exp73Net,
    x: torch.Tensor,
) -> dict[str, dict[str, torch.Tensor]]:
    """Replay the frozen Exp7.3 hidden dynamics and expose internal LIF states.

    State definitions follow MacroMultiSpikeLIF exactly:

        I_t      = alpha * I_{t-1} + W x_t
        U_t^-    = beta * U_{t-1} + I_t
        S_t      = H(U_t^- - threshold) with binary event cap
        U_t      = U_t^- - threshold * S_t

    The returned ``post_reset`` state is U_t. Forward communication to the next
    hidden layer remains the binary spike S_t, exactly as in Exp7.3.
    """
    batch, steps, channels = x.shape
    if channels != exp72.EXPECTED_CHANNELS:
        raise ValueError(f"Expected {exp72.EXPECTED_CHANNELS} channels, got {channels}")

    syn = [
        torch.zeros(batch, HIDDEN_WIDTH, device=x.device, dtype=x.dtype)
        for _ in LAYERS
    ]
    mem = [torch.zeros_like(syn[0]), torch.zeros_like(syn[1])]
    collected: dict[str, dict[str, list[torch.Tensor]]] = {
        layer: {state: [] for state in STATES} for layer in LAYERS
    }

    for timestep in range(steps):
        current = x[:, timestep]
        for layer_index, layer in enumerate(LAYERS):
            alpha = getattr(model, f"alpha_{layer_index}")
            syn[layer_index] = (
                alpha * syn[layer_index] + model.hidden_linears[layer_index](current)
            )
            spike, post_reset, pre_reset = model.hidden_lifs[layer_index](
                syn[layer_index], mem[layer_index]
            )
            mem[layer_index] = post_reset
            collected[layer]["syn_current"].append(syn[layer_index])
            collected[layer]["pre_reset"].append(pre_reset)
            collected[layer]["spike"].append(spike)
            collected[layer]["post_reset"].append(post_reset)
            current = spike

    return {
        layer: {
            state: torch.stack(collected[layer][state], dim=1)
            for state in STATES
        }
        for layer in LAYERS
    }


def _valid_mask(lengths: torch.Tensor, n_steps: int) -> torch.Tensor:
    return torch.arange(n_steps, device=lengths.device)[None, :] < lengths[:, None]


def _whole_mean(
    values: torch.Tensor,
    lengths: torch.Tensor,
) -> torch.Tensor:
    mask = _valid_mask(lengths, values.shape[1]).to(values.dtype).unsqueeze(-1)
    summed = (values * mask).sum(dim=1)
    return summed / lengths.clamp_min(1).to(values.dtype).unsqueeze(1)


def _fixed250_ordered_mean(
    values: torch.Tensor,
    lengths: torch.Tensor,
    bin_steps: int,
) -> torch.Tensor:
    if bin_steps <= 0:
        raise ValueError("bin_steps must be positive")
    batch, steps, width = values.shape
    n_bins = int(math.ceil(steps / bin_steps))
    padded_steps = n_bins * bin_steps
    mask = _valid_mask(lengths, steps).to(values.dtype)
    if padded_steps != steps:
        value_pad = torch.zeros(
            batch,
            padded_steps - steps,
            width,
            device=values.device,
            dtype=values.dtype,
        )
        mask_pad = torch.zeros(
            batch,
            padded_steps - steps,
            device=values.device,
            dtype=values.dtype,
        )
        values = torch.cat([values, value_pad], dim=1)
        mask = torch.cat([mask, mask_pad], dim=1)
    masked = values * mask.unsqueeze(-1)
    sums = masked.reshape(batch, n_bins, bin_steps, width).sum(dim=2)
    counts = mask.reshape(batch, n_bins, bin_steps).sum(dim=2).clamp_min(1.0)
    means = sums / counts.unsqueeze(-1)
    return means.flatten(start_dim=1)


def _aggregate_tensor(
    values: torch.Tensor,
    lengths: torch.Tensor,
    aggregation: str,
    bin_steps: int,
) -> torch.Tensor:
    if aggregation == "whole_mean":
        return _whole_mean(values, lengths)
    if aggregation == "fixed250_ordered_mean":
        return _fixed250_ordered_mean(values, lengths, bin_steps)
    raise ValueError(aggregation)


def _aggregate_numpy(
    values: np.ndarray,
    lengths: np.ndarray,
    aggregation: str,
    bin_steps: int,
) -> np.ndarray:
    tensor = torch.from_numpy(np.asarray(values, dtype=np.float32))
    lt = torch.from_numpy(np.asarray(lengths, dtype=np.int64))
    return _aggregate_tensor(tensor, lt, aggregation, bin_steps).numpy()


def _split_arrays(data: exp3.Data) -> dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    return {
        "train": (data.Xtr, data.ytr, data.ltr),
        "val": (data.Xva, data.yva, data.lva),
        "test": (data.Xte, data.yte, data.lte),
    }


def _load_exp73_l2_cache(
    repo_root: Path,
    seed: int,
) -> tuple[dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]], dict[str, Any]]:
    npz_path, meta_path = _source_l2_cache_paths(repo_root, seed)
    if not npz_path.exists() or not meta_path.exists():
        raise FileNotFoundError(
            f"Missing Exp7.3 WCCE L2 cache for seed {seed}: {npz_path} / {meta_path}"
        )
    metadata = json.loads(meta_path.read_text(encoding="utf-8"))
    if metadata.get("source_method") != SOURCE_METHOD:
        raise ValueError(
            f"Unexpected Exp7.3 cache source for seed {seed}: {metadata.get('source_method')!r}"
        )
    with np.load(npz_path, allow_pickle=False) as z:
        splits = {
            split: (
                z[f"{split}_l2"].copy(),
                z[f"{split}_y"].copy(),
                z[f"{split}_lengths"].copy(),
            )
            for split in ("train", "val", "test")
        }
    return splits, metadata


def extract_seed_features(
    spec: ExtractionSpec,
    config: Config,
    force: bool = False,
) -> dict[str, Any]:
    validate_extraction_spec(spec)
    metadata_path = _extraction_meta_path(config.results_dir, spec.seed)
    expected_paths = [
        _feature_cache_path(config.results_dir, probe)
        for probe in probe_specs()
        if probe.seed == spec.seed
    ]
    if metadata_path.exists() and all(path.exists() for path in expected_paths) and not force:
        return json.loads(metadata_path.read_text(encoding="utf-8"))

    torch.set_num_threads(config.threads)
    data = exp3.prepare_data(config.repo_root)
    device = torch.device(config.device)
    model, checkpoint, checkpoint_path = _load_source_model(config, data, spec.seed)

    feature_parts: dict[tuple[str, str, str, str], list[np.ndarray]] = {}
    label_parts: dict[str, list[np.ndarray]] = {split: [] for split in ("train", "val", "test")}
    length_parts: dict[str, list[np.ndarray]] = {split: [] for split in ("train", "val", "test")}

    for split, (X, y, lengths) in _split_arrays(data).items():
        loader = exp3.loader(
            X,
            y,
            lengths,
            config.batch_size,
            False,
            exp3.dseed(spec.seed, EXPERIMENT_ID, "extract", split),
        )
        with torch.no_grad():
            for xb, yb, lb in loader:
                xb = xb.to(device)
                lb_device = lb.to(device)
                trajectory = _hidden_state_trajectory(model, xb)
                for layer in LAYERS:
                    for state in STATES:
                        values = trajectory[layer][state]
                        for aggregation in AGGREGATIONS:
                            features = _aggregate_tensor(
                                values,
                                lb_device,
                                aggregation,
                                data.bin_steps,
                            )
                            key = (split, layer, state, aggregation)
                            feature_parts.setdefault(key, []).append(
                                features.cpu().numpy().astype(np.float32, copy=False)
                            )
                label_parts[split].append(yb.numpy().astype(np.int64, copy=False))
                length_parts[split].append(lb.numpy().astype(np.int64, copy=False))

    features_by_key = {
        key: np.concatenate(parts, axis=0)
        for key, parts in feature_parts.items()
    }
    labels = {split: np.concatenate(parts) for split, parts in label_parts.items()}
    split_lengths = {split: np.concatenate(parts) for split, parts in length_parts.items()}

    source_cache, source_cache_meta = _load_exp73_l2_cache(config.repo_root, spec.seed)
    reproduction: dict[str, dict[str, float]] = {}
    for split in ("train", "val", "test"):
        source_l2, source_y, source_lengths = source_cache[split]
        if not np.array_equal(labels[split], source_y):
            raise ValueError(f"Label identity mismatch against Exp7.3 L2 cache: seed={spec.seed} split={split}")
        if not np.array_equal(split_lengths[split], source_lengths):
            raise ValueError(f"Length identity mismatch against Exp7.3 L2 cache: seed={spec.seed} split={split}")
        reproduction[split] = {}
        for aggregation in AGGREGATIONS:
            expected = _aggregate_numpy(
                source_l2,
                source_lengths,
                aggregation,
                data.bin_steps,
            )
            actual = features_by_key[(split, "l2", "spike", aggregation)]
            max_abs = float(np.max(np.abs(expected - actual))) if expected.size else 0.0
            reproduction[split][aggregation] = max_abs
            if max_abs > SOURCE_REPLAY_ATOL:
                raise ValueError(
                    "Replayed L2 spike features do not reproduce Exp7.3 cache: "
                    f"seed={spec.seed} split={split} aggregation={aggregation} max_abs={max_abs}"
                )

    for probe in probe_specs():
        if probe.seed != spec.seed:
            continue
        path = _feature_cache_path(config.results_dir, probe)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            train_x=features_by_key[("train", probe.layer, probe.state, probe.aggregation)],
            train_y=labels["train"],
            val_x=features_by_key[("val", probe.layer, probe.state, probe.aggregation)],
            val_y=labels["val"],
            test_x=features_by_key[("test", probe.layer, probe.state, probe.aggregation)],
            test_y=labels["test"],
        )

    payload: dict[str, Any] = {
        "experiment_id": EXPERIMENT_ID,
        "protocol_version": PROTOCOL_VERSION,
        "spec": asdict(spec),
        "source": {
            "experiment_id": exp73.EXPERIMENT_ID,
            "protocol_version": exp73.PROTOCOL_VERSION,
            "method": SOURCE_METHOD,
            "checkpoint": str(checkpoint_path.relative_to(config.repo_root)),
            "best_epoch": int(checkpoint.get("best_epoch", -1)),
            "stage2_cache_source_method": source_cache_meta.get("source_method"),
            "stage2_cache_source_best_epoch": source_cache_meta.get("source_best_epoch"),
        },
        "contract": {
            "architecture": ARCHITECTURE,
            "frozen_l1_l2": True,
            "forward_communication": "binary hidden spikes exactly as Exp7.3 A2",
            "states": list(STATES),
            "primary_states": list(PRIMARY_STATES),
            "secondary_states": list(SECONDARY_STATES),
            "aggregations": list(AGGREGATIONS),
            "fixed250_bin_steps": int(data.bin_steps),
            "fixed250_bin_ms": float(exp3.FIXED_MS),
            "whole_feature_dim": HIDDEN_WIDTH,
            "fixed250_feature_dim": int(data.n_bins * HIDDEN_WIDTH),
            "probe_training": "none during extraction; feature cache only",
        },
        "provenance": {
            "labels": list(data.labels),
            "fs_hz": float(data.fs),
            "timesteps": int(data.T),
            "n_fixed250_bins": int(data.n_bins),
            "split_seed": int(exp3.SPLIT_SEED),
            "split": {key: list(value) for key, value in data.split.items()},
        },
        "source_replay_max_abs_delta": reproduction,
        "source_replay_tolerance": SOURCE_REPLAY_ATOL,
        "feature_cache_count": len(expected_paths),
    }
    _save_json(metadata_path, payload)
    return payload


def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
    }


def _probe_seed(spec: ProbeSpec) -> int:
    return int(
        exp3.dseed(
            spec.seed,
            EXPERIMENT_ID,
            "probe",
            spec.layer,
            spec.state,
            spec.aggregation,
        )
    )


def run_probe(
    spec: ProbeSpec,
    config: Config,
    force: bool = False,
) -> dict[str, Any]:
    validate_probe_spec(spec)
    output_path = _evaluation_path(config.results_dir, spec)
    if output_path.exists() and not force:
        return json.loads(output_path.read_text(encoding="utf-8"))

    extraction_meta = _extraction_meta_path(config.results_dir, spec.seed)
    cache_path = _feature_cache_path(config.results_dir, spec)
    if not extraction_meta.exists() or not cache_path.exists():
        raise FileNotFoundError(
            f"Missing Exp7.3.5 extraction artifacts for {spec.key}: {extraction_meta} / {cache_path}"
        )
    source = json.loads(extraction_meta.read_text(encoding="utf-8"))
    with np.load(cache_path, allow_pickle=False) as z:
        features = {
            split: (
                z[f"{split}_x"].astype(np.float64, copy=True),
                z[f"{split}_y"].astype(np.int64, copy=True),
            )
            for split in ("train", "val", "test")
        }

    scaler = StandardScaler().fit(features["train"][0])
    transformed = {
        split: (scaler.transform(x), y)
        for split, (x, y) in features.items()
    }

    seed = _probe_seed(spec)
    candidates: list[dict[str, Any]] = []
    best: tuple[float, float, LogisticRegression] | None = None
    for C in C_GRID:
        classifier = LogisticRegression(
            C=C,
            max_iter=MAX_ITER,
            solver="lbfgs",
            random_state=seed,
            fit_intercept=True,
        ).fit(transformed["train"][0], transformed["train"][1])
        val_pred = classifier.predict(transformed["val"][0])
        val_ba = float(balanced_accuracy_score(transformed["val"][1], val_pred))
        candidates.append(
            {
                "C": float(C),
                "val_balanced_accuracy": val_ba,
                "n_iter_max": int(np.asarray(classifier.n_iter_).max()),
            }
        )
        if best is None or val_ba > best[0] + 1e-12:
            best = (val_ba, float(C), classifier)
    if best is None:
        raise RuntimeError(f"No LogisticRegression candidate selected for {spec.key}")

    selected_val_ba, selected_C, classifier = best
    split_metrics = {
        split: _metrics(y, classifier.predict(x))
        for split, (x, y) in transformed.items()
    }
    payload: dict[str, Any] = {
        "experiment_id": EXPERIMENT_ID,
        "protocol_version": PROTOCOL_VERSION,
        "spec": asdict(spec),
        "source": source["source"],
        "contract": {
            "frozen_snn": True,
            "snn_retrained": False,
            "feature_cache": str(cache_path.relative_to(config.repo_root)),
            "scaler_fit": "train only",
            "scaler_with_mean": True,
            "scaler_with_std": True,
            "classifier": "LogisticRegression",
            "solver": "lbfgs",
            "fit_intercept": True,
            "max_iter": MAX_ITER,
            "C_grid": list(C_GRID),
            "selection": "highest validation balanced accuracy; first C in grid wins ties",
            "test_untouched_until_C_selection": True,
        },
        "feature_dim": int(transformed["train"][0].shape[1]),
        "selected_C": selected_C,
        "selected_val_balanced_accuracy": selected_val_ba,
        "candidate_validation": candidates,
        "metrics": split_metrics,
        "scaler": {
            "scale_min": float(np.min(scaler.scale_)),
            "scale_max": float(np.max(scaler.scale_)),
            "mean_l2": float(np.linalg.norm(scaler.mean_)),
        },
    }
    _save_json(output_path, payload)
    return payload


def _run_row(payload: dict[str, Any]) -> dict[str, Any]:
    spec = payload["spec"]
    metrics = payload["metrics"]
    return {
        "seed": int(spec["seed"]),
        "layer": spec["layer"],
        "state": spec["state"],
        "state_role": "primary" if spec["state"] in PRIMARY_STATES else "secondary",
        "aggregation": spec["aggregation"],
        "feature_dim": int(payload["feature_dim"]),
        "selected_C": float(payload["selected_C"]),
        "train_accuracy": float(metrics["train"]["accuracy"]),
        "train_ba": float(metrics["train"]["balanced_accuracy"]),
        "train_macro_f1": float(metrics["train"]["macro_f1"]),
        "val_accuracy": float(metrics["val"]["accuracy"]),
        "val_ba": float(metrics["val"]["balanced_accuracy"]),
        "val_macro_f1": float(metrics["val"]["macro_f1"]),
        "test_accuracy": float(metrics["test"]["accuracy"]),
        "test_ba": float(metrics["test"]["balanced_accuracy"]),
        "test_macro_f1": float(metrics["test"]["macro_f1"]),
    }


def _contrast_rows(runs: pd.DataFrame) -> pd.DataFrame:
    definitions = (
        ("membrane_integration_pre_minus_current", "pre_reset", "syn_current"),
        ("spike_quantization_pre_minus_spike", "pre_reset", "spike"),
        ("reset_pre_minus_post", "pre_reset", "post_reset"),
        ("postreset_minus_spike", "post_reset", "spike"),
    )
    rows: list[dict[str, Any]] = []
    for contrast, left_state, right_state in definitions:
        for layer in LAYERS:
            for aggregation in AGGREGATIONS:
                left = runs[
                    (runs.layer == layer)
                    & (runs.aggregation == aggregation)
                    & (runs.state == left_state)
                ][["seed", "test_ba"]]
                right = runs[
                    (runs.layer == layer)
                    & (runs.aggregation == aggregation)
                    & (runs.state == right_state)
                ][["seed", "test_ba"]]
                merged = left.merge(right, on="seed", suffixes=("_left", "_right"))
                for row in merged.itertuples(index=False):
                    rows.append(
                        {
                            "contrast": contrast,
                            "layer": layer,
                            "aggregation": aggregation,
                            "seed": int(row.seed),
                            "left_state": left_state,
                            "right_state": right_state,
                            "left_test_ba": float(row.test_ba_left),
                            "right_test_ba": float(row.test_ba_right),
                            "delta_pp": 100.0 * (
                                float(row.test_ba_left) - float(row.test_ba_right)
                            ),
                        }
                    )
    return pd.DataFrame(rows)


def _flatten_summary(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out.columns = [
        "_".join(str(part) for part in column if str(part))
        if isinstance(column, tuple)
        else str(column)
        for column in out.columns
    ]
    return out


def finalize(config: Config) -> None:
    missing_extractions = [
        _extraction_meta_path(config.results_dir, seed)
        for seed in SEEDS
        if not _extraction_meta_path(config.results_dir, seed).exists()
    ]
    if missing_extractions:
        raise FileNotFoundError(
            "Missing Exp7.3.5 extraction metadata:\n"
            + "\n".join(str(path) for path in missing_extractions)
        )

    payloads: list[dict[str, Any]] = []
    missing: list[Path] = []
    for spec in probe_specs():
        path = _evaluation_path(config.results_dir, spec)
        if not path.exists():
            missing.append(path)
        else:
            payloads.append(json.loads(path.read_text(encoding="utf-8")))
    if missing:
        preview = "\n".join(str(path) for path in missing[:10])
        raise FileNotFoundError(f"Missing {len(missing)} Exp7.3.5 probe evaluations:\n{preview}")

    runs = pd.DataFrame([_run_row(payload) for payload in payloads]).sort_values(
        ["aggregation", "layer", "state", "seed"]
    )
    config.results_dir.mkdir(parents=True, exist_ok=True)
    runs.to_csv(config.results_dir / "method_runs.csv", index=False)

    metric_cols = [
        "selected_C",
        "train_ba",
        "val_ba",
        "test_ba",
        "test_accuracy",
        "test_macro_f1",
    ]
    summary = (
        runs.groupby(["aggregation", "layer", "state", "state_role"], sort=False)[metric_cols]
        .agg(["mean", "std"])
        .reset_index()
    )
    _flatten_summary(summary).to_csv(config.results_dir / "method_summary.csv", index=False)

    contrasts = _contrast_rows(runs)
    contrasts.to_csv(config.results_dir / "contrast_runs.csv", index=False)
    contrast_summary = (
        contrasts.groupby(["contrast", "layer", "aggregation"], sort=False)["delta_pp"]
        .agg(["count", "mean", "std"])
        .reset_index()
    )
    contrast_summary.to_csv(config.results_dir / "contrast_summary.csv", index=False)

    extraction_rows: list[dict[str, Any]] = []
    for seed in SEEDS:
        payload = json.loads(
            _extraction_meta_path(config.results_dir, seed).read_text(encoding="utf-8")
        )
        for split, values in payload["source_replay_max_abs_delta"].items():
            for aggregation, max_abs in values.items():
                extraction_rows.append(
                    {
                        "seed": seed,
                        "split": split,
                        "aggregation": aggregation,
                        "l2_spike_replay_max_abs_delta": float(max_abs),
                        "tolerance": float(payload["source_replay_tolerance"]),
                    }
                )
    pd.DataFrame(extraction_rows).to_csv(
        config.results_dir / "source_reproduction_checks.csv", index=False
    )

    manifest = {
        "experiment_id": EXPERIMENT_ID,
        "protocol_version": PROTOCOL_VERSION,
        "architecture": ARCHITECTURE,
        "source_method": SOURCE_METHOD,
        "seeds": list(SEEDS),
        "layers": list(LAYERS),
        "primary_states": list(PRIMARY_STATES),
        "secondary_states": list(SECONDARY_STATES),
        "aggregations": list(AGGREGATIONS),
        "C_grid": list(C_GRID),
        "expected_extraction_tasks": EXPECTED_EXTRACTION_TASKS,
        "expected_probe_tasks": EXPECTED_PROBE_TASKS,
        "completed_probe_tasks": int(len(runs)),
        "primary_contrasts": [
            "membrane_integration_pre_minus_current",
            "spike_quantization_pre_minus_spike",
        ],
        "secondary_contrasts": ["reset_pre_minus_post", "postreset_minus_spike"],
        "interpretation": {
            "positive_spike_quantization_pre_minus_spike": (
                "pre-threshold membrane contains more linearly accessible test information than binary spikes"
            ),
            "near_zero_spike_quantization_pre_minus_spike": (
                "binary hidden spikes preserve nearly all linearly accessible information measured by this probe"
            ),
        },
    }
    _save_json(config.results_dir / "manifest.json", manifest)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Exp7.3.5 hidden-state information-loss probe")
    parser.add_argument("--repo-root", type=Path, default=None)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--threads", type=int, default=1)
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("list-runs")

    extract = subparsers.add_parser("extract")
    extract.add_argument("--array-task-id", type=int, required=True)
    extract.add_argument("--force", action="store_true")

    probe = subparsers.add_parser("probe")
    probe.add_argument("--array-task-id", type=int, required=True)
    probe.add_argument("--force", action="store_true")

    subparsers.add_parser("finalize")
    return parser


def main() -> None:
    args = _parser().parse_args()
    repo_root = find_repo_root(args.repo_root)
    config = Config(
        repo_root=repo_root,
        results_dir=results_dir(repo_root),
        device=args.device,
        batch_size=args.batch_size,
        threads=args.threads,
    )

    if args.command == "list-runs":
        print(
            json.dumps(
                {
                    "extractions": [asdict(spec) for spec in extraction_specs()],
                    "probes": [asdict(spec) for spec in probe_specs()],
                },
                indent=2,
            )
        )
        return
    if args.command == "extract":
        specs = extraction_specs()
        if not 0 <= args.array_task_id < len(specs):
            raise IndexError(args.array_task_id)
        payload = extract_seed_features(specs[args.array_task_id], config, force=args.force)
        print(json.dumps(payload, indent=2, sort_keys=True))
        return
    if args.command == "probe":
        specs = probe_specs()
        if not 0 <= args.array_task_id < len(specs):
            raise IndexError(args.array_task_id)
        payload = run_probe(specs[args.array_task_id], config, force=args.force)
        print(json.dumps(payload, indent=2, sort_keys=True))
        return
    if args.command == "finalize":
        finalize(config)
        return
    raise ValueError(args.command)


if __name__ == "__main__":
    main()
