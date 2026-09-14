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

from scripts import experiment_3_0_1_single_tau_objective_comparison as exp3
from scripts import experiment_3_0_2_hidden_multitau_architectures as exp302
from scripts import experiment_7_2_two_layer_tau_training as exp72


EXPERIMENT_ID = "experiment_7_2_2_output_readout_decomposition"
PROTOCOL_VERSION = "output_readout_decomposition_v1"

SOURCE_ANALOG = "probe_analog_sum"
SOURCE_IF = "probe_if_beta1_cap1"
SOURCE_LIF = "probe_lif_beta05_cap1"
SOURCE_MULTI = "probe_lif_beta05_cap31"
SOURCE_BIPOLAR = "probe_bipolar_lif_beta05_cap1"
SOURCE_NATIVE_ANALOG = "native_w_analog_sum"
SOURCE_NATIVE_LIF = "native_e2e_lif"

PROBE_SOURCES = (
    SOURCE_ANALOG,
    SOURCE_IF,
    SOURCE_LIF,
    SOURCE_MULTI,
    SOURCE_BIPOLAR,
)
NATIVE_SOURCES = (SOURCE_NATIVE_ANALOG, SOURCE_NATIVE_LIF)
METRICS = ("accuracy", "balanced_accuracy", "macro_f1")
THRESHOLD = float(exp72.THRESHOLD)
BETA_LIF_NAME = 0.5
MULTISPIKE_CAP = 31


@dataclass(frozen=True)
class Config:
    repo_root: Path
    results_dir: Path
    device: str = "cpu"
    batch_size: int = exp72.BATCH_SIZE
    threads: int = 1


def find_repo_root(start: Path | None = None) -> Path:
    return exp72.find_repo_root(start)


def results_dir(repo_root: Path) -> Path:
    return repo_root / "notebooks" / "artifacts" / EXPERIMENT_ID / PROTOCOL_VERSION


def run_specs() -> list[exp72.RunSpec]:
    # Exp7.2.2 is a frozen-checkpoint extension: exactly the Exp7.2 84-run grid.
    return exp72.run_specs()


def _save_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _run_path(root: Path, spec: exp72.RunSpec) -> Path:
    return root / "runs" / f"{spec.key}.json"


def _base_config(config: Config) -> exp72.Config:
    return exp72.Config(
        config.repo_root,
        exp72.results_dir(config.repo_root),
        config.device,
        exp72.MAX_EPOCHS,
        config.batch_size,
        config.threads,
    )


def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
    }


def _predict_from_scores(scores: np.ndarray, classes: np.ndarray) -> np.ndarray:
    if scores.ndim != 2:
        raise ValueError(f"Expected [N,K] scores, got {scores.shape}")
    return classes[np.argmax(scores, axis=1)]


def _valid_mask(lengths: np.ndarray, steps: int) -> np.ndarray:
    return np.arange(steps)[None, :] < lengths[:, None]


def _whole_count(l2: np.ndarray, lengths: np.ndarray) -> np.ndarray:
    if l2.ndim != 3:
        raise ValueError(f"Expected [N,T,D] L2 spikes, got {l2.shape}")
    valid = _valid_mask(lengths, l2.shape[1]).astype(l2.dtype)[..., None]
    return (l2 * valid).sum(axis=1)


def _extract_l2(
    model: exp72.TwoLayerTauSNN,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    ls: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for X, y, lengths in loader:
            trajectory = model.forward_trajectory(X.to(device))
            xs.append(trajectory["hidden_spikes"][-1].cpu().numpy().astype(np.float32))
            ys.append(y.numpy())
            ls.append(lengths.numpy())
    return np.concatenate(xs), np.concatenate(ys), np.concatenate(ls)


def _fit_wholecount_probe(
    train_x: np.ndarray,
    train_y: np.ndarray,
    val_x: np.ndarray,
    val_y: np.ndarray,
    test_x: np.ndarray,
    test_y: np.ndarray,
    seed: int,
    n_classes: int,
) -> dict[str, Any]:
    """Fit the exact Exp7.2-style standardized LogisticRegression probe.

    The returned W_eff and b_eff fold StandardScaler into a single affine
    classifier so streaming analog accumulation can use raw L2 spikes:

        logits = W_eff @ sum_t z_t + b_eff.
    """
    scaler = StandardScaler().fit(train_x)
    train_z = scaler.transform(train_x)
    val_z = scaler.transform(val_x)
    test_z = scaler.transform(test_x)

    best: tuple[float, float, LogisticRegression] | None = None
    for C in exp302.PROBE_C_GRID:
        classifier = LogisticRegression(
            C=C,
            max_iter=5000,
            solver="lbfgs",
            random_state=seed,
        ).fit(train_z, train_y)
        val_ba = float(balanced_accuracy_score(val_y, classifier.predict(val_z)))
        if best is None or val_ba > best[0] + 1e-12:
            best = (val_ba, float(C), classifier)
    if best is None:
        raise RuntimeError("No whole-count probe candidate selected")

    _, selected_C, classifier = best
    if len(classifier.classes_) != n_classes or classifier.coef_.shape != (n_classes, train_x.shape[1]):
        raise RuntimeError(
            f"Expected multinomial {n_classes}-class probe, got classes={classifier.classes_} "
            f"coef={classifier.coef_.shape}"
        )

    # Fold x_scaled=(x-mean)/scale into logits=W_eff x + b_eff.
    W_eff = classifier.coef_.astype(np.float64) / scaler.scale_[None, :]
    b_eff = classifier.intercept_.astype(np.float64) - (
        classifier.coef_.astype(np.float64) * (scaler.mean_ / scaler.scale_)[None, :]
    ).sum(axis=1)
    classes = classifier.classes_.astype(int)

    metrics = {
        "train": _metrics(train_y, classifier.predict(train_z)),
        "val": _metrics(val_y, classifier.predict(val_z)),
        "test": _metrics(test_y, classifier.predict(test_z)),
    }
    for name, raw_x, y in (
        ("train", train_x, train_y),
        ("val", val_x, val_y),
        ("test", test_x, test_y),
    ):
        scores = raw_x.astype(np.float64) @ W_eff.T + b_eff[None, :]
        pred = _predict_from_scores(scores, classes)
        if not np.array_equal(pred, classifier.predict(scaler.transform(raw_x))):
            raise RuntimeError(f"Folded scaler/probe prediction mismatch on {name}")

    return {
        "probe_C": selected_C,
        "classes": classes,
        "W_eff": W_eff,
        "b_eff": b_eff,
        "metrics": metrics,
        "feature_dim": int(train_x.shape[1]),
    }


def _evidence_diagnostics(
    l2: np.ndarray,
    lengths: np.ndarray,
    W: np.ndarray,
) -> dict[str, float]:
    valid = _valid_mask(lengths, l2.shape[1])
    negative = 0
    positions = 0
    abs_sum = 0.0
    for t in range(l2.shape[1]):
        current = l2[:, t].astype(np.float64) @ W.T
        vt = valid[:, t]
        if not np.any(vt):
            continue
        selected = current[vt]
        negative += int((selected < 0).sum())
        positions += int(selected.size)
        abs_sum += float(np.abs(selected).sum())
    return {
        "negative_evidence_fraction": negative / max(positions, 1),
        "mean_abs_evidence": abs_sum / max(positions, 1),
    }


def _simulate_lif(
    l2: np.ndarray,
    lengths: np.ndarray,
    W: np.ndarray,
    bias: np.ndarray,
    beta: float,
    threshold: float,
    cap: int,
    fs: float,
) -> tuple[np.ndarray, dict[str, float]]:
    """Apply W per timestep, then MacroMultiSpikeLIF forward dynamics.

    Bias is intentionally added only once to the final class score. This keeps
    the affine probe reference exact and avoids injecting the intercept at every
    timestep.
    """
    n, steps, _ = l2.shape
    k = W.shape[0]
    mem = np.zeros((n, k), dtype=np.float64)
    counts = np.zeros((n, k), dtype=np.float64)
    valid = _valid_mask(lengths, steps)
    event_total = 0.0
    valid_positions = 0
    potential_multi = 0

    for t in range(steps):
        current = l2[:, t].astype(np.float64) @ W.T
        vt = valid[:, t]
        if not np.any(vt):
            continue
        pre = beta * mem + current
        spikes = np.floor(np.maximum(pre, 0.0) / threshold)
        spikes = np.minimum(spikes, float(cap))
        spikes[~vt] = 0.0
        new_mem = pre - spikes * threshold
        mem[vt] = new_mem[vt]
        counts += spikes

        selected_pre = pre[vt]
        event_total += float(spikes[vt].sum())
        valid_positions += int(selected_pre.size)
        potential_multi += int((selected_pre >= 2.0 * threshold).sum())

    scores = counts + bias[None, :]
    diag = {
        "mean_events_per_valid_output_position": event_total / max(valid_positions, 1),
        "events_per_neuron_second": event_total * fs / max(valid_positions, 1),
        "potential_multi_crossing_fraction": potential_multi / max(valid_positions, 1),
        "zero_spike_sample_fraction": float((counts.sum(axis=1) == 0).mean()),
        "mean_abs_final_membrane": float(np.abs(mem).mean()),
        "mean_total_output_events_per_sample": float(counts.sum(axis=1).mean()),
    }
    return scores, diag


def _simulate_bipolar_lif(
    l2: np.ndarray,
    lengths: np.ndarray,
    W: np.ndarray,
    bias: np.ndarray,
    beta: float,
    threshold: float,
    cap: int,
    fs: float,
) -> tuple[np.ndarray, dict[str, float]]:
    """Encode positive and negative class evidence in separate spike banks."""
    n, steps, _ = l2.shape
    k = W.shape[0]
    pos_mem = np.zeros((n, k), dtype=np.float64)
    neg_mem = np.zeros((n, k), dtype=np.float64)
    pos_count = np.zeros((n, k), dtype=np.float64)
    neg_count = np.zeros((n, k), dtype=np.float64)
    valid = _valid_mask(lengths, steps)
    valid_positions = 0
    pos_events = 0.0
    neg_events = 0.0
    potential_multi = 0

    for t in range(steps):
        evidence = l2[:, t].astype(np.float64) @ W.T
        pos_current = np.maximum(evidence, 0.0)
        neg_current = np.maximum(-evidence, 0.0)
        vt = valid[:, t]
        if not np.any(vt):
            continue

        pos_pre = beta * pos_mem + pos_current
        neg_pre = beta * neg_mem + neg_current
        pos_spk = np.minimum(np.floor(np.maximum(pos_pre, 0.0) / threshold), float(cap))
        neg_spk = np.minimum(np.floor(np.maximum(neg_pre, 0.0) / threshold), float(cap))
        pos_spk[~vt] = 0.0
        neg_spk[~vt] = 0.0
        pos_new = pos_pre - pos_spk * threshold
        neg_new = neg_pre - neg_spk * threshold
        pos_mem[vt] = pos_new[vt]
        neg_mem[vt] = neg_new[vt]
        pos_count += pos_spk
        neg_count += neg_spk

        pos_events += float(pos_spk[vt].sum())
        neg_events += float(neg_spk[vt].sum())
        valid_positions += int(pos_pre[vt].size)
        potential_multi += int(((pos_pre[vt] >= 2.0 * threshold) | (neg_pre[vt] >= 2.0 * threshold)).sum())

    scores = pos_count - neg_count + bias[None, :]
    total_events = pos_events + neg_events
    diag = {
        "mean_events_per_valid_output_position": total_events / max(valid_positions, 1),
        "events_per_neuron_second": total_events * fs / max(valid_positions, 1),
        "potential_multi_crossing_fraction": potential_multi / max(valid_positions, 1),
        "zero_spike_sample_fraction": float(((pos_count + neg_count).sum(axis=1) == 0).mean()),
        "mean_abs_final_membrane": float((np.abs(pos_mem) + np.abs(neg_mem)).mean() / 2.0),
        "mean_total_output_events_per_sample": float((pos_count + neg_count).sum(axis=1).mean()),
        "positive_event_fraction": pos_events / max(total_events, 1.0),
        "negative_event_fraction": neg_events / max(total_events, 1.0),
    }
    return scores, diag


def _analog_scores(l2: np.ndarray, lengths: np.ndarray, W: np.ndarray, bias: np.ndarray) -> np.ndarray:
    return _whole_count(l2, lengths).astype(np.float64) @ W.T + bias[None, :]


def _split_readouts(
    l2: np.ndarray,
    y: np.ndarray,
    lengths: np.ndarray,
    W: np.ndarray,
    bias: np.ndarray,
    classes: np.ndarray,
    beta_lif: float,
    fs: float,
) -> tuple[dict[str, dict[str, float]], dict[str, dict[str, float]]]:
    metrics: dict[str, dict[str, float]] = {}
    diagnostics: dict[str, dict[str, float]] = {}

    analog = _analog_scores(l2, lengths, W, bias)
    metrics[SOURCE_ANALOG] = _metrics(y, _predict_from_scores(analog, classes))
    diagnostics[SOURCE_ANALOG] = _evidence_diagnostics(l2, lengths, W)

    conditions = (
        (SOURCE_IF, 1.0, 1),
        (SOURCE_LIF, beta_lif, 1),
        (SOURCE_MULTI, beta_lif, MULTISPIKE_CAP),
    )
    for source, beta, cap in conditions:
        scores, diag = _simulate_lif(l2, lengths, W, bias, beta, THRESHOLD, cap, fs)
        metrics[source] = _metrics(y, _predict_from_scores(scores, classes))
        diagnostics[source] = {**_evidence_diagnostics(l2, lengths, W), **diag}

    scores, diag = _simulate_bipolar_lif(
        l2, lengths, W, bias, beta_lif, THRESHOLD, 1, fs
    )
    metrics[SOURCE_BIPOLAR] = _metrics(y, _predict_from_scores(scores, classes))
    diagnostics[SOURCE_BIPOLAR] = {**_evidence_diagnostics(l2, lengths, W), **diag}
    return metrics, diagnostics


def evaluate_one(
    spec: exp72.RunSpec,
    data: exp3.Data,
    config: Config,
    force: bool = False,
) -> dict[str, Any]:
    out_path = _run_path(config.results_dir, spec)
    if out_path.exists() and not force:
        return json.loads(out_path.read_text(encoding="utf-8"))

    base = _base_config(config)
    checkpoint = exp72._path(base.results_dir, "checkpoints", spec, ".pt")
    if not checkpoint.exists():
        raise FileNotFoundError(
            f"Missing Exp7.2 checkpoint for {spec.key}: {checkpoint}. "
            "Exp7.2.2 never retrains the SNN backbone."
        )

    torch.set_num_threads(config.threads)
    device = torch.device(config.device)
    model, checkpoint_payload = exp72.load_model(spec, data, base)
    split_loaders = exp72.loaders(data, spec.seed, config.batch_size, False)
    extracted = {
        split: _extract_l2(model, loader, device)
        for split, loader in split_loaders.items()
    }
    counts = {
        split: (_whole_count(l2, lengths), y)
        for split, (l2, y, lengths) in extracted.items()
    }

    probe = _fit_wholecount_probe(
        counts["train"][0], counts["train"][1],
        counts["val"][0], counts["val"][1],
        counts["test"][0], counts["test"][1],
        exp3.dseed(spec.seed, EXPERIMENT_ID, spec.key, "wholecount_probe"),
        len(data.labels),
    )
    W_probe = probe["W_eff"]
    b_probe = probe["b_eff"]
    classes = probe["classes"]
    beta_lif = math.exp(-(1000.0 / data.fs) / exp72.OUTPUT_TAU_MEM_MS)

    metrics_by_source: dict[str, dict[str, dict[str, float]]] = {source: {} for source in PROBE_SOURCES}
    test_diagnostics: dict[str, dict[str, float]] = {}
    for split, (l2, y, lengths) in extracted.items():
        split_metrics, split_diag = _split_readouts(
            l2, y, lengths, W_probe, b_probe, classes, beta_lif, data.fs
        )
        for source, metric in split_metrics.items():
            metrics_by_source[source][split] = metric
        if split == "test":
            test_diagnostics.update(split_diag)

    # Sanity: analog accumulation must exactly reproduce the fitted probe labels.
    for split in ("train", "val", "test"):
        for metric in METRICS:
            if not np.isclose(
                metrics_by_source[SOURCE_ANALOG][split][metric],
                probe["metrics"][split][metric],
                atol=1e-12,
            ):
                raise RuntimeError(f"Analog/probe mismatch: {spec.key} {split} {metric}")

    native_payload: dict[str, Any] = {}
    if spec.training_family == exp72.FAMILY_E2E:
        if model.output_linear is None:
            raise RuntimeError("E2E checkpoint has no output_linear")
        W_native = model.output_linear.weight.detach().cpu().numpy().astype(np.float64)
        b_native = np.zeros(W_native.shape[0], dtype=np.float64)
        native_metrics = {SOURCE_NATIVE_ANALOG: {}, SOURCE_NATIVE_LIF: {}}
        native_diag: dict[str, dict[str, float]] = {}
        for split, (l2, y, lengths) in extracted.items():
            analog_scores = _analog_scores(l2, lengths, W_native, b_native)
            native_metrics[SOURCE_NATIVE_ANALOG][split] = _metrics(
                y, _predict_from_scores(analog_scores, np.arange(W_native.shape[0]))
            )
            lif_scores, diag = _simulate_lif(
                l2, lengths, W_native, b_native, beta_lif, THRESHOLD, 1, data.fs
            )
            native_metrics[SOURCE_NATIVE_LIF][split] = _metrics(
                y, _predict_from_scores(lif_scores, np.arange(W_native.shape[0]))
            )
            if split == "test":
                native_diag[SOURCE_NATIVE_ANALOG] = _evidence_diagnostics(l2, lengths, W_native)
                native_diag[SOURCE_NATIVE_LIF] = {
                    **_evidence_diagnostics(l2, lengths, W_native), **diag
                }

        # The numpy reconstruction should reproduce the repository's native E2E readout.
        native_repo = {
            split: exp72.evaluate_native(spec, model, loader, device)
            for split, loader in split_loaders.items()
        }
        for split in ("train", "val", "test"):
            for metric in METRICS:
                if not np.isclose(
                    native_metrics[SOURCE_NATIVE_LIF][split][metric],
                    native_repo[split][metric],
                    atol=1e-12,
                ):
                    raise RuntimeError(f"Native LIF reconstruction mismatch: {spec.key} {split} {metric}")
        metrics_by_source.update(native_metrics)
        test_diagnostics.update(native_diag)
        native_payload = {
            "native_weight_shape": list(W_native.shape),
            "native_has_bias": False,
        }

    payload = {
        "experiment_id": EXPERIMENT_ID,
        "protocol_version": PROTOCOL_VERSION,
        "spec": asdict(spec),
        "reuses_exp7_2_checkpoint": True,
        "snn_backbone_retrained": False,
        "backbone_frozen": True,
        "readout_probe_fitted": True,
        "checkpoint": str(checkpoint.relative_to(config.repo_root)),
        "checkpoint_best_epoch": int(checkpoint_payload["best_epoch"]),
        "output_lif": {
            "alpha_syn": 0.0,
            "tau_syn_ms": 0.0,
            "beta_mem": beta_lif,
            "tau_mem_ms": float(exp72.OUTPUT_TAU_MEM_MS),
            "threshold": THRESHOLD,
            "native_cap": 1,
        },
        "probe": {
            "feature_dim": int(probe["feature_dim"]),
            "probe_C": float(probe["probe_C"]),
            "weight_shape": list(W_probe.shape),
            "bias_shape": list(b_probe.shape),
            "scaler_folded_into_affine_readout": True,
            "bias_applied_once_at_sequence_end": True,
        },
        "metrics": metrics_by_source,
        "test_diagnostics": test_diagnostics,
        **native_payload,
    }
    _save_json(out_path, payload)
    return payload


def _performance_rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
    spec = payload["spec"]
    base = {
        "architecture": spec["architecture"],
        "training_family": spec["training_family"],
        "regularization": spec["regularization"],
        "seed": int(spec["seed"]),
    }
    rows: list[dict[str, Any]] = []
    for source, splits in payload["metrics"].items():
        for split, metrics in splits.items():
            rows.append({
                **base,
                "source": source,
                "split": split,
                **{m: float(metrics[m]) for m in METRICS},
            })
    return rows


def _diagnostic_rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
    spec = payload["spec"]
    base = {
        "architecture": spec["architecture"],
        "training_family": spec["training_family"],
        "regularization": spec["regularization"],
        "seed": int(spec["seed"]),
    }
    return [
        {**base, "source": source, **{k: float(v) for k, v in values.items()}}
        for source, values in payload.get("test_diagnostics", {}).items()
    ]


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


def finalize(repo_root: Path) -> dict[str, Any]:
    root = results_dir(repo_root)
    perf_rows: list[dict[str, Any]] = []
    diag_rows: list[dict[str, Any]] = []
    probe_rows: list[dict[str, Any]] = []
    missing: list[str] = []
    for spec in run_specs():
        path = _run_path(root, spec)
        if not path.exists():
            missing.append(spec.key)
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        perf_rows.extend(_performance_rows(payload))
        diag_rows.extend(_diagnostic_rows(payload))
        probe_rows.append({
            "architecture": spec.architecture,
            "training_family": spec.training_family,
            "regularization": spec.regularization,
            "seed": spec.seed,
            "probe_C": float(payload["probe"]["probe_C"]),
            "beta_lif": float(payload["output_lif"]["beta_mem"]),
        })
    if missing:
        raise RuntimeError(f"Missing {len(missing)} Exp7.2.2 runs; first={missing[:8]}")

    perf = pd.DataFrame(perf_rows)
    perf.to_csv(root / "readout_runs.csv", index=False)
    summary = _aggregate(
        perf,
        ["architecture", "training_family", "regularization", "source", "split"],
        list(METRICS),
    )
    summary.to_csv(root / "readout_summary.csv", index=False)

    diagnostics = pd.DataFrame(diag_rows)
    diagnostics.to_csv(root / "diagnostics_runs.csv", index=False)
    diag_value_cols = [
        c for c in diagnostics.columns
        if c not in {"architecture", "training_family", "regularization", "seed", "source"}
    ]
    diag_summary = _aggregate(
        diagnostics,
        ["architecture", "training_family", "regularization", "source"],
        diag_value_cols,
    )
    diag_summary.to_csv(root / "diagnostics_summary.csv", index=False)

    pd.DataFrame(probe_rows).to_csv(root / "probe_parameters_runs.csv", index=False)

    test = perf[perf.split == "test"].copy()
    contrasts = [
        ("analog_minus_if", SOURCE_ANALOG, SOURCE_IF),
        ("if_minus_lif", SOURCE_IF, SOURCE_LIF),
        ("analog_minus_lif", SOURCE_ANALOG, SOURCE_LIF),
        ("multispike_minus_lif", SOURCE_MULTI, SOURCE_LIF),
        ("bipolar_minus_lif", SOURCE_BIPOLAR, SOURCE_LIF),
        ("native_analog_minus_native_lif", SOURCE_NATIVE_ANALOG, SOURCE_NATIVE_LIF),
        ("probe_analog_minus_native_analog", SOURCE_ANALOG, SOURCE_NATIVE_ANALOG),
        ("probe_analog_minus_native_lif", SOURCE_ANALOG, SOURCE_NATIVE_LIF),
    ]
    delta_rows: list[dict[str, Any]] = []
    for (architecture, family, regularization, seed), group in test.groupby(
        ["architecture", "training_family", "regularization", "seed"]
    ):
        indexed = group.set_index("source")
        for contrast, left, right in contrasts:
            if left not in indexed.index or right not in indexed.index:
                continue
            for metric in METRICS:
                delta_rows.append({
                    "contrast": contrast,
                    "architecture": architecture,
                    "training_family": family,
                    "regularization": regularization,
                    "seed": int(seed),
                    "metric": metric,
                    "delta": float(indexed.loc[left, metric] - indexed.loc[right, metric]),
                })
    delta_runs = pd.DataFrame(delta_rows)
    delta_runs.to_csv(root / "paired_delta_runs.csv", index=False)
    delta_summary = _aggregate(
        delta_runs,
        ["contrast", "architecture", "training_family", "regularization", "metric"],
        ["delta"],
    )
    delta_summary.to_csv(root / "paired_deltas.csv", index=False)

    architecture_source = exp72.results_dir(repo_root) / "architecture_table.csv"
    if not architecture_source.exists():
        raise FileNotFoundError(f"Run Exp7.2 finalizer first; missing {architecture_source}")
    pd.read_csv(architecture_source).to_csv(root / "architecture_table.csv", index=False)

    manifest = {
        "experiment_id": EXPERIMENT_ID,
        "protocol_version": PROTOCOL_VERSION,
        "reuses_exp7_2_checkpoints": True,
        "snn_backbone_training_runs": 0,
        "checkpoint_evaluations": len(run_specs()),
        "readout_probe_fit_per_checkpoint": True,
        "probe_sources": list(PROBE_SOURCES),
        "native_sources_e2e_only": list(NATIVE_SOURCES),
        "primary_contrasts": [
            "native_analog_minus_native_lif",
            "probe_analog_minus_native_analog",
            "analog_minus_if",
            "if_minus_lif",
            "multispike_minus_lif",
            "bipolar_minus_lif",
        ],
        "notebook_inputs": [
            "architecture_table.csv",
            "readout_summary.csv",
            "paired_deltas.csv",
            "diagnostics_summary.csv",
        ],
    }
    _save_json(root / "manifest.json", manifest)
    return manifest


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=exp72.BATCH_SIZE)
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run-one")
    run.add_argument("--array-task-id", type=int)
    run.add_argument("--architecture", choices=exp72.ARCHITECTURE_ORDER)
    run.add_argument("--training-family", choices=exp72.TRAINING_FAMILIES)
    run.add_argument("--regularization", choices=exp72.REGULARIZATION_CONDITIONS)
    run.add_argument("--seed", type=int, choices=exp72.TRAIN_SEEDS)
    run.add_argument("--force", action="store_true")

    sub.add_parser("finalize")
    sub.add_parser("list-runs")
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
        for i, spec in enumerate(run_specs()):
            print(i, spec.key)
        return
    if args.command == "finalize":
        print(json.dumps(finalize(repo_root), indent=2))
        return

    specs = run_specs()
    if args.array_task_id is not None:
        if not 0 <= args.array_task_id < len(specs):
            raise ValueError(args.array_task_id)
        spec = specs[args.array_task_id]
    else:
        if None in (args.architecture, args.training_family, args.regularization, args.seed):
            raise ValueError("Specify --array-task-id or all explicit run fields")
        spec = exp72.RunSpec(
            args.architecture, args.training_family, args.regularization, args.seed
        )

    data = exp72.prepare_data(repo_root)
    payload = evaluate_one(spec, data, config, force=args.force)
    print(json.dumps({
        "spec": payload["spec"],
        "probe_C": payload["probe"]["probe_C"],
        "snn_backbone_retrained": payload["snn_backbone_retrained"],
    }, indent=2))


if __name__ == "__main__":
    main()
