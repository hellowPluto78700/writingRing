from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from scripts import experiment_0_1_general_comparison as exp01
from scripts import experiment_3_0_1_single_tau_objectives as exp3
from scripts import experiment_7_2_3_objective_temporal_dynamics as exp723
from scripts import experiment_7_2_two_layer_tau_training as exp72

EXPERIMENT_ID = "experiment_7_2_4_global_window_eval"
PROTOCOL_VERSION = "global_window_eval_v1"
GLOBAL_WC = "l2_global_wholecount_linear"
GLOBAL_F250 = "l2_global_fixed250_linear"
VALID_WC = "l2_wholecount_linear"
VALID_F250 = "l2_fixed250_linear"
GLOBAL_SOURCES = (GLOBAL_WC, GLOBAL_F250)
ARCHITECTURE_ORDER = exp723.ARCHITECTURE_ORDER
FAMILIES = exp723.FAMILIES
SEEDS = exp723.SEEDS
REGULARIZATIONS = exp723.REGULARIZATIONS
FAMILY_LABELS = {
    exp723.A1: "A1 shared TSCE",
    exp723.A2: "A2 shared WC",
    exp723.F1: "F1 Fixed250 TSCE",
    exp723.F2: "F2 Fixed250 WC",
    exp723.S1: "S1 spike WC, beta=.5",
    exp723.S2: "S2 spike TSCE, beta=.5",
    exp723.S3: "S3 spike WC, beta=1",
    exp723.S4: "S4 spike TSCE, beta=1",
}
TAIL_WINDOWS = (
    (0, 16, "0_250ms"),
    (16, 32, "250_500ms"),
    (32, 48, "500_750ms"),
    (48, 64, "750_1000ms"),
    (64, None, "ge_1000ms"),
)

RunSpec = exp723.RunSpec


@dataclass(frozen=True)
class Config:
    repo_root: Path
    results_dir: Path
    device: str = "cpu"
    batch_size: int = exp72.BATCH_SIZE
    threads: int = 1


def find_repo_root(start: Path | None = None) -> Path:
    return exp723.find_repo_root(start)


def results_dir(repo_root: Path) -> Path:
    return repo_root / "notebooks" / "artifacts" / EXPERIMENT_ID / PROTOCOL_VERSION


def run_specs() -> list[RunSpec]:
    return exp723.all_specs()


def _evaluation_path(root: Path, spec: RunSpec) -> Path:
    return root / "evaluations" / spec.family / f"{spec.key}.json"


def _save_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _base_config(config: Config) -> exp723.Config:
    return exp723.Config(
        repo_root=config.repo_root,
        results_dir=exp723.results_dir(config.repo_root),
        device=config.device,
        batch_size=config.batch_size,
        threads=config.threads,
        max_epochs=exp723.MAX_EPOCHS,
    )


def global_wholecount_features(l2: torch.Tensor) -> np.ndarray:
    """Whole-window L2 spike counts. Intentionally has no lengths argument."""
    if l2.ndim != 3:
        raise ValueError(f"Expected [N,T,D] L2 tensor, got shape {tuple(l2.shape)}")
    return l2.sum(dim=1).numpy()


def global_fixed250_features(l2: torch.Tensor, bin_steps: int) -> np.ndarray:
    """Absolute Fixed250 counts over the complete padded window, without masking."""
    if l2.ndim != 3:
        raise ValueError(f"Expected [N,T,D] L2 tensor, got shape {tuple(l2.shape)}")
    if bin_steps < 1:
        raise ValueError("bin_steps must be positive")
    n, steps, dim = l2.shape
    if steps != exp72.EXPECTED_STEPS:
        raise ValueError(f"Expected {exp72.EXPECTED_STEPS} timesteps, got {steps}")
    if steps % bin_steps != 0:
        raise ValueError(f"Global Fixed250 requires exact bins: T={steps}, bin_steps={bin_steps}")
    return l2.reshape(n, steps // bin_steps, bin_steps, dim).sum(dim=2).flatten(1).numpy()


def _probe_seed(spec: RunSpec, source: str) -> int:
    # Pair each global probe with the exact random seed used by its Exp7.2.3 valid-length counterpart.
    paired_source = VALID_WC if source == GLOBAL_WC else VALID_F250
    return exp3.dseed(spec.seed, exp723.EXPERIMENT_ID, spec.key, paired_source)


def _fit_global_probe(
    source: str,
    splits: dict[str, tuple[torch.Tensor, np.ndarray, np.ndarray]],
    spec: RunSpec,
    bin_steps: int,
) -> dict[str, Any]:
    if source == GLOBAL_WC:
        build = lambda l2: global_wholecount_features(l2)
    elif source == GLOBAL_F250:
        build = lambda l2: global_fixed250_features(l2, bin_steps)
    else:
        raise ValueError(source)
    features = {name: (build(l2), y) for name, (l2, y, _lengths) in splits.items()}
    probe = exp01._fit_linear_probe(
        features["train"][0], features["train"][1],
        features["val"][0], features["val"][1],
        features["test"][0], features["test"][1],
        _probe_seed(spec, source),
    )
    return {
        "source": source,
        "feature_dim": int(probe["feature_dim"]),
        "probe_C": float(probe["probe_C"]),
        "paired_valid_source": VALID_WC if source == GLOBAL_WC else VALID_F250,
        "metrics": {split: probe[split] for split in ("train", "val", "test")},
    }


def _tail_activity(l2: torch.Tensor, lengths: np.ndarray, fs: float) -> dict[str, float]:
    arr = l2.numpy().astype(np.float64, copy=False)
    lengths = np.asarray(lengths, dtype=np.int64)
    n, steps, width = arr.shape
    positions = np.arange(steps)[None, :]
    valid = positions < lengths[:, None]
    valid_spikes = float((arr * valid[..., None]).sum())
    tail_spikes = float((arr * (~valid)[..., None]).sum())
    total = valid_spikes + tail_spikes
    valid_steps = int(lengths.sum())
    tail_steps = int((steps - lengths).sum())
    valid_seconds = valid_steps / float(fs)
    tail_seconds = tail_steps / float(fs)
    return {
        "n_samples": float(n),
        "valid_spikes": valid_spikes,
        "tail_spikes": tail_spikes,
        "tail_fraction": tail_spikes / total if total > 0 else 0.0,
        "valid_spikes_per_neuron_second": valid_spikes / (width * valid_seconds) if valid_seconds > 0 else 0.0,
        "tail_spikes_per_neuron_second": tail_spikes / (width * tail_seconds) if tail_seconds > 0 else 0.0,
        "valid_steps": float(valid_steps),
        "tail_steps": float(tail_steps),
    }


def _tail_offset_activity(l2: torch.Tensor, lengths: np.ndarray, fs: float) -> list[dict[str, Any]]:
    arr = l2.numpy().astype(np.float64, copy=False)
    lengths = np.asarray(lengths, dtype=np.int64)
    _n, steps, width = arr.shape
    rows: list[dict[str, Any]] = []
    for start, stop, label in TAIL_WINDOWS:
        spike_count = 0.0
        available_steps = 0
        for sample_index, length in enumerate(lengths.tolist()):
            left = min(steps, int(length) + int(start))
            right = steps if stop is None else min(steps, int(length) + int(stop))
            if right <= left:
                continue
            spike_count += float(arr[sample_index, left:right].sum())
            available_steps += right - left
        seconds = available_steps / float(fs)
        rows.append({
            "offset": label,
            "offset_start_steps": int(start),
            "offset_stop_steps": None if stop is None else int(stop),
            "spike_count": spike_count,
            "available_steps": int(available_steps),
            "spikes_per_neuron_second": spike_count / (width * seconds) if seconds > 0 else 0.0,
        })
    return rows


def evaluate_one(spec: RunSpec, data: exp3.Data, config: Config, force: bool = False) -> dict[str, Any]:
    destination = _evaluation_path(config.results_dir, spec)
    if destination.exists() and not force:
        return json.loads(destination.read_text(encoding="utf-8"))
    torch.set_num_threads(config.threads)
    device = torch.device(config.device)
    model, checkpoint_payload, reused = exp723._load_any_model(spec, data, _base_config(config))
    loaders = exp723._loaders(data, spec, config.batch_size, False)
    splits = {name: exp723._extract_l2(model, loader, device) for name, loader in loaders.items()}
    probes = {source: _fit_global_probe(source, splits, spec, data.bin_steps) for source in GLOBAL_SOURCES}
    tail = {name: _tail_activity(l2, lengths, data.fs) for name, (l2, _y, lengths) in splits.items()}
    tail_offsets = {
        name: _tail_offset_activity(l2, lengths, data.fs)
        for name, (l2, _y, lengths) in splits.items()
    }
    payload = {
        "experiment_id": EXPERIMENT_ID,
        "protocol_version": PROTOCOL_VERSION,
        "source_experiment_id": exp723.EXPERIMENT_ID,
        "spec": asdict(spec),
        "reuses_exp7_2_checkpoint": bool(reused),
        "source_best_epoch": int(checkpoint_payload["best_epoch"]),
        "global_probes": probes,
        "tail_activity": tail,
        "tail_offset_activity": tail_offsets,
    }
    _save_json(destination, payload)
    return payload


def _aggregate(df: pd.DataFrame, groups: list[str], values: list[str]) -> pd.DataFrame:
    grouped = df.groupby(groups, dropna=False)[values]
    return pd.concat([
        grouped.size().rename("n"),
        grouped.mean().add_suffix("_mean"),
        grouped.std(ddof=1).fillna(0).add_suffix("_std"),
    ], axis=1).reset_index()


def _global_probe_rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
    spec = payload["spec"]
    base = {
        "architecture": spec["architecture"],
        "family": spec["family"],
        "regularization": spec["regularization"],
        "seed": int(spec["seed"]),
    }
    rows: list[dict[str, Any]] = []
    for source, probe in payload["global_probes"].items():
        for split, metrics in probe["metrics"].items():
            rows.append({
                **base,
                "source": source,
                "split": split,
                "feature_dim": int(probe["feature_dim"]),
                "probe_C": float(probe["probe_C"]),
                "accuracy": float(metrics["accuracy"]),
                "balanced_accuracy": float(metrics["balanced_accuracy"]),
                "macro_f1": float(metrics["macro_f1"]),
            })
    return rows


def _tail_rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
    spec = payload["spec"]
    base = {
        "architecture": spec["architecture"],
        "family": spec["family"],
        "regularization": spec["regularization"],
        "seed": int(spec["seed"]),
    }
    return [{**base, "split": split, **metrics} for split, metrics in payload["tail_activity"].items()]


def _tail_offset_rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
    spec = payload["spec"]
    base = {
        "architecture": spec["architecture"],
        "family": spec["family"],
        "regularization": spec["regularization"],
        "seed": int(spec["seed"]),
    }
    rows: list[dict[str, Any]] = []
    for split, offset_rows in payload["tail_offset_activity"].items():
        for row in offset_rows:
            rows.append({**base, "split": split, **row})
    return rows


def _load_exp723_runs(repo_root: Path) -> pd.DataFrame:
    path = exp723.results_dir(repo_root) / "performance_runs.csv"
    if not path.exists():
        raise FileNotFoundError(f"Exp7.2.4 requires finalized Exp7.2.3 runs: {path}")
    runs = pd.read_csv(path)
    expected = {"architecture", "family", "regularization", "seed", "source", "split",
                "accuracy", "balanced_accuracy", "macro_f1"}
    missing = expected.difference(runs.columns)
    if missing:
        raise RuntimeError(f"Exp7.2.3 performance_runs.csv missing columns: {sorted(missing)}")
    return runs


def _paired_delta_rows(old_runs: pd.DataFrame, global_runs: pd.DataFrame) -> pd.DataFrame:
    old = old_runs[(old_runs["split"] == "test") & old_runs["source"].isin([VALID_WC, VALID_F250])]
    new = global_runs[global_runs["split"] == "test"]
    combined = pd.concat([old, new], ignore_index=True, sort=False)
    keys = ["architecture", "family", "regularization", "seed"]
    contrasts = (
        ("valid_phase_gain", VALID_F250, VALID_WC),
        ("global_phase_gain", GLOBAL_F250, GLOBAL_WC),
        ("wc_global_minus_valid", GLOBAL_WC, VALID_WC),
        ("f250_global_minus_valid", GLOBAL_F250, VALID_F250),
    )
    rows: list[dict[str, Any]] = []
    for key, group in combined.groupby(keys):
        indexed = group.set_index("source")
        for contrast, left, right in contrasts:
            if left not in indexed.index or right not in indexed.index:
                raise RuntimeError(f"Missing sources for {key}: {contrast} requires {left}, {right}")
            for metric in ("accuracy", "balanced_accuracy", "macro_f1"):
                rows.append({
                    "contrast": contrast,
                    "architecture": key[0],
                    "family": key[1],
                    "regularization": key[2],
                    "seed": int(key[3]),
                    "metric": metric,
                    "delta": float(indexed.loc[left, metric] - indexed.loc[right, metric]),
                })
    return pd.DataFrame(rows)


def _report_table(old_runs: pd.DataFrame, global_runs: pd.DataFrame, regularization: str) -> pd.DataFrame:
    old = old_runs[(old_runs["split"] == "test") & (old_runs["regularization"] == regularization)]
    new = global_runs[(global_runs["split"] == "test") & (global_runs["regularization"] == regularization)]
    rows: list[dict[str, Any]] = []
    for family in FAMILIES:
        def mean_ba(frame: pd.DataFrame, source: str) -> float:
            selected = frame[(frame["family"] == family) & (frame["source"] == source)]
            expected_n = len(ARCHITECTURE_ORDER) * len(SEEDS)
            if len(selected) != expected_n:
                raise RuntimeError(
                    f"Expected {expected_n} {regularization}/{family}/{source} rows, got {len(selected)}"
                )
            return float(selected["balanced_accuracy"].mean())

        native = mean_ba(old, "native")
        wc_valid = mean_ba(old, VALID_WC)
        f250_valid = mean_ba(old, VALID_F250)
        wc_global = mean_ba(new, GLOBAL_WC)
        f250_global = mean_ba(new, GLOBAL_F250)
        rows.append({
            "family": family,
            "family_label": FAMILY_LABELS[family],
            "regularization": regularization,
            "n": len(ARCHITECTURE_ORDER) * len(SEEDS),
            "native_ba_valid": native,
            "l2_wc_valid_ba": wc_valid,
            "l2_f250_valid_ba": f250_valid,
            "valid_phase_gain_pp": 100.0 * (f250_valid - wc_valid),
            "l2_wc_global_ba": wc_global,
            "l2_f250_global_ba": f250_global,
            "global_phase_gain_pp": 100.0 * (f250_global - wc_global),
            "wc_global_minus_valid_pp": 100.0 * (wc_global - wc_valid),
            "f250_global_minus_valid_pp": 100.0 * (f250_global - f250_valid),
        })
    return pd.DataFrame(rows)


def finalize(repo_root: Path) -> dict[str, Any]:
    root = results_dir(repo_root)
    global_rows: list[dict[str, Any]] = []
    tail_rows: list[dict[str, Any]] = []
    offset_rows: list[dict[str, Any]] = []
    missing: list[str] = []
    for spec in run_specs():
        path = _evaluation_path(root, spec)
        if not path.exists():
            missing.append(spec.key)
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        global_rows.extend(_global_probe_rows(payload))
        tail_rows.extend(_tail_rows(payload))
        offset_rows.extend(_tail_offset_rows(payload))
    if missing:
        raise RuntimeError(f"Missing {len(missing)} Exp7.2.4 evaluations; first={missing[:8]}")

    global_runs = pd.DataFrame(global_rows)
    global_runs.to_csv(root / "global_window_probe_runs.csv", index=False)
    global_summary = _aggregate(
        global_runs,
        ["architecture", "family", "regularization", "source", "split"],
        ["accuracy", "balanced_accuracy", "macro_f1"],
    )
    global_summary.to_csv(root / "global_window_probe_summary.csv", index=False)

    tail = pd.DataFrame(tail_rows)
    tail.to_csv(root / "l2_tail_activity_runs.csv", index=False)
    _aggregate(
        tail,
        ["architecture", "family", "regularization", "split"],
        ["tail_fraction", "valid_spikes_per_neuron_second", "tail_spikes_per_neuron_second"],
    ).to_csv(root / "l2_tail_activity_summary.csv", index=False)

    offsets = pd.DataFrame(offset_rows)
    offsets.to_csv(root / "l2_tail_offset_runs.csv", index=False)
    _aggregate(
        offsets,
        ["architecture", "family", "regularization", "split", "offset"],
        ["spikes_per_neuron_second"],
    ).to_csv(root / "l2_tail_offset_summary.csv", index=False)

    old_runs = _load_exp723_runs(repo_root)
    paired = _paired_delta_rows(old_runs, global_runs)
    paired.to_csv(root / "global_vs_valid_delta_runs.csv", index=False)
    _aggregate(
        paired,
        ["contrast", "architecture", "family", "regularization", "metric"],
        ["delta"],
    ).to_csv(root / "global_vs_valid_delta_summary.csv", index=False)

    report_files: list[str] = []
    for regularization in REGULARIZATIONS:
        report = _report_table(old_runs, global_runs, regularization)
        path = root / f"report_table_{regularization}.csv"
        report.to_csv(path, index=False)
        report_files.append(path.name)

    manifest = {
        "experiment_id": EXPERIMENT_ID,
        "protocol_version": PROTOCOL_VERSION,
        "source_experiment_id": exp723.EXPERIMENT_ID,
        "source_protocol_version": exp723.PROTOCOL_VERSION,
        "architectures": list(ARCHITECTURE_ORDER),
        "families": list(FAMILIES),
        "seeds": list(SEEDS),
        "regularizations": list(REGULARIZATIONS),
        "training_runs": 0,
        "checkpoint_evaluations": len(run_specs()),
        "global_probe_sources": list(GLOBAL_SOURCES),
        "global_feature_contract": "Full 256-step L2 trajectory; no valid-length mask for WC or Fixed250 features",
        "notebook_inputs": [
            "global_window_probe_summary.csv",
            "global_vs_valid_delta_summary.csv",
            "l2_tail_activity_summary.csv",
            "l2_tail_offset_summary.csv",
            *report_files,
        ],
    }
    _save_json(root / "manifest.json", manifest)
    return manifest


def _select(array_task_id: int | None, args: argparse.Namespace) -> RunSpec:
    specs = run_specs()
    if array_task_id is not None:
        if not 0 <= array_task_id < len(specs):
            raise ValueError(array_task_id)
        return specs[array_task_id]
    if None in (args.architecture, args.family, args.regularization, args.seed):
        raise ValueError("Specify --array-task-id or all explicit run fields")
    return RunSpec(args.architecture, args.family, args.regularization, args.seed)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=exp72.BATCH_SIZE)
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run")
    run.add_argument("--array-task-id", type=int)
    run.add_argument("--architecture", choices=ARCHITECTURE_ORDER)
    run.add_argument("--family", choices=FAMILIES)
    run.add_argument("--regularization", choices=REGULARIZATIONS)
    run.add_argument("--seed", type=int, choices=SEEDS)
    run.add_argument("--force", action="store_true")
    sub.add_parser("list")
    sub.add_parser("finalize")
    return parser


def main() -> None:
    args = _parser().parse_args()
    repo_root = find_repo_root(args.repo_root)
    config = Config(repo_root, results_dir(repo_root), args.device, args.batch_size, args.threads)
    if args.command == "list":
        for index, spec in enumerate(run_specs()):
            print(index, spec.key)
        return
    if args.command == "finalize":
        print(json.dumps(finalize(repo_root), indent=2))
        return
    data = exp72.prepare_data(repo_root)
    spec = _select(args.array_task_id, args)
    payload = evaluate_one(spec, data, config, args.force)
    print(json.dumps({"spec": payload["spec"], "source_best_epoch": payload["source_best_epoch"]}, indent=2))


if __name__ == "__main__":
    main()
