from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression

from scripts import experiment_4_0_fixed250_temporal_snn as exp40
from scripts import experiment_4_0_1_multispike_macro_lif as exp401


EXPERIMENT_ID = "experiment_4_0_4_hidden_state_probe"
PROTOCOL_VERSION = "rsnn_cap_variant_endpoint_state_probe_v1"
SOURCE_ARCHITECTURE = "rsnn"
SOURCE_HIDDEN_WIDTH = 128
SOURCE_TAU_MEM_MS = 250.0
SEEDS = exp40.SEEDS
VARIANTS: tuple[tuple[str, int, int], ...] = exp401.VARIANTS
EXPECTED_RUNS = len(VARIANTS) * len(SEEDS)
BATCH_SIZE = exp401.BATCH_SIZE
PROBE_MAX_ITER = exp40.LOGREG_MAX_ITER
FEATURE_DESCRIPTION = (
    "post-reset hidden membrane at the causal valid endpoint; one 128-D vector per gesture"
)


@dataclass(frozen=True)
class ProbeSpec:
    variant: str
    hidden_cap: int
    output_cap: int
    seed: int

    @property
    def key(self) -> str:
        return (
            f"rsnn__h{SOURCE_HIDDEN_WIDTH}__tau{int(SOURCE_TAU_MEM_MS)}ms__"
            f"{self.variant}__hcap{self.hidden_cap}__ocap{self.output_cap}__seed{self.seed}"
        )


def results_dir(repo_root: Path) -> Path:
    return repo_root / "notebooks" / "artifacts" / EXPERIMENT_ID / PROTOCOL_VERSION


def run_specs() -> list[ProbeSpec]:
    return [
        ProbeSpec(variant, hidden_cap, output_cap, seed)
        for variant, hidden_cap, output_cap in VARIANTS
        for seed in SEEDS
    ]


def source_spec(spec: ProbeSpec) -> exp401.RunSpec:
    return exp401.RunSpec(
        architecture=SOURCE_ARCHITECTURE,
        hidden_width=SOURCE_HIDDEN_WIDTH,
        tau_mem_ms=SOURCE_TAU_MEM_MS,
        variant=spec.variant,
        hidden_cap=spec.hidden_cap,
        output_cap=spec.output_cap,
        seed=spec.seed,
    )


def probe_path(root: Path, spec: ProbeSpec) -> Path:
    return root / "probes" / f"{spec.key}.json"


def _save_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _assert_checkpoint_identity(
    checkpoint: dict[str, object],
    spec: ProbeSpec,
) -> None:
    stored = checkpoint.get("spec")
    if not isinstance(stored, dict):
        raise ValueError(f"Source checkpoint {spec.key} has no spec dictionary")
    expected = source_spec(spec)
    fields = (
        "architecture",
        "hidden_width",
        "tau_mem_ms",
        "variant",
        "hidden_cap",
        "output_cap",
        "seed",
    )
    for field in fields:
        actual_value = stored.get(field)
        expected_value = getattr(expected, field)
        if field == "tau_mem_ms":
            if not np.isclose(float(actual_value), float(expected_value)):
                raise ValueError(
                    f"Checkpoint identity mismatch for {spec.key}: {field}="
                    f"{actual_value!r}, expected {expected_value!r}"
                )
        elif actual_value != expected_value:
            raise ValueError(
                f"Checkpoint identity mismatch for {spec.key}: {field}="
                f"{actual_value!r}, expected {expected_value!r}"
            )


def load_frozen_model(
    spec: ProbeSpec,
    repo_root: Path,
    n_classes: int,
    device: torch.device,
) -> tuple[exp401.MacroTemporalDecoder, Path, dict[str, object]]:
    source_root = exp401.results_dir(repo_root)
    checkpoint_file = exp401.checkpoint_path(source_root, source_spec(spec))
    if not checkpoint_file.exists():
        raise FileNotFoundError(f"Missing Exp4.0.1 checkpoint: {checkpoint_file}")
    checkpoint = torch.load(checkpoint_file, map_location=device, weights_only=False)
    if not isinstance(checkpoint, dict):
        raise ValueError(f"Unexpected checkpoint payload: {checkpoint_file}")
    _assert_checkpoint_identity(checkpoint, spec)
    state_dict = checkpoint.get("model_state_dict")
    if not isinstance(state_dict, dict):
        raise ValueError(f"Missing model_state_dict: {checkpoint_file}")

    model = exp401.MacroTemporalDecoder(
        architecture=SOURCE_ARCHITECTURE,
        hidden_width=SOURCE_HIDDEN_WIDTH,
        tau_mem_ms=SOURCE_TAU_MEM_MS,
        n_classes=n_classes,
        hidden_cap=spec.hidden_cap,
        output_cap=spec.output_cap,
    ).to(device)
    model.load_state_dict(state_dict)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model, checkpoint_file, checkpoint


def extract_hidden_endpoint_states(
    model: exp401.MacroTemporalDecoder,
    data_loader: torch.utils.data.DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    """Return one post-reset hidden membrane vector at each valid endpoint.

    This intentionally bypasses the output projection and output LIF. The
    feature contains no phase-wise flattening: every gesture contributes one
    H-dimensional vector selected causally at valid_bins - 1.
    """
    state_parts: list[np.ndarray] = []
    label_parts: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for X, y, valid_bins in data_loader:
            X = X.to(device)
            valid_bins = valid_bins.to(device)
            batch, n_bins, _ = X.shape
            hidden_mem = torch.zeros(
                batch,
                model.hidden_width,
                device=device,
                dtype=X.dtype,
            )
            prev_hidden_spikes = torch.zeros_like(hidden_mem)
            hidden_memories: list[torch.Tensor] = []
            for b in range(n_bins):
                current = model.input_hidden(X[:, b])
                if model.recurrent is not None:
                    current = current + model.recurrent(prev_hidden_spikes)
                hidden_spikes, hidden_mem, _ = model.hidden_lif(current, hidden_mem)
                hidden_memories.append(hidden_mem)
                prev_hidden_spikes = hidden_spikes
            hidden_sequence = torch.stack(hidden_memories, dim=1)
            endpoint = exp40.valid_final_membrane(hidden_sequence, valid_bins)
            state_parts.append(endpoint.cpu().numpy())
            label_parts.append(y.numpy())

    states = np.concatenate(state_parts)
    labels = np.concatenate(label_parts)
    if states.shape[1] != SOURCE_HIDDEN_WIDTH:
        raise ValueError(
            f"Expected {SOURCE_HIDDEN_WIDTH}-D endpoint state, got {states.shape}"
        )
    if not np.isfinite(states).all():
        raise ValueError("Endpoint-state probe features contain non-finite values")
    return states, labels


def make_eval_loaders(
    data: exp40.BinnedData,
    batch_size: int,
    spec: ProbeSpec,
) -> list[torch.utils.data.DataLoader]:
    partitions = (
        (data.Xtr, data.ytr, data.btr),
        (data.Xva, data.yva, data.bva),
        (data.Xte, data.yte, data.bte),
    )
    return [
        exp40.loader(
            *partition,
            batch_size,
            False,
            exp40.base.dseed(spec.seed, "exp4_0_4_probe_loader", split),
        )
        for partition, split in zip(
            partitions,
            ("train", "val", "test"),
            strict=True,
        )
    ]


def fit_endpoint_state_probe(
    model: exp401.MacroTemporalDecoder,
    loaders: list[torch.utils.data.DataLoader],
    device: torch.device,
) -> dict[str, object]:
    extracted = [
        extract_hidden_endpoint_states(model, loader, device) for loader in loaders
    ]
    Xtr, ytr = extracted[0]
    probe = LogisticRegression(
        max_iter=PROBE_MAX_ITER,
        class_weight="balanced",
        solver="lbfgs",
    )
    probe.fit(Xtr, ytr)

    metrics_by_split: dict[str, object] = {}
    for split, (X, y) in zip(("train", "val", "test"), extracted, strict=True):
        metrics_by_split[split] = exp40.metrics(y, probe.predict(X))
    return {
        "feature": FEATURE_DESCRIPTION,
        "feature_dim": int(Xtr.shape[1]),
        "temporal_phase_access": False,
        "output_neuron_access": False,
        "source_snn_frozen": True,
        "classifier": "balanced LogisticRegression(lbfgs)",
        "metrics": metrics_by_split,
    }


def run_one(
    spec: ProbeSpec,
    data: exp40.BinnedData,
    repo_root: Path,
    root: Path,
    device: torch.device,
    batch_size: int,
    threads: int,
    force: bool,
) -> dict[str, object]:
    destination = probe_path(root, spec)
    if destination.exists() and not force:
        return json.loads(destination.read_text(encoding="utf-8"))

    torch.set_num_threads(threads)
    model, source_checkpoint, checkpoint = load_frozen_model(
        spec,
        repo_root,
        len(data.labels),
        device,
    )
    loaders = make_eval_loaders(data, batch_size, spec)
    probe = fit_endpoint_state_probe(model, loaders, device)
    payload: dict[str, object] = {
        "experiment_id": EXPERIMENT_ID,
        "protocol_version": PROTOCOL_VERSION,
        "spec": spec.__dict__,
        "source": {
            "experiment_id": exp401.EXPERIMENT_ID,
            "protocol_version": exp401.PROTOCOL_VERSION,
            "architecture": SOURCE_ARCHITECTURE,
            "hidden_width": SOURCE_HIDDEN_WIDTH,
            "tau_mem_ms": SOURCE_TAU_MEM_MS,
            "checkpoint": str(source_checkpoint.relative_to(repo_root)),
            "best_epoch": int(checkpoint.get("best_epoch", -1)),
        },
        "input_information_contract": (
            "same scaled Fixed250 vectors as Exp4.0.1; no raw/sub-bin timing added"
        ),
        "probe": probe,
        "provenance": {
            "split_seed": int(exp40.base.SPLIT_SEED),
            "train_users": data.split["train_users"],
            "val_users": data.split["val_users"],
            "test_users": data.split["test_users"],
            "labels": data.labels,
            "event_channels": exp40.EVENT_CHANNELS,
            "fixed_bin_ms": exp40.FIXED_MS,
            "channel_scale": data.channel_scale.tolist(),
            "state_dynamics": (
                "replay frozen Exp4.0.1 hidden dynamics through all padded bins; "
                "select hidden post-reset membrane at causal valid endpoint"
            ),
            "probe_training": "probe fits train endpoint states only; val/test are evaluation only",
        },
    }
    _save_json(destination, payload)
    return payload


def _probe_rows(root: Path) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for spec in run_specs():
        path = probe_path(root, spec)
        if not path.exists():
            raise FileNotFoundError(f"Missing required probe artifact: {path}")
        payload = json.loads(path.read_text(encoding="utf-8"))
        probe = payload["probe"]
        for split in ("train", "val", "test"):
            metrics = probe["metrics"][split]
            rows.append(
                {
                    "architecture": SOURCE_ARCHITECTURE,
                    "hidden_width": SOURCE_HIDDEN_WIDTH,
                    "tau_mem_ms": SOURCE_TAU_MEM_MS,
                    "variant": spec.variant,
                    "hidden_cap": spec.hidden_cap,
                    "output_cap": spec.output_cap,
                    "seed": spec.seed,
                    "split": split,
                    "state_probe_feature_dim": int(probe["feature_dim"]),
                    "state_probe_accuracy": float(metrics["accuracy"]),
                    "state_probe_balanced_accuracy": float(
                        metrics["balanced_accuracy"]
                    ),
                    "state_probe_macro_f1": float(metrics["macro_f1"]),
                }
            )
    return pd.DataFrame(rows)


def _source_rows(repo_root: Path) -> pd.DataFrame:
    source_file = exp401.results_dir(repo_root) / "runs.csv"
    if not source_file.exists():
        raise FileNotFoundError(f"Missing finalized Exp4.0.1 runs: {source_file}")
    source = pd.read_csv(source_file)
    source = source[
        (source["architecture"] == SOURCE_ARCHITECTURE)
        & (source["hidden_width"] == SOURCE_HIDDEN_WIDTH)
        & np.isclose(source["tau_mem_ms"], SOURCE_TAU_MEM_MS)
        & source["variant"].isin([variant for variant, _, _ in VARIANTS])
    ].copy()
    expected_rows = EXPECTED_RUNS * 3
    if len(source) != expected_rows:
        raise ValueError(
            f"Expected {expected_rows} source rows for RSNN cap variants, got {len(source)}"
        )
    expected_caps = {
        variant: (hidden_cap, output_cap)
        for variant, hidden_cap, output_cap in VARIANTS
    }
    for row in source.itertuples(index=False):
        expected_hidden, expected_output = expected_caps[str(row.variant)]
        if int(row.hidden_cap) != expected_hidden or int(row.output_cap) != expected_output:
            raise ValueError(
                f"Unexpected cap identity for {row.variant}: "
                f"({row.hidden_cap},{row.output_cap})"
            )
    return source


def _flatten_summary_columns(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out.columns = [
        "_".join(str(part) for part in column if str(part))
        if isinstance(column, tuple)
        else str(column)
        for column in out.columns
    ]
    return out


def _write_summary(runs: pd.DataFrame, root: Path) -> None:
    test = runs[runs["split"] == "test"].copy()
    test["probe_minus_fully_spiking_ba"] = (
        test["state_probe_balanced_accuracy"]
        - test["valid_count_balanced_accuracy"]
    )
    test["probe_minus_fully_spiking_macro_f1"] = (
        test["state_probe_macro_f1"] - test["valid_count_macro_f1"]
    )
    summary = (
        test.groupby(["variant", "hidden_cap", "output_cap"])[
            [
                "valid_count_balanced_accuracy",
                "valid_count_macro_f1",
                "state_probe_balanced_accuracy",
                "state_probe_macro_f1",
                "probe_minus_fully_spiking_ba",
                "probe_minus_fully_spiking_macro_f1",
            ]
        ]
        .agg(["mean", "std"])
        .reset_index()
    )
    _flatten_summary_columns(summary).to_csv(root / "summary.csv", index=False)

    binary = test[test["variant"] == "binary"]
    if len(binary) != len(SEEDS):
        raise ValueError("Binary reference must contain exactly five test rows")
    binary_by_seed = {
        int(row.seed): row for row in binary.itertuples(index=False)
    }
    effect_rows: list[dict[str, object]] = []
    for row in test.itertuples(index=False):
        ref = binary_by_seed[int(row.seed)]
        effect_rows.append(
            {
                "variant": str(row.variant),
                "hidden_cap": int(row.hidden_cap),
                "output_cap": int(row.output_cap),
                "seed": int(row.seed),
                "fully_spiking_ba": float(row.valid_count_balanced_accuracy),
                "state_probe_ba": float(row.state_probe_balanced_accuracy),
                "probe_minus_fully_spiking_ba": float(
                    row.state_probe_balanced_accuracy
                    - row.valid_count_balanced_accuracy
                ),
                "fully_spiking_delta_vs_binary": float(
                    row.valid_count_balanced_accuracy
                    - ref.valid_count_balanced_accuracy
                ),
                "state_probe_delta_vs_binary": float(
                    row.state_probe_balanced_accuracy
                    - ref.state_probe_balanced_accuracy
                ),
            }
        )
    pd.DataFrame(effect_rows).to_csv(root / "paired_effects.csv", index=False)


def finalize(repo_root: Path, root: Path) -> None:
    missing = [str(probe_path(root, spec)) for spec in run_specs() if not probe_path(root, spec).exists()]
    if missing:
        preview = "\n".join(missing[:10])
        raise FileNotFoundError(
            f"Cannot finalize: {len(missing)} required probe artifacts are missing.\n{preview}"
        )

    probe = _probe_rows(root)
    source = _source_rows(repo_root)
    merge_keys = [
        "architecture",
        "hidden_width",
        "tau_mem_ms",
        "variant",
        "hidden_cap",
        "output_cap",
        "seed",
        "split",
    ]
    source_columns = merge_keys + [
        "best_epoch",
        "parameter_count",
        "valid_count_accuracy",
        "valid_count_balanced_accuracy",
        "valid_count_macro_f1",
        "valid_membrane_accuracy",
        "valid_membrane_balanced_accuracy",
        "valid_membrane_macro_f1",
        "hidden_tail_event_fraction",
        "output_tail_event_fraction",
        "hidden_mean_events_per_neuron_step",
        "output_mean_events_per_neuron_step",
    ]
    runs = probe.merge(
        source[source_columns],
        on=merge_keys,
        how="inner",
        validate="one_to_one",
    )
    expected_rows = EXPECTED_RUNS * 3
    if len(runs) != expected_rows:
        raise ValueError(f"Expected {expected_rows} merged rows, got {len(runs)}")

    root.mkdir(parents=True, exist_ok=True)
    runs.to_csv(root / "runs.csv", index=False)
    _write_summary(runs, root)
    _save_json(
        root / "provenance.json",
        {
            "experiment_id": EXPERIMENT_ID,
            "protocol_version": PROTOCOL_VERSION,
            "source_experiment": exp401.EXPERIMENT_ID,
            "source_protocol": exp401.PROTOCOL_VERSION,
            "source_architecture": SOURCE_ARCHITECTURE,
            "source_hidden_width": SOURCE_HIDDEN_WIDTH,
            "source_tau_mem_ms": SOURCE_TAU_MEM_MS,
            "variants": [
                {
                    "variant": variant,
                    "hidden_cap": hidden_cap,
                    "output_cap": output_cap,
                }
                for variant, hidden_cap, output_cap in VARIANTS
            ],
            "seeds": list(SEEDS),
            "expected_probe_tasks": EXPECTED_RUNS,
            "feature": FEATURE_DESCRIPTION,
            "temporal_phase_access": False,
            "output_neuron_access": False,
            "source_snn_training": "none; all SNN checkpoints are frozen Exp4.0.1 artifacts",
            "scientific_question": (
                "Does poor fully-spiking performance, especially Multi-H with output cap 1, "
                "reflect poor recurrent hidden memory or information loss after that hidden state?"
            ),
        },
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Experiment 4.0.4 frozen Exp4.0.1 hidden endpoint-state probes"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run-one")
    run.add_argument("--array-task-id", type=int, required=True)
    run.add_argument("--device", default="cpu")
    run.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    run.add_argument("--threads", type=int, default=1)
    run.add_argument("--force", action="store_true")

    sub.add_parser("finalize")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    repo_root = exp40.find_repo_root()
    root = results_dir(repo_root)

    if args.command == "finalize":
        finalize(repo_root, root)
        print(root / "summary.csv")
        return

    specs = run_specs()
    if args.array_task_id < 0 or args.array_task_id >= len(specs):
        raise IndexError(
            f"array-task-id {args.array_task_id} outside [0,{len(specs) - 1}]"
        )
    data = exp40.prepare_binned_data(repo_root)
    spec = specs[args.array_task_id]
    payload = run_one(
        spec=spec,
        data=data,
        repo_root=repo_root,
        root=root,
        device=torch.device(args.device),
        batch_size=args.batch_size,
        threads=args.threads,
        force=args.force,
    )
    test_ba = payload["probe"]["metrics"]["test"]["balanced_accuracy"]
    print(
        json.dumps(
            {
                "completed": spec.key,
                "test_state_probe_ba": test_ba,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
