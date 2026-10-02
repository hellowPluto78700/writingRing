#!/usr/bin/env python3
"""Exp17.1 post-hoc fixed-group WholeCount probes.

This artifact-only validation distinguishes two explanations for Exp17.1 C1:

1. routing bottleneck: the rescued epoch-20 low group becomes more OOD-decodable,
   but the native main head does not exploit that transferable information;
2. representation bottleneck: train-side selectivity improves, but the fixed low
   group itself does not become more OOD-decodable.

No SNN is retrained. Neuron groups are frozen from the original epoch-20
training-only occupancy ranking. Probe C is selected on validation only.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from core_benchmark_v1.data import loader
from core_benchmark_v1.probes import fit_probe
from core_benchmark_v1.protocol import SPLITS
from core_benchmark_v1.storage import load_torch, save_json
from core_benchmark_v1.training import metrics
from scripts import experiment_17_1_gradient_starvation as exp17_1


GROUPS = ("low", "high")


@dataclass(frozen=True)
class ProbeSpec:
    case: str
    seed: int

    @property
    def key(self) -> str:
        return f"{self.case}__seed{self.seed}"


def probe_specs() -> list[ProbeSpec]:
    return [
        ProbeSpec(case, seed)
        for seed in exp17_1.FORMAL_SEEDS
        for case in exp17_1.CASES
    ]


def _checkpoint_path(config: exp17_1.Config, spec: ProbeSpec) -> Path:
    return config.results_dir / "formal" / spec.key / "checkpoint.pt"


def _bootstrap_groups_path(config: exp17_1.Config, seed: int) -> Path:
    return config.results_dir / "bootstrap" / f"seed{seed}" / "groups.json"


def _load_selected_model_and_groups(
    config: exp17_1.Config,
    spec: ProbeSpec,
):
    p, _, _ = exp17_1._core(config)
    checkpoint = load_torch(_checkpoint_path(config, spec))
    model = exp17_1._make_model(spec.seed, p)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)

    checkpoint_groups = {
        name: np.asarray(checkpoint["groups"][name], dtype=np.int64)
        for name in GROUPS
    }
    bootstrap = json.loads(
        _bootstrap_groups_path(config, spec.seed).read_text(encoding="utf-8")
    )
    bootstrap_groups = {
        name: np.asarray(bootstrap[name], dtype=np.int64)
        for name in GROUPS
    }
    for name in GROUPS:
        if not np.array_equal(checkpoint_groups[name], bootstrap_groups[name]):
            raise AssertionError(
                f"{spec.key}: selected checkpoint {name} group differs from "
                "the frozen epoch-20 bootstrap group"
            )
    return model, bootstrap_groups


def _whole_count_features(
    model: torch.nn.Module,
    arrays: dict[str, np.ndarray],
    p,
    seed: int,
    indices: np.ndarray,
) -> dict[str, np.ndarray]:
    model.eval()
    result: dict[str, np.ndarray] = {}
    index = torch.as_tensor(indices, dtype=torch.long)
    with torch.no_grad():
        for split in SPLITS:
            batches: list[np.ndarray] = []
            for x, _y, lengths in loader(arrays, split, p, seed, shuffle=False):
                out = model(x, lengths)
                # BenchmarkNet pads inactive timesteps with zero, so temporal
                # sum here is exactly the CoreBenchmark WholeCount geometry.
                counts = out["spike"][1].sum(1)
                batches.append(counts[:, index].cpu().numpy())
            result[split] = np.concatenate(batches, axis=0)
            if len(result[split]) != len(arrays[f"{split}_y"]):
                raise AssertionError(f"{split} feature count changed")
    return result


def _fit_group_probe(
    features: dict[str, np.ndarray],
    arrays: dict[str, np.ndarray],
    p,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    scaler, probe, search = fit_probe(
        features["train"],
        arrays["train_y"],
        features["val"],
        arrays["val_y"],
        "no_bias",
        p,
    )
    row: dict[str, Any] = {
        "decoder": "no_bias",
        "aggregation": "whole_count",
        "dimension": int(features["train"].shape[1]),
        "C": float(probe.C),
        "with_mean": False,
    }
    for split in SPLITS:
        prediction = probe.predict(
            scaler.transform(features[split].astype(np.float64))
        )
        split_metrics = metrics(arrays[f"{split}_y"], prediction)
        row.update({
            f"{split}_{name}": value
            for name, value in split_metrics.items()
        })
    row["train_test_gap"] = row["train_ba"] - row["test_ba"]
    row["val_test_gap"] = row["val_ba"] - row["test_ba"]
    return row, search


def run_probe_task(
    config: exp17_1.Config,
    task_id: int,
) -> dict[str, Any]:
    specs = probe_specs()
    if not 0 <= task_id < len(specs):
        raise IndexError(task_id)
    spec = specs[task_id]
    p, _, arrays = exp17_1._core(config)
    model, groups = _load_selected_model_and_groups(config, spec)

    directory = config.results_dir / "posthoc_group_probes" / spec.key
    directory.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    searches: list[dict[str, Any]] = []

    native = json.loads(
        (config.results_dir / "formal" / spec.key / "metrics.json").read_text(
            encoding="utf-8"
        )
    )

    for group in GROUPS:
        features = _whole_count_features(
            model, arrays, p, spec.seed, groups[group]
        )
        row, search = _fit_group_probe(features, arrays, p)
        row.update({
            "case": spec.case,
            "seed": spec.seed,
            "group": group,
            "group_source_epoch": exp17_1.BRANCH_EPOCH,
            "group_source": "train-only epoch-20 occupancy quartile",
            "native_train_ba": native["train"]["ba"],
            "native_val_ba": native["val"]["ba"],
            "native_test_ba": native["test"]["ba"],
            "probe_minus_native_test_ba": (
                row["test_ba"] - native["test"]["ba"]
            ),
        })
        rows.append(row)
        searches.extend({
            "case": spec.case,
            "seed": spec.seed,
            "group": group,
            **candidate,
        } for candidate in search)

    save_json(directory / "probes.json", {"rows": rows})
    save_json(directory / "probe_search.json", {"rows": searches})
    return {"status": "PASS", "run": spec.key, "rows": rows}


def _mean_std(values: list[float]) -> dict[str, float]:
    return {
        "mean": float(np.mean(values)),
        "std": float(np.std(values, ddof=1)),
    }


def aggregate(config: exp17_1.Config) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for spec in probe_specs():
        path = (
            config.results_dir
            / "posthoc_group_probes"
            / spec.key
            / "probes.json"
        )
        if not path.exists():
            raise FileNotFoundError(path)
        rows.extend(json.loads(path.read_text(encoding="utf-8"))["rows"])

    summary: dict[str, Any] = {
        "rows": rows,
        "case_group_summary": {},
        "paired_vs_C0": {},
        "primary_question": (
            "Does C1 improve OOD decodability of the fixed epoch-20 low group "
            "even though native C1 test BA did not improve?"
        ),
    }

    for case in exp17_1.CASES:
        summary["case_group_summary"][case] = {}
        for group in GROUPS:
            chosen = [
                row for row in rows
                if row["case"] == case and row["group"] == group
            ]
            summary["case_group_summary"][case][group] = {
                key: _mean_std([float(row[key]) for row in chosen])
                for key in (
                    "train_ba",
                    "val_ba",
                    "test_ba",
                    "train_test_gap",
                    "probe_minus_native_test_ba",
                )
            }

    by_key = {
        (row["case"], int(row["seed"]), row["group"]): row
        for row in rows
    }
    for case in exp17_1.CASES[1:]:
        summary["paired_vs_C0"][case] = {}
        for group in GROUPS:
            deltas = {}
            for metric in ("train_ba", "val_ba", "test_ba", "train_test_gap"):
                values = [
                    float(by_key[(case, seed, group)][metric])
                    - float(by_key[("C0", seed, group)][metric])
                    for seed in exp17_1.FORMAL_SEEDS
                ]
                deltas[metric] = {
                    "by_seed": {
                        str(seed): value
                        for seed, value in zip(exp17_1.FORMAL_SEEDS, values)
                    },
                    **_mean_std(values),
                }
            summary["paired_vs_C0"][case][group] = deltas

    out = config.results_dir / "posthoc_group_probes" / "aggregate"
    out.mkdir(parents=True, exist_ok=True)
    save_json(out / "summary.json", summary)
    return summary


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path)
    parser.add_argument("--core-results", type=Path)
    sub = parser.add_subparsers(dest="command", required=True)

    probe = sub.add_parser("probe")
    probe.add_argument("--task-id", type=int, required=True)

    sub.add_parser("aggregate")
    return parser


def main() -> None:
    args = _parser().parse_args()
    config = exp17_1.config_from_args(args)
    if args.command == "probe":
        print(json.dumps(run_probe_task(config, args.task_id), indent=2))
    elif args.command == "aggregate":
        print(json.dumps(aggregate(config), indent=2))
    else:
        raise ValueError(args.command)


if __name__ == "__main__":
    main()
