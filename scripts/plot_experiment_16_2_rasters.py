#!/usr/bin/env python3
"""Plot Exp16.2 L1/L2 spike rasters over the full padded time window.

The script consumes the saved Exp16.2 traces.npz artifacts. It does not
rerun the model. By default it chooses one representative test sample (the
sample whose valid length is closest to the test-set median) and uses that
same sample for every case/seed so that rasters are directly comparable.

Each output figure contains:
  row 1: L1 spike raster
  row 2: L2 spike raster

The x-axis always spans the complete padded window. A dashed vertical line
marks the valid/padding boundary; timesteps after that boundary remain visible.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from scripts import experiment_16_2_matched_budget_selective_write as exp16_2
from scripts import experiment_16_prefix_supervised_selective_memory as exp16


DEFAULT_CASES = ("C0", "GZ0", "S90", "G90", "S70", "G70", "S50", "G50")


def _parse_cases(value: str) -> tuple[str, ...]:
    cases = tuple(part.strip() for part in value.split(",") if part.strip())
    unknown = sorted(set(cases) - set(DEFAULT_CASES))
    if unknown:
        raise argparse.ArgumentTypeError(
            f"Unknown case(s): {', '.join(unknown)}; "
            f"valid cases: {', '.join(DEFAULT_CASES)}"
        )
    if not cases:
        raise argparse.ArgumentTypeError("At least one case is required")
    return cases


def _parse_seeds(value: str) -> tuple[int, ...]:
    try:
        seeds = tuple(int(part.strip()) for part in value.split(",") if part.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Seeds must be comma-separated integers") from exc
    unknown = sorted(set(seeds) - set(exp16_2.FORMAL_SEEDS))
    if unknown:
        raise argparse.ArgumentTypeError(
            f"Unknown seed(s): {unknown}; valid seeds: {exp16_2.FORMAL_SEEDS}"
        )
    if not seeds:
        raise argparse.ArgumentTypeError("At least one seed is required")
    return seeds


def _default_config(
    results_dir: Path | None,
    core_results_dir: Path | None,
) -> exp16_2.Config:
    repo_root = exp16_2.find_repo_root()
    return exp16_2.Config(
        repo_root=repo_root,
        results_dir=(
            results_dir.resolve()
            if results_dir is not None
            else exp16_2.default_results_dir(repo_root)
        ),
        core_results_dir=(
            core_results_dir.resolve()
            if core_results_dir is not None
            else (repo_root / exp16.CORE_RESULTS_REL).resolve()
        ),
    )


def _representative_test_sample(arrays: dict[str, np.ndarray]) -> int:
    lengths = arrays["test_lengths"].astype(np.float64)
    median_length = float(np.median(lengths))
    return int(np.argmin(np.abs(lengths - median_length)))


def _resolve_sample_index(
    arrays: dict[str, np.ndarray],
    sample_index: int | None,
    sample_id: str | None,
) -> int:
    n = len(arrays["test_y"])
    if sample_index is not None and sample_id is not None:
        raise ValueError("Use only one of --sample-index and --sample-id")

    if sample_id is not None:
        ids = arrays["test_ids"].astype(str)
        matches = np.flatnonzero(ids == str(sample_id))
        if len(matches) != 1:
            raise ValueError(
                f"--sample-id {sample_id!r} matched {len(matches)} test samples"
            )
        return int(matches[0])

    if sample_index is not None:
        if sample_index < 0 or sample_index >= n:
            raise IndexError(f"--sample-index must be in [0, {n - 1}]")
        return int(sample_index)

    return _representative_test_sample(arrays)


def _sample_metadata(
    arrays: dict[str, np.ndarray],
    labels: Iterable[str],
    sample_index: int,
) -> dict[str, object]:
    label_names = tuple(labels)
    label_index = int(arrays["test_y"][sample_index])
    return {
        "split": "test",
        "sample_index": int(sample_index),
        "sample_id": str(arrays["test_ids"][sample_index]),
        "user": str(arrays["test_users"][sample_index]),
        "label_index": label_index,
        "label": str(label_names[label_index]),
        "valid_length": int(arrays["test_lengths"][sample_index]),
    }


def _load_spikes(trace_path: Path, sample_index: int) -> tuple[np.ndarray, np.ndarray]:
    if not trace_path.exists():
        raise FileNotFoundError(trace_path)
    with np.load(trace_path, allow_pickle=False) as traces:
        l1 = np.asarray(traces["test__L1__spike"][sample_index], dtype=np.uint8)
        l2 = np.asarray(traces["test__L2__spike"][sample_index], dtype=np.uint8)
    if l1.ndim != 2 or l2.ndim != 2:
        raise ValueError(f"Expected [time, neuron] rasters in {trace_path}")
    if l1.shape[0] != l2.shape[0]:
        raise ValueError(f"L1/L2 time dimensions differ in {trace_path}")
    return l1, l2


def _plot_two_layer_raster(
    l1: np.ndarray,
    l2: np.ndarray,
    valid_length: int,
    *,
    title: str,
    output_path: Path,
    dpi: int,
) -> None:
    full_steps = int(l1.shape[0])
    if valid_length < 0 or valid_length > full_steps:
        raise ValueError(
            f"valid_length={valid_length} is outside the full window [0, {full_steps}]"
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(
        2,
        1,
        figsize=(12, 7),
        sharex=True,
        constrained_layout=True,
    )

    for ax, spikes, layer_name in (
        (axes[0], l1, "L1"),
        (axes[1], l2, "L2"),
    ):
        ax.imshow(
            spikes.T,
            aspect="auto",
            interpolation="nearest",
            origin="lower",
            cmap="binary",
            vmin=0,
            vmax=1,
            extent=(-0.5, full_steps - 0.5, -0.5, spikes.shape[1] - 0.5),
        )
        boundary = valid_length - 0.5
        ax.axvline(boundary, linestyle="--", linewidth=1.0)
        if valid_length < full_steps:
            ax.axvspan(boundary, full_steps - 0.5, alpha=0.06)
        ax.set_xlim(-0.5, full_steps - 0.5)
        ax.set_ylim(-0.5, spikes.shape[1] - 0.5)
        ax.set_ylabel(f"{layer_name} neuron")
        ax.set_title(f"{layer_name} spikes")

    axes[1].set_xlabel("Timestep (full padded window)")
    fig.suptitle(title)
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)


def generate_rasters(
    config: exp16_2.Config,
    *,
    cases: tuple[str, ...],
    seeds: tuple[int, ...],
    sample_index: int | None,
    sample_id: str | None,
    output_dir: Path | None,
    dpi: int,
) -> dict[str, object]:
    protocol, _, arrays = exp16._load_core(config)
    chosen_index = _resolve_sample_index(arrays, sample_index, sample_id)
    sample = _sample_metadata(arrays, protocol.labels, chosen_index)
    raster_dir = (
        output_dir.resolve()
        if output_dir is not None
        else config.results_dir / "rasters"
    )
    raster_dir.mkdir(parents=True, exist_ok=True)

    outputs: list[dict[str, object]] = []
    expected_full_steps: int | None = None

    for seed in seeds:
        for case in cases:
            run_key = f"{case}__seed{seed}"
            trace_path = config.results_dir / "runs" / run_key / "traces.npz"
            l1, l2 = _load_spikes(trace_path, chosen_index)

            full_steps = int(l1.shape[0])
            if expected_full_steps is None:
                expected_full_steps = full_steps
            elif full_steps != expected_full_steps:
                raise ValueError(
                    f"Full window changed across runs: {full_steps} vs "
                    f"{expected_full_steps} in {run_key}"
                )

            output_path = raster_dir / f"{run_key}__L1_L2.png"
            title = (
                f"Exp16.2 {run_key} | test index {chosen_index} | "
                f"id={sample['sample_id']} | user={sample['user']} | "
                f"label={sample['label']} | valid={sample['valid_length']} / "
                f"full={full_steps}"
            )
            _plot_two_layer_raster(
                l1,
                l2,
                int(sample["valid_length"]),
                title=title,
                output_path=output_path,
                dpi=dpi,
            )
            outputs.append(
                {
                    "case": case,
                    "seed": seed,
                    "run_key": run_key,
                    "trace_path": str(trace_path.relative_to(config.repo_root)),
                    "figure_path": str(output_path.relative_to(config.repo_root)),
                    "full_steps": full_steps,
                }
            )

    manifest = {
        "experiment_id": exp16_2.EXPERIMENT_ID,
        "protocol_version": exp16_2.PROTOCOL_VERSION,
        "plot": "two-row L1/L2 spike raster",
        "x_axis_policy": "full padded window; valid-length boundary shown as dashed line",
        "sample": sample,
        "cases": list(cases),
        "seeds": list(seeds),
        "full_steps": expected_full_steps,
        "outputs": outputs,
    }
    manifest_path = raster_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results",
        type=Path,
        help="Exp16.2 result root; defaults to the standard artifact directory",
    )
    parser.add_argument(
        "--core-results",
        type=Path,
        help="CoreBenchmark result root; defaults to core_benchmark_v1/results/main",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Output directory; defaults to <results>/rasters",
    )
    parser.add_argument(
        "--cases",
        type=_parse_cases,
        default=DEFAULT_CASES,
        help="Comma-separated cases; default: all formal Exp16.2 cases",
    )
    parser.add_argument(
        "--seeds",
        type=_parse_seeds,
        default=tuple(exp16_2.FORMAL_SEEDS),
        help="Comma-separated formal seeds; default: 11,23,37",
    )
    parser.add_argument(
        "--sample-index",
        type=int,
        help="Test-set row index. Default: valid length closest to test median",
    )
    parser.add_argument(
        "--sample-id",
        type=str,
        help="Test sample ID; mutually exclusive with --sample-index",
    )
    parser.add_argument("--dpi", type=int, default=180)
    args = parser.parse_args()

    config = _default_config(args.results, args.core_results)
    manifest = generate_rasters(
        config,
        cases=args.cases,
        seeds=args.seeds,
        sample_index=args.sample_index,
        sample_id=args.sample_id,
        output_dir=args.output_dir,
        dpi=args.dpi,
    )
    sample = manifest["sample"]
    output_root = args.output_dir.resolve() if args.output_dir else config.results_dir / "rasters"
    print(f"Generated {len(manifest['outputs'])} rasters in {output_root.resolve()}")
    print(
        f"Sample: test index={sample['sample_index']}, id={sample['sample_id']}, "
        f"user={sample['user']}, label={sample['label']}, "
        f"valid={sample['valid_length']}, full={manifest['full_steps']}"
    )


if __name__ == "__main__":
    main()
