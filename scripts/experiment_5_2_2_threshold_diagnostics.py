from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from scripts import experiment_3_0_1_single_tau_objectives as base
from scripts import experiment_5_2_2_frozen_local_multitau_syn as exp522


DIAGNOSTIC_VERSION = "threshold_operating_point_v1"
THRESHOLD_METRICS = (
    "firing_rate_hz",
    "spike_probability",
    "mean_abs_input_current",
    "mean_abs_pre_reset_membrane",
    "pre_reset_above_threshold_probability",
)


def diagnostic_path(root: Path, spec: exp522.RunSpec) -> Path:
    return root / "threshold_diagnostics" / f"{spec.key}.json"


def pre_reset_membrane(trajectory: dict[str, torch.Tensor]) -> torch.Tensor:
    """Recover exact L3 U^- from the subtractive-reset MacroMultiSpikeLIF state.

    MacroMultiSpikeLIF applies:
        U^- = beta * U_(t-1) + I_t
        S_t = threshold(U^-)
        U_t = U^- - S_t * theta

    Exp5.2.2 uses hidden cap=1 and theta=0.5, so U^- is exactly recoverable
    from the stored post-reset membrane and emitted spike trajectory.
    """
    return trajectory["hidden_membranes"] + trajectory["hidden_spikes"] * float(
        exp522.THRESHOLD
    )


def _group_slices(profile: str) -> dict[str, slice]:
    groups: dict[str, slice] = {"all": slice(0, exp522.TEMPORAL_WIDTH)}
    groups.update(
        {str(shift): group for shift, group in exp522.profile_group_slices(profile).items()}
    )
    return groups


def _empty_accumulators(profile: str) -> dict[str, dict[str, object]]:
    return {
        shift: {
            "slice": group,
            "entries": 0.0,
            "spikes": 0.0,
            "input_abs": 0.0,
            "pre_reset_abs": 0.0,
            "pre_reset_above": 0.0,
        }
        for shift, group in _group_slices(profile).items()
    }


def _update_accumulators(
    accumulators: dict[str, dict[str, object]],
    trajectory: dict[str, torch.Tensor],
    lengths: torch.Tensor,
) -> None:
    spikes = trajectory["hidden_spikes"]
    input_current = trajectory["hidden_synaptic"]
    u_minus = pre_reset_membrane(trajectory)
    valid = base.mask(lengths, spikes.shape[1]).unsqueeze(-1)

    for accumulator in accumulators.values():
        group = accumulator["slice"]
        if not isinstance(group, slice):
            raise TypeError("Threshold diagnostic group slice is invalid")
        group_spikes = spikes[:, :, group]
        group_input = input_current[:, :, group]
        group_u_minus = u_minus[:, :, group]
        group_valid = valid.expand_as(group_spikes)
        group_valid_float = group_valid.to(group_spikes.dtype)
        entries = float(group_valid_float.sum().item())

        accumulator["entries"] = float(accumulator["entries"]) + entries
        accumulator["spikes"] = float(accumulator["spikes"]) + float(
            ((group_spikes > 0) & group_valid).sum().item()
        )
        accumulator["input_abs"] = float(accumulator["input_abs"]) + float(
            (group_input.abs() * group_valid_float).sum().item()
        )
        accumulator["pre_reset_abs"] = float(accumulator["pre_reset_abs"]) + float(
            (group_u_minus.abs() * group_valid_float).sum().item()
        )
        accumulator["pre_reset_above"] = float(
            accumulator["pre_reset_above"]
        ) + float(
            ((group_u_minus > float(exp522.THRESHOLD)) & group_valid).sum().item()
        )


def _finalize_accumulators(
    accumulators: dict[str, dict[str, object]],
    fs: float,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for shift, accumulator in accumulators.items():
        group = accumulator["slice"]
        if not isinstance(group, slice):
            raise TypeError("Threshold diagnostic group slice is invalid")
        entries = float(accumulator["entries"])
        spike_probability = float(accumulator["spikes"]) / max(entries, exp522.EPS)
        rows.append(
            {
                "shift": shift,
                "n_neurons": group.stop - group.start,
                "threshold": float(exp522.THRESHOLD),
                "firing_rate_hz": spike_probability * float(fs),
                "spike_probability": spike_probability,
                "mean_abs_input_current": float(accumulator["input_abs"])
                / max(entries, exp522.EPS),
                "mean_abs_pre_reset_membrane": float(accumulator["pre_reset_abs"])
                / max(entries, exp522.EPS),
                "pre_reset_above_threshold_probability": float(
                    accumulator["pre_reset_above"]
                )
                / max(entries, exp522.EPS),
            }
        )
    return rows


def evaluate_split(
    model: exp522.SynapticPhaseDecoder,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    fs: float,
    reset_shifts: tuple[int, ...],
) -> list[dict[str, object]]:
    accumulators = _empty_accumulators(model.profile)
    model.eval()
    with torch.no_grad():
        for l2, _, lengths in loader:
            l2 = l2.to(device)
            lengths = lengths.to(device)
            trajectory = model.forward_trajectory(l2, reset_shifts=reset_shifts)
            _update_accumulators(accumulators, trajectory, lengths)
    return _finalize_accumulators(accumulators, fs)


def evaluate_one(
    spec: exp522.RunSpec,
    data: base.Data,
    config: exp522.Config,
    force: bool = False,
) -> dict[str, object]:
    destination = diagnostic_path(config.results_dir, spec)
    if destination.exists() and not force:
        return json.loads(destination.read_text(encoding="utf-8"))

    main_evaluation = exp522.evaluation_path(config.results_dir, spec)
    if not main_evaluation.exists():
        raise FileNotFoundError(
            f"Missing main Exp5.2.2 evaluation for threshold diagnostics: {main_evaluation}"
        )
    cache = exp522.load_source_cache(spec.seed, data, config)
    model, _ = exp522.load_model(spec, data, config)
    loaders = exp522._make_loaders(data, cache, spec, config, train_shuffle=False)
    device = torch.device(config.device)

    unique_rows: dict[tuple[int, ...], list[dict[str, object]]] = {}
    logical_rows: list[dict[str, object]] = []
    for intervention in exp522.INTERVENTIONS:
        effective = exp522.effective_reset_shifts(spec.profile, intervention)
        if effective not in unique_rows:
            rows: list[dict[str, object]] = []
            for split, loader in loaders.items():
                for row in evaluate_split(model, loader, device, data.fs, effective):
                    rows.append({"split": split, **row})
            unique_rows[effective] = rows
        for row in unique_rows[effective]:
            logical_rows.append(
                {
                    "profile": spec.profile,
                    "readout": spec.readout,
                    "seed": spec.seed,
                    "intervention": intervention,
                    "effective_mask_id": exp522.effective_mask_id(
                        spec.profile, intervention
                    ),
                    "effective_reset_shifts": effective,
                    **row,
                }
            )

    payload = {
        "experiment_id": exp522.EXPERIMENT_ID,
        "protocol_version": exp522.PROTOCOL_VERSION,
        "diagnostic_version": DIAGNOSTIC_VERSION,
        "spec": spec.__dict__,
        "threshold_fixed_not_swept": float(exp522.THRESHOLD),
        "u_minus_definition": "exact MacroMultiSpikeLIF pre-reset membrane: U^- = beta*U_(t-1) + I_t",
        "strict_threshold_probability": "P(U^- > theta)",
        "metrics": THRESHOLD_METRICS,
        "rows": logical_rows,
    }
    exp522._save_json(destination, payload)
    return payload


def finalize_threshold_diagnostics(repo_root: Path) -> Path:
    root = exp522.results_dir(repo_root)
    main_activity_path = root / "activity.csv"
    if not main_activity_path.exists():
        raise FileNotFoundError(
            f"Main Exp5.2.2 finalizer must run first: missing {main_activity_path}"
        )
    main_activity = pd.read_csv(main_activity_path)

    diagnostic_rows: list[dict[str, object]] = []
    for spec in exp522.run_specs():
        path = diagnostic_path(root, spec)
        if not path.exists():
            raise FileNotFoundError(f"Missing Exp5.2.2 threshold diagnostic: {path}")
        payload = json.loads(path.read_text(encoding="utf-8"))
        if (
            payload.get("experiment_id") != exp522.EXPERIMENT_ID
            or payload.get("protocol_version") != exp522.PROTOCOL_VERSION
            or payload.get("diagnostic_version") != DIAGNOSTIC_VERSION
            or payload.get("spec") != spec.__dict__
        ):
            raise ValueError(f"Threshold diagnostic identity mismatch: {path}")
        diagnostic_rows.extend(payload["rows"])

    diagnostics = pd.DataFrame(diagnostic_rows)
    keys = [
        "profile",
        "readout",
        "seed",
        "intervention",
        "effective_mask_id",
        "split",
        "shift",
    ]
    merged = main_activity.merge(
        diagnostics[
            keys
            + [
                "threshold",
                "firing_rate_hz",
                "spike_probability",
                "mean_abs_input_current",
                "mean_abs_pre_reset_membrane",
                "pre_reset_above_threshold_probability",
            ]
        ],
        on=keys,
        how="left",
        validate="one_to_one",
    )
    if merged["spike_probability"].isna().any():
        raise ValueError("Threshold diagnostics failed to cover all activity rows")
    if not np.allclose(
        merged["spike_probability"].to_numpy(),
        merged["nonzero_fraction"].to_numpy(),
        atol=1e-7,
        rtol=1e-6,
    ):
        raise ValueError("P(S_t=1) disagrees with binary-spike nonzero_fraction")
    if not np.allclose(
        merged["mean_abs_input_current"].to_numpy(),
        merged["mean_abs_syn"].to_numpy(),
        atol=1e-6,
        rtol=1e-5,
    ):
        raise ValueError("E|I_t| disagrees with existing mean_abs_syn diagnostic")

    destination = root / "threshold_activity.csv"
    merged.to_csv(destination, index=False)

    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["threshold_operating_point_diagnostics"] = {
        "diagnostic_version": DIAGNOSTIC_VERSION,
        "threshold_fixed_not_swept": float(exp522.THRESHOLD),
        "metrics": list(THRESHOLD_METRICS),
        "u_minus_definition": "MacroMultiSpikeLIF pre-reset membrane before threshold/subtractive reset",
        "strict_threshold_probability": "P(U^- > theta)",
        "grouping": "overall and each tau_syn/shift group; emphasize shifts 6 and 7",
        "file": destination.name,
    }
    files = manifest.setdefault("files", {})
    if not isinstance(files, dict):
        raise TypeError("Exp5.2.2 manifest files field is invalid")
    files["threshold_activity"] = destination.name
    exp522._save_json(manifest_path, manifest)
    return destination


def _config_from_args(args: argparse.Namespace) -> exp522.Config:
    repo_root = (
        Path(args.repo_root).resolve() if args.repo_root else exp522.find_repo_root()
    )
    return exp522.Config(
        repo_root=repo_root,
        results_dir=exp522.results_dir(repo_root),
        device=args.device,
        epochs=exp522.EPOCHS,
        batch_size=args.batch_size,
        threads=args.threads,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Exp5.2.2 fixed-threshold per-timescale operating-point diagnostics"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    evaluate = subparsers.add_parser("evaluate-one")
    evaluate.add_argument("--repo-root", type=str, default=None)
    evaluate.add_argument("--device", type=str, default="cpu")
    evaluate.add_argument("--threads", type=int, default=1)
    evaluate.add_argument("--batch-size", type=int, default=exp522.BATCH_SIZE)
    evaluate.add_argument("--force", action="store_true")
    evaluate.add_argument("--array-task-id", type=int, required=True)

    finalize = subparsers.add_parser("finalize")
    finalize.add_argument("--repo-root", type=str, default=None)
    return parser


def main() -> None:
    args = _parser().parse_args()
    if args.command == "finalize":
        repo_root = (
            Path(args.repo_root).resolve()
            if args.repo_root
            else exp522.find_repo_root()
        )
        print(finalize_threshold_diagnostics(repo_root))
        return

    config = _config_from_args(args)
    data = base.prepare_data(config.repo_root)
    specs = exp522.run_specs()
    task_id = int(args.array_task_id)
    if not 0 <= task_id < len(specs):
        raise IndexError(f"evaluate-one task {task_id} outside 0..{len(specs)-1}")
    payload = evaluate_one(specs[task_id], data, config, force=args.force)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
