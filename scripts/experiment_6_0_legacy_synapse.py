from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path

import torch

from scripts import experiment_6_0_multiscale_phase_evidence as base


SYNAPSE_MODE = "legacy_unnormalized"
LEGACY_SUBDIR = "legacy_unnormalized_synapse"
SYNAPTIC_UPDATE = "I_t = alpha*I_{t-1} + W*x_t"
_BASE_ARCHITECTURE_MANIFEST = base.architecture_manifest


class LegacySingleHiddenMultiTauSNN(base.SingleHiddenMultiTauSNN):
    """Exp6.0 control using the legacy unnormalized synaptic-current update."""

    def forward_trajectory(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        if x.ndim != 3:
            raise ValueError(f"Expected [B,T,C] input, got {tuple(x.shape)}")
        batch, n_steps, channels = x.shape
        if channels != base.EXPECTED_CHANNELS:
            raise ValueError(
                f"Expected {base.EXPECTED_CHANNELS} channels, got {channels}"
            )

        syn = torch.zeros(
            batch, base.HIDDEN_WIDTH, device=x.device, dtype=x.dtype
        )
        hidden_mem = torch.zeros_like(syn)
        output_mem = torch.zeros(
            batch, self.n_classes, device=x.device, dtype=x.dtype
        )

        hidden_spikes: list[torch.Tensor] = []
        output_spikes: list[torch.Tensor] = []
        pre_output_evidence: list[torch.Tensor] = []

        alpha = self.syn_alpha.to(device=x.device, dtype=x.dtype)
        for step in range(n_steps):
            projected = self.input_hidden(x[:, step])
            syn = alpha * syn + projected
            hidden_spike, hidden_mem, _ = self.hidden_lif(syn, hidden_mem)
            evidence = self.output_linear(hidden_spike)
            output_spike, output_mem, _ = self.output_lif(evidence, output_mem)
            hidden_spikes.append(hidden_spike)
            pre_output_evidence.append(evidence)
            output_spikes.append(output_spike)

        return {
            "hidden_spikes": torch.stack(hidden_spikes, dim=1),
            "pre_output_evidence": torch.stack(pre_output_evidence, dim=1),
            "output_spikes": torch.stack(output_spikes, dim=1),
        }


def legacy_results_dir(repo_root: Path) -> Path:
    return base.results_dir(repo_root) / LEGACY_SUBDIR


def _legacy_model(
    spec: base.RunSpec,
    data: base.exp3.Data,
) -> LegacySingleHiddenMultiTauSNN:
    base._validate_spec(spec)
    return LegacySingleHiddenMultiTauSNN(
        n_classes=len(data.labels),
        fs=data.fs,
        mem_shift=spec.mem_shift,
    )


def legacy_architecture_manifest(
    spec: base.RunSpec,
    data: base.exp3.Data,
) -> dict[str, object]:
    manifest = _BASE_ARCHITECTURE_MANIFEST(spec, data)
    hidden = manifest["hidden"]
    if not isinstance(hidden, dict):
        raise TypeError("Exp6.0 hidden architecture manifest must be a dict")
    hidden["synaptic_update"] = SYNAPTIC_UPDATE
    hidden["synapse_mode"] = SYNAPSE_MODE
    return manifest


def _install_legacy_overrides() -> None:
    # Exp6.0's trainer/evaluator resolves these module globals at call time.
    # Each Slurm task is a separate process, so this override is process-local.
    base._new_model = _legacy_model  # type: ignore[assignment]
    base.architecture_manifest = legacy_architecture_manifest  # type: ignore[assignment]


def run_one(
    spec: base.RunSpec,
    data: base.exp3.Data,
    config: base.Config,
    force: bool,
) -> dict[str, object]:
    _install_legacy_overrides()
    payload = base.run_one(spec, data, config, force)
    payload["synapse_mode"] = SYNAPSE_MODE
    provenance = payload.get("provenance")
    if isinstance(provenance, dict):
        provenance["synaptic_update"] = SYNAPTIC_UPDATE
        provenance["paired_control"] = (
            "all Exp6.0 settings and random streams match the normalized condition; "
            "only the hidden synaptic input coefficient differs"
        )
    base._save_json(base.evaluation_path(config.results_dir, spec), payload)
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Experiment 6.0 legacy unnormalized-synapse control"
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=base.EPOCHS)
    parser.add_argument("--batch-size", type=int, default=base.BATCH_SIZE)
    parser.add_argument("--force", action="store_true")
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run-one")
    run.add_argument("--array-task-id", type=int, required=True)
    sub.add_parser("list-runs")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    repo_root = base.find_repo_root()
    config = base.Config(
        repo_root=repo_root,
        results_dir=legacy_results_dir(repo_root),
        device=args.device,
        epochs=args.epochs,
        batch_size=args.batch_size,
        threads=args.threads,
    )

    specs = base.run_specs()
    if args.command == "list-runs":
        for index, spec in enumerate(specs):
            print(index, SYNAPSE_MODE, spec.key)
        return
    if args.command == "run-one":
        task_id = int(args.array_task_id)
        if task_id < 0 or task_id >= len(specs):
            raise IndexError(
                f"array-task-id {task_id} outside [0, {len(specs) - 1}]"
            )
        spec = specs[task_id]
        data = base.prepare_data(repo_root)
        payload = run_one(spec, data, config, args.force)
        print(
            json.dumps(
                {
                    "synapse_mode": SYNAPSE_MODE,
                    "synaptic_update": SYNAPTIC_UPDATE,
                    "spec": asdict(spec),
                    "best_epoch": payload["best_epoch"],
                    "val_ba": payload["metrics"]["val"]["balanced_accuracy"],
                    "test_ba": payload["metrics"]["test"]["balanced_accuracy"],
                },
                indent=2,
            )
        )
        return
    raise RuntimeError(f"Unhandled command: {args.command}")


if __name__ == "__main__":
    main()
