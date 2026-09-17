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

from scripts import experiment_3_0_1_single_tau_objectives as exp3
from scripts import experiment_4_0_1_multispike_macro_lif as exp401
from scripts import experiment_5_0_local_evidence_objectives as exp50
from scripts import experiment_7_2_two_layer_tau_training as exp72
from scripts import experiment_7_3_training_strategy_decomposition as exp73


EXPERIMENT_ID = "experiment_7_4_latent_softmax_evidence"
PROTOCOL_VERSION = "latent_softmax_evidence_v1"
ARCHITECTURE = exp73.ARCHITECTURE
SHIFTS = exp73.SHIFTS
REGULARIZATION = exp73.REGULARIZATION
SEEDS = exp73.SEEDS
HIDDEN_WIDTH = exp73.HIDDEN_WIDTH
CE_GAIN = exp73.CE_GAIN
MAX_EPOCHS = exp73.MAX_EPOCHS
MIN_EPOCHS = exp73.MIN_EPOCHS
PATIENCE = exp73.PATIENCE
TEMPERATURE = 1.0
EPS = 1e-12

HEAD_CASES = (
    ("A_a2_direct_linear", "direct"),
    ("B_two_linear", "two_linear"),
    ("C_latent_softmax", "softmax"),
    ("D_latent_softmax_rms", "softmax_rms"),
)


@dataclass(frozen=True)
class RunSpec:
    seed: int
    head: str

    @property
    def method(self) -> str:
        for name, head in HEAD_CASES:
            if self.head == head:
                return name
        raise ValueError(self.head)

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
    return exp73.find_repo_root(start)


def results_dir(repo_root: Path) -> Path:
    return repo_root / "notebooks" / "artifacts" / EXPERIMENT_ID / PROTOCOL_VERSION


def run_specs() -> list[RunSpec]:
    return [RunSpec(seed, head) for seed in SEEDS for _, head in HEAD_CASES]


def validate_spec(spec: RunSpec) -> None:
    if spec.seed not in SEEDS:
        raise ValueError(spec.seed)
    if spec.head not in {head for _, head in HEAD_CASES}:
        raise ValueError(spec.head)
    _ = spec.method


def _save_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _path(root: Path, kind: str, key: str, suffix: str) -> Path:
    return root / kind / f"{key}{suffix}"


class LatentEvidenceHead(nn.Module):
    """Four matched heads for the Exp7.4 mechanism ablation."""

    def __init__(
        self,
        head: str,
        n_classes: int,
        temperature: float = TEMPERATURE,
    ) -> None:
        super().__init__()
        if head not in {value for _, value in HEAD_CASES}:
            raise ValueError(head)
        if temperature <= 0:
            raise ValueError(temperature)
        self.head = head
        self.n_classes = int(n_classes)
        self.temperature = float(temperature)

        if head == "direct":
            self.direct = nn.Linear(HIDDEN_WIDTH, n_classes, bias=False)
            self.to_latent = None
            self.to_class = None
        else:
            self.direct = None
            self.to_latent = nn.Linear(HIDDEN_WIDTH, HIDDEN_WIDTH, bias=False)
            self.to_class = nn.Linear(HIDDEN_WIDTH, n_classes, bias=False)

    def forward(
        self, z: torch.Tensor, collect_diagnostics: bool = False
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if self.head == "direct":
            assert self.direct is not None
            evidence = self.direct(z)
            aux: dict[str, torch.Tensor] = {}
        else:
            assert self.to_latent is not None
            assert self.to_class is not None
            h = self.to_latent(z)
            latent_rms = torch.sqrt(torch.mean(h.square(), dim=-1) + EPS)

            if self.head == "two_linear":
                routed = h
                aux = {"latent_rms": latent_rms}
            else:
                q = F.softmax(h / self.temperature, dim=-1)
                entropy = -(q * torch.log(q.clamp_min(EPS))).sum(dim=-1)
                aux = {
                    "latent_rms": latent_rms,
                    "softmax_entropy": entropy,
                    "softmax_entropy_normalized": entropy / math.log(HIDDEN_WIDTH),
                    "softmax_max_prob": q.max(dim=-1).values,
                    "softmax_effective_states": torch.exp(entropy),
                }
                if self.head == "softmax":
                    routed = q
                elif self.head == "softmax_rms":
                    magnitude = latent_rms
                    routed = q * magnitude.unsqueeze(-1)
                    aux["magnitude_rms"] = magnitude
                else:
                    raise ValueError(self.head)

            evidence = self.to_class(routed)

        if collect_diagnostics:
            aux = dict(aux)
            aux["evidence_rms"] = torch.sqrt(
                torch.mean(evidence.square(), dim=-1) + EPS
            )
            return evidence, aux
        return evidence, {}


class Exp74Net(nn.Module):
    """Exp7.3 A2 hidden backbone with a matched Exp7.4 classification head."""

    def __init__(self, head: str, n_classes: int, fs: float) -> None:
        super().__init__()
        self.head_name = head
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
                    threshold=exp73.THRESHOLD,
                    max_spikes_per_dt=1,
                    surrogate_slope=exp72.SURROGATE_SLOPE,
                )
                for _ in range(2)
            ]
        )
        self.register_buffer("alpha_0", exp50.alpha_vector(HIDDEN_WIDTH, SHIFTS[0]))
        self.register_buffer("alpha_1", exp50.alpha_vector(HIDDEN_WIDTH, SHIFTS[1]))
        self.head = LatentEvidenceHead(head, n_classes, temperature=TEMPERATURE)

    def forward_trajectory(
        self, x: torch.Tensor, collect_diagnostics: bool = False
    ) -> dict[str, Any]:
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
        diagnostic_steps: dict[str, list[torch.Tensor]] = {}

        for t in range(steps):
            cur = x[:, t]
            for li in range(2):
                alpha = getattr(self, f"alpha_{li}")
                syn[li] = alpha * syn[li] + self.hidden_linears[li](cur)
                spk, mem[li], _ = self.hidden_lifs[li](syn[li], mem[li])
                hidden[li].append(spk)
                cur = spk

            evidence, aux = self.head(cur, collect_diagnostics=collect_diagnostics)
            evidence_steps.append(evidence)
            for key, value in aux.items():
                diagnostic_steps.setdefault(key, []).append(value)

        payload: dict[str, Any] = {
            "hidden_spikes": tuple(torch.stack(v, dim=1) for v in hidden),
            "evidence": torch.stack(evidence_steps, dim=1),
        }
        if collect_diagnostics:
            payload["diagnostics"] = {
                key: torch.stack(values, dim=1)
                for key, values in diagnostic_steps.items()
            }
        return payload


def _evaluate(
    model: Exp74Net,
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
            trajectory = model.forward_trajectory(X)
            loss, scores = exp73._objective_loss_scores(
                trajectory["evidence"], lengths, y, "wcce"
            )
            ys.append(y.cpu().numpy())
            preds.append(scores.argmax(dim=1).cpu().numpy())
            loss_sum += float(loss) * len(y)
            n_total += len(y)

    out = exp72._metrics(np.concatenate(ys), np.concatenate(preds))
    out["objective_loss"] = loss_sum / max(n_total, 1)
    return out


def _diagnostic_summary(
    model: Exp74Net,
    loader: Iterable,
    device: torch.device,
) -> dict[str, dict[str, float]]:
    values: dict[str, list[torch.Tensor]] = {}
    model.eval()
    with torch.no_grad():
        for X, _, lengths in loader:
            X = X.to(device)
            lengths = lengths.to(device)
            trajectory = model.forward_trajectory(X, collect_diagnostics=True)
            valid = exp73._valid_mask(lengths, X.shape[1])
            for key, tensor in trajectory["diagnostics"].items():
                values.setdefault(key, []).append(tensor[valid].detach().cpu())

    summary: dict[str, dict[str, float]] = {}
    for key, chunks in values.items():
        merged = torch.cat(chunks) if chunks else torch.empty(0)
        if merged.numel() == 0:
            continue
        summary[key] = {
            "mean": float(merged.mean()),
            "std": float(merged.std(unbiased=False)),
            "min": float(merged.min()),
            "max": float(merged.max()),
        }
    return summary


def _head_parameter_count(model: Exp74Net) -> int:
    return sum(parameter.numel() for parameter in model.head.parameters())


def _probe_test_ba(probes: dict[str, Any], source: str) -> float:
    return float(probes[source]["metrics"]["test"]["balanced_accuracy"])


def run_one(spec: RunSpec, config: Config, force: bool = False) -> dict[str, Any]:
    validate_spec(spec)
    eval_path = _path(config.results_dir, "evaluations", spec.key, ".json")
    checkpoint_path = _path(config.results_dir, "checkpoints", spec.key, ".pt")
    if eval_path.exists() and checkpoint_path.exists() and not force:
        return json.loads(eval_path.read_text(encoding="utf-8"))

    data = exp3.prepare_data(config.repo_root)
    device = torch.device(config.device)
    torch.set_num_threads(config.threads)

    # Reuse the Exp7.3 E2E pairing seeds so Case A is a strict A2 replication
    # and all four Exp7.4 heads see matched hidden initialization and data order.
    exp3.seed_all(exp73._e2e_pair_seed(spec.seed, "model_init"))
    model = Exp74Net(spec.head, len(data.labels), data.fs).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=exp72.LR, weight_decay=exp72.WEIGHT_DECAY
    )

    train_loader = exp73._raw_loaders(
        data, spec.seed, config.batch_size, shuffle_train=True
    )["train"]
    eval_loaders = exp73._raw_loaders(
        data, spec.seed, config.batch_size, shuffle_train=False
    )

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
            trajectory = model.forward_trajectory(X)
            loss, _ = exp73._objective_loss_scores(
                trajectory["evidence"], lengths, y, "wcce"
            )
            loss.backward()
            optimizer.step()

            train_loss_sum += float(loss.detach()) * len(y)
            n_total += len(y)

        train_metrics = _evaluate(model, eval_loaders["train"], device)
        val_metrics = _evaluate(model, eval_loaders["val"], device)
        history.append(
            {
                "epoch": float(epoch),
                "train_ba": float(train_metrics["balanced_accuracy"]),
                "val_ba": float(val_metrics["balanced_accuracy"]),
                "train_loss": train_loss_sum / max(n_total, 1),
                "val_loss": float(val_metrics["objective_loss"]),
            }
        )

        if exp73._checkpoint_improved(val_metrics, best_ba, best_loss):
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
        raise RuntimeError(f"No checkpoint selected for {spec.key}")

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
            "temperature": TEMPERATURE,
            "pairing": (
                "Exp7.3 A2 model-init and loader seeds are reused across heads "
                "for a strict paired comparison"
            ),
        },
        checkpoint_path,
    )

    model.load_state_dict(best_state, strict=True)
    native_metrics = {
        split: _evaluate(model, loader, device)
        for split, loader in eval_loaders.items()
    }
    diagnostics = {
        split: _diagnostic_summary(model, loader, device)
        for split, loader in eval_loaders.items()
    }

    l2_splits = {
        split: exp73._extract_l2(model, loader, device)
        for split, loader in eval_loaders.items()
    }
    probes = exp73._fit_probes(l2_splits, spec.seed, data.bin_steps)

    payload = {
        "experiment_id": EXPERIMENT_ID,
        "protocol_version": PROTOCOL_VERSION,
        "method": spec.method,
        "spec": asdict(spec),
        "contract": {
            "architecture": ARCHITECTURE,
            "shifts": [list(value) for value in SHIFTS],
            "regularization": REGULARIZATION,
            "objective": "wcce",
            "aggregation": "valid_timestep_mean",
            "bias": False,
            "temperature": TEMPERATURE,
            "all_l1_l2_head_trainable": True,
            "checkpoint_metric": "native_validation_balanced_accuracy",
            "checkpoint_tiebreak": "native_validation_objective_loss",
            "head_trainable_parameters": _head_parameter_count(model),
            "case_a_reuses_exp73_a2_pairing": True,
        },
        "best_epoch": best_epoch,
        "stopped_epoch": stopped_epoch,
        "native_metrics": native_metrics,
        "diagnostics": diagnostics,
        "representation_probes": probes,
    }
    _save_json(eval_path, payload)

    history_path = _path(config.results_dir, "histories", spec.key, ".csv")
    history_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(history).to_csv(history_path, index=False)
    return payload


def _load_required_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def _diagnostic_mean(payload: dict[str, Any], key: str) -> float:
    node = payload.get("diagnostics", {}).get("test", {}).get(key)
    return float(node["mean"]) if node is not None else float("nan")


def _row(payload: dict[str, Any]) -> dict[str, Any]:
    spec = payload["spec"]
    native = payload["native_metrics"]
    probes = payload["representation_probes"]
    return {
        "method": payload["method"],
        "head": spec["head"],
        "seed": int(spec["seed"]),
        "architecture": ARCHITECTURE,
        "best_epoch": int(payload["best_epoch"]),
        "stopped_epoch": int(payload["stopped_epoch"]),
        "head_trainable_parameters": int(
            payload["contract"]["head_trainable_parameters"]
        ),
        "train_ba": float(native["train"]["balanced_accuracy"]),
        "val_ba": float(native["val"]["balanced_accuracy"]),
        "test_ba": float(native["test"]["balanced_accuracy"]),
        "test_objective_loss": float(native["test"]["objective_loss"]),
        "l2_wholecount_probe_test_ba": _probe_test_ba(
            probes, "l2_wholecount_linear"
        ),
        "l2_fixed250_probe_test_ba": _probe_test_ba(
            probes, "l2_fixed250_linear"
        ),
        "test_latent_rms_mean": _diagnostic_mean(payload, "latent_rms"),
        "test_softmax_entropy_mean": _diagnostic_mean(
            payload, "softmax_entropy"
        ),
        "test_softmax_entropy_normalized_mean": _diagnostic_mean(
            payload, "softmax_entropy_normalized"
        ),
        "test_softmax_max_prob_mean": _diagnostic_mean(
            payload, "softmax_max_prob"
        ),
        "test_softmax_effective_states_mean": _diagnostic_mean(
            payload, "softmax_effective_states"
        ),
        "test_magnitude_rms_mean": _diagnostic_mean(payload, "magnitude_rms"),
        "test_evidence_rms_mean": _diagnostic_mean(payload, "evidence_rms"),
    }


def _contrast_frames(runs: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    contrasts = (
        (
            "B_minus_A_factorization",
            "B_two_linear",
            "A_a2_direct_linear",
        ),
        (
            "C_minus_B_softmax_normalization",
            "C_latent_softmax",
            "B_two_linear",
        ),
        (
            "D_minus_C_restore_rms_magnitude",
            "D_latent_softmax_rms",
            "C_latent_softmax",
        ),
        (
            "D_minus_B_softmax_plus_magnitude_vs_linear",
            "D_latent_softmax_rms",
            "B_two_linear",
        ),
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
                    "test_ba_delta_pp": 100.0
                    * (
                        float(left_rows.loc[seed, "test_ba"])
                        - float(right_rows.loc[seed, "test_ba"])
                    ),
                    "l2_wholecount_probe_delta_pp": 100.0
                    * (
                        float(left_rows.loc[seed, "l2_wholecount_probe_test_ba"])
                        - float(right_rows.loc[seed, "l2_wholecount_probe_test_ba"])
                    ),
                }
            )

    contrast_runs = pd.DataFrame(rows)
    contrast_summary = (
        contrast_runs.groupby(
            ["contrast", "left_method", "right_method"], sort=False
        )[["test_ba_delta_pp", "l2_wholecount_probe_delta_pp"]]
        .agg(["mean", "std"])
        .reset_index()
    )
    contrast_summary.columns = [
        "_".join(str(value) for value in column if str(value))
        if isinstance(column, tuple)
        else str(column)
        for column in contrast_summary.columns
    ]
    return contrast_runs, contrast_summary


def finalize(config: Config) -> dict[str, Any]:
    rows = []
    for spec in run_specs():
        payload = _load_required_json(
            _path(config.results_dir, "evaluations", spec.key, ".json")
        )
        rows.append(_row(payload))

    runs = pd.DataFrame(rows)
    expected = len(HEAD_CASES) * len(SEEDS)
    if len(runs) != expected:
        raise RuntimeError(f"Expected {expected} method/seed rows, got {len(runs)}")

    config.results_dir.mkdir(parents=True, exist_ok=True)
    runs.to_csv(config.results_dir / "method_runs.csv", index=False)

    metric_cols = [
        "test_ba",
        "test_objective_loss",
        "l2_wholecount_probe_test_ba",
        "l2_fixed250_probe_test_ba",
        "best_epoch",
        "test_latent_rms_mean",
        "test_softmax_entropy_mean",
        "test_softmax_entropy_normalized_mean",
        "test_softmax_max_prob_mean",
        "test_softmax_effective_states_mean",
        "test_magnitude_rms_mean",
        "test_evidence_rms_mean",
    ]
    summary = (
        runs.groupby(
            ["method", "head", "architecture", "head_trainable_parameters"],
            sort=False,
        )[metric_cols]
        .agg(["mean", "std"])
        .reset_index()
    )
    summary.columns = [
        "_".join(str(value) for value in column if str(value))
        if isinstance(column, tuple)
        else str(column)
        for column in summary.columns
    ]
    summary.to_csv(config.results_dir / "method_summary.csv", index=False)

    contrast_runs, contrast_summary = _contrast_frames(runs)
    contrast_runs.to_csv(config.results_dir / "contrast_runs.csv", index=False)
    contrast_summary.to_csv(config.results_dir / "contrast_summary.csv", index=False)

    manifest = {
        "experiment_id": EXPERIMENT_ID,
        "protocol_version": PROTOCOL_VERSION,
        "architecture": ARCHITECTURE,
        "shifts": [list(value) for value in SHIFTS],
        "seeds": list(SEEDS),
        "regularization": REGULARIZATION,
        "objective": "wcce",
        "temperature": TEMPERATURE,
        "methods": [name for name, _ in HEAD_CASES],
        "counts": {
            "parallel_runs": len(run_specs()),
            "methods": len(HEAD_CASES),
            "seeds": len(SEEDS),
        },
        "primary_metric": "native_test_balanced_accuracy",
        "checkpoint_rule": (
            "native validation BA primary, native WCCE loss tiebreak"
        ),
        "pairing_contract": (
            "All methods reuse Exp7.3 A2 model-init and loader seeds. "
            "Case A is intended as a strict A2 replication."
        ),
        "hypotheses": {
            "B_minus_A": (
                "Controls for adding a factorized 128x128 then 128x12 linear head."
            ),
            "C_minus_B": (
                "Isolates the effect of per-timestep 128-way softmax normalization."
            ),
            "D_minus_C": (
                "Tests whether restoring deterministic latent RMS magnitude "
                "recovers performance lost by softmax normalization."
            ),
        },
    }
    _save_json(config.results_dir / "manifest.json", manifest)
    return manifest


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Exp7.4 latent softmax evidence ablation"
    )
    parser.add_argument("--repo-root", type=Path, default=None)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=exp72.BATCH_SIZE)
    parser.add_argument("--max-epochs", type=int, default=MAX_EPOCHS)
    parser.add_argument("--force", action="store_true")

    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run")
    run.add_argument("--array-task-id", type=int, required=True)
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

    if args.command == "run":
        specs = run_specs()
        if args.array_task_id < 0 or args.array_task_id >= len(specs):
            raise IndexError(args.array_task_id)
        run_one(specs[args.array_task_id], config, force=args.force)
        return
    if args.command == "finalize":
        finalize(config)
        return
    raise ValueError(args.command)


if __name__ == "__main__":
    main()
