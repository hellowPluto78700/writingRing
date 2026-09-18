from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import torch
from torch import nn
import torch.nn.functional as F

from scripts import experiment_0_1_general_comparison as exp01
from scripts import experiment_3_0_1_single_tau_objectives as exp3
from scripts import experiment_7_2_two_layer_tau_training as exp72
from scripts import experiment_7_3_training_strategy_decomposition as exp73
from scripts import experiment_8_0_local_backbone_tau_sweep as exp80
from scripts import experiment_8_1_hidden_quantization_ablation as exp81
from scripts import experiment_8_1_1_two_layer_mt_factorial as exp811
from scripts import experiment_9_0_within_user_generalization as exp90
from scripts import experiment_10_0_airborne_motion_ablation as exp10


EXPERIMENT_ID = "experiment_10_1_backbone_coding_supervision"
PROTOCOL_VERSION = "single_split_backbone_coding_supervision_v1"

VARIANT_ORIGINAL = exp10.VARIANT_ORIGINAL
VARIANT_POSTENCODE = exp10.VARIANT_POSTENCODE
VARIANTS = (VARIANT_ORIGINAL, VARIANT_POSTENCODE)

ROTATION = 0
CODINGS = ("bb", "mm")
OBJECTIVE_L2 = "l2_wcce"
OBJECTIVE_JOINT = "l1_l2_joint"
OBJECTIVE_L1_TSCE = "l2_wcce_plus_l1_tsce"
OBJECTIVES = (OBJECTIVE_L2, OBJECTIVE_JOINT, OBJECTIVE_L1_TSCE)
L1_TSCE_LAMBDA = 0.1

MODEL_SEEDS = (11, 23, 37)
EXPECTED_RUNS = len(VARIANTS) * len(CODINGS) * len(OBJECTIVES) * len(MODEL_SEEDS)

WIDTH = exp73.HIDDEN_WIDTH
SHIFTS = (2, 3, 4)
THRESHOLD_MULTIPLIERS = exp811.THRESHOLD_MULTIPLIERS
MAX_EPOCHS = exp73.MAX_EPOCHS
MIN_EPOCHS = exp73.MIN_EPOCHS
PATIENCE = exp73.PATIENCE
BATCH_SIZE = exp72.BATCH_SIZE
SPLITS = ("train", "val", "test")
LAYERS = ("l1", "l2")
PROBE_STATES = ("pre_reset", "communication")
PROBE_AGGREGATIONS = ("whole_mean", "fixed250_ordered_mean")


@dataclass(frozen=True)
class RunSpec:
    variant: str
    coding: str
    objective: str
    seed: int

    @property
    def key(self) -> str:
        return (
            f"{self.variant}__{self.coding}__{self.objective}"
            f"__rotation{ROTATION}__seed{self.seed}"
        )

    @property
    def l1_coding(self) -> str:
        return "binary" if self.coding == "bb" else "hetero3"

    @property
    def l2_coding(self) -> str:
        return "binary" if self.coding == "bb" else "hetero3"


@dataclass(frozen=True)
class Config:
    repo_root: Path
    results_dir: Path
    device: str = "cpu"
    threads: int = 1
    batch_size: int = BATCH_SIZE
    max_epochs: int = MAX_EPOCHS


def find_repo_root(start: Path | None = None) -> Path:
    return exp10.find_repo_root(start)


def results_dir(repo_root: Path) -> Path:
    return (
        repo_root
        / "notebooks"
        / "artifacts"
        / EXPERIMENT_ID
        / PROTOCOL_VERSION
    )


def run_specs() -> list[RunSpec]:
    return [
        RunSpec(variant, coding, objective, seed)
        for variant in VARIANTS
        for coding in CODINGS
        for objective in OBJECTIVES
        for seed in MODEL_SEEDS
    ]


def validate_spec(spec: RunSpec) -> None:
    if spec.variant not in VARIANTS:
        raise ValueError(spec.variant)
    if spec.coding not in CODINGS:
        raise ValueError(spec.coding)
    if spec.objective not in OBJECTIVES:
        raise ValueError(spec.objective)
    if spec.seed not in MODEL_SEEDS:
        raise ValueError(spec.seed)


def _save_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _path(root: Path, kind: str, key: str, suffix: str) -> Path:
    return root / kind / f"{key}{suffix}"


def _fold_assignment_path(config: Config) -> Path:
    return config.results_dir / "split" / "cross_user_fold_assignment.csv"


def _split_manifest_path(config: Config) -> Path:
    return config.results_dir / "split" / f"rotation{ROTATION}.csv"


def prepare_all(config: Config) -> dict[str, Any]:
    manifests: dict[str, pd.DataFrame] = {}
    labels_by_variant: dict[str, tuple[str, ...]] = {}
    for variant in VARIANTS:
        loaded, manifest, labels = exp10._load_variant_manifest(
            config.repo_root, variant
        )
        manifests[variant] = manifest
        labels_by_variant[variant] = labels
        del loaded

    reference = manifests[VARIANT_ORIGINAL]
    exp10._assert_paired_geometry(
        reference,
        manifests[VARIANT_POSTENCODE],
        reference_name=VARIANT_ORIGINAL,
        candidate_name=VARIANT_POSTENCODE,
    )
    if labels_by_variant[VARIANT_ORIGINAL] != labels_by_variant[VARIANT_POSTENCODE]:
        raise RuntimeError("D0/D1 class mapping mismatch")

    config.results_dir.mkdir(parents=True, exist_ok=True)
    split_root = config.results_dir / "split"
    split_root.mkdir(parents=True, exist_ok=True)

    canonical = reference.drop(columns=["pi", "si"]).sort_values("sample_id")
    canonical.to_csv(
        config.results_dir / "canonical_sample_manifest.csv", index=False
    )

    assignment = exp90._assign_cross_user_folds(reference)
    assignment.to_csv(_fold_assignment_path(config), index=False)
    split_manifest = exp90._apply_rotation(assignment, ROTATION)
    split_manifest.drop(columns=["pi", "si"]).to_csv(
        _split_manifest_path(config), index=False
    )

    users = {
        split: sorted(
            split_manifest.loc[split_manifest.split == split, "user"]
            .astype(str)
            .unique()
            .tolist()
        )
        for split in SPLITS
    }
    counts = split_manifest.split.value_counts()
    audit = {
        "experiment_id": EXPERIMENT_ID,
        "protocol_version": PROTOCOL_VERSION,
        "question": (
            "Under one locked cross-user split and time-shared readout, "
            "separate D1 input masking, two-layer MT3 communication, "
            "cooperative L1/L2 joint supervision, and 0.1 L1 timestep CE."
        ),
        "variants": list(VARIANTS),
        "variant_definitions": {
            VARIANT_ORIGINAL: "D0 = Encoder(a)",
            VARIANT_POSTENCODE: "D1 = m * Encoder(a)",
        },
        "rotation": ROTATION,
        "split_protocol": (
            "Exp10.0/Exp9.0 cross-user fold assignment; rotation0 only: "
            "test fold0, validation fold1, train folds2/3/4"
        ),
        "split_users": users,
        "split_samples": {
            split: int(counts.get(split, 0)) for split in SPLITS
        },
        "codings": {
            "bb": "binary L1 + binary L2",
            "mm": "MT3 L1 + MT3 L2, thresholds 0.5x/1.0x/1.5x",
        },
        "objectives": {
            OBJECTIVE_L2: "CE(mean_t W2 z2_t)",
            OBJECTIVE_JOINT: "CE(mean_t (W1 z1_t + W2 z2_t))",
            OBJECTIVE_L1_TSCE: (
                "CE(mean_t W2 z2_t) + 0.1 * mean_valid_t CE(W1 z1_t, y); "
                "W1 is training-only"
            ),
        },
        "l1_tsce_lambda": L1_TSCE_LAMBDA,
        "model_seeds": list(MODEL_SEEDS),
        "expected_runs": EXPECTED_RUNS,
        "architecture": "30->128->128->12",
        "shifts": list(SHIFTS),
        "readout": "time-shared only; no phase-aware readout",
        "model_init_seed_contract": (
            "exp73._e2e_pair_seed(seed, 'model_init'); "
            "variant/coding/objective excluded"
        ),
        "loader_seed_contract": (
            "exp73._raw_loaders(data, seed, ...); "
            "variant/coding/objective excluded"
        ),
    }
    _save_json(config.results_dir / "audit.json", audit)
    return audit


def _prepare_run_data(config: Config, variant: str) -> exp3.Data:
    loaded, manifest, labels = exp10._load_variant_manifest(
        config.repo_root, variant
    )
    canonical_path = config.results_dir / "canonical_sample_manifest.csv"
    if not canonical_path.exists():
        raise FileNotFoundError(
            f"Missing {canonical_path}; run Exp10.1 prepare first"
        )
    canonical = pd.read_csv(canonical_path)
    exp10._assert_paired_geometry(
        canonical,
        manifest,
        reference_name="prepared_canonical",
        candidate_name=variant,
    )
    saved = pd.read_csv(
        _fold_assignment_path(config), usecols=["sample_id", "cv_fold"]
    )
    assignment = manifest.merge(
        saved, on="sample_id", how="inner", validate="one_to_one"
    )
    if len(assignment) != len(manifest):
        raise RuntimeError("Prepared fold assignment does not match current data")
    assignment["cv_fold"] = assignment.cv_fold.astype(int)
    split_manifest = exp90._apply_rotation(assignment, ROTATION)
    data, _ = exp90._to_data(loaded, split_manifest, labels)
    return data


class Exp101Net(nn.Module):
    def __init__(self, spec: RunSpec, n_classes: int, fs: float) -> None:
        super().__init__()
        validate_spec(spec)
        self.spec = spec
        self.hidden_linears = nn.ModuleList(
            [
                nn.Linear(exp72.EXPECTED_CHANNELS, WIDTH, bias=False),
                nn.Linear(WIDTH, WIDTH, bias=False),
            ]
        )
        beta_hidden = float(
            np.exp(-(1000.0 / float(fs)) / float(exp72.TAU_MEM_MS))
        )
        self.l1_lif = exp811._lif(spec.l1_coding, beta_hidden)
        self.l2_lif = exp811._lif(spec.l2_coding, beta_hidden)
        self.register_buffer("alpha_0", exp811._alpha_vector(spec.l1_coding))
        self.register_buffer("alpha_1", exp811._alpha_vector(spec.l2_coding))

        # W2 is constructed before the optional W1 head so all shared model
        # parameters have identical initialization across objective cases.
        self.output_linear = nn.Linear(WIDTH, n_classes, bias=False)
        self.l1_output_linear: nn.Linear | None = None
        if spec.objective in {OBJECTIVE_JOINT, OBJECTIVE_L1_TSCE}:
            self.l1_output_linear = nn.Linear(WIDTH, n_classes, bias=False)

    def forward_trajectory(self, x: torch.Tensor) -> dict[str, Any]:
        batch, steps, channels = x.shape
        if channels != exp72.EXPECTED_CHANNELS:
            raise ValueError(channels)

        syn1 = torch.zeros(batch, WIDTH, device=x.device, dtype=x.dtype)
        mem1 = torch.zeros_like(syn1)
        syn2 = torch.zeros(batch, WIDTH, device=x.device, dtype=x.dtype)
        mem2 = torch.zeros_like(syn2)

        l1_comm: list[torch.Tensor] = []
        l2_comm: list[torch.Tensor] = []
        l1_pre: list[torch.Tensor] = []
        l2_pre: list[torch.Tensor] = []
        l2_evidence: list[torch.Tensor] = []
        l1_evidence: list[torch.Tensor] = []

        for timestep in range(steps):
            syn1 = self.alpha_0 * syn1 + self.hidden_linears[0](x[:, timestep])
            out1, mem1, pre1 = self.l1_lif(syn1, mem1)
            syn2 = self.alpha_1 * syn2 + self.hidden_linears[1](out1)
            out2, mem2, pre2 = self.l2_lif(syn2, mem2)

            l1_comm.append(out1)
            l2_comm.append(out2)
            l1_pre.append(pre1)
            l2_pre.append(pre2)
            l2_evidence.append(self.output_linear(out2))
            if self.l1_output_linear is not None:
                l1_evidence.append(self.l1_output_linear(out1))

        return {
            "hidden_communication": (
                torch.stack(l1_comm, dim=1),
                torch.stack(l2_comm, dim=1),
            ),
            "hidden_pre_reset": (
                torch.stack(l1_pre, dim=1),
                torch.stack(l2_pre, dim=1),
            ),
            "l2_evidence": torch.stack(l2_evidence, dim=1),
            "l1_evidence": (
                torch.stack(l1_evidence, dim=1)
                if l1_evidence
                else None
            ),
        }


def _valid_mask(lengths: torch.Tensor, n_steps: int) -> torch.Tensor:
    return torch.arange(
        n_steps, device=lengths.device
    )[None, :] < lengths[:, None]


def _valid_mean(
    values: torch.Tensor, lengths: torch.Tensor
) -> torch.Tensor:
    mask = _valid_mask(lengths, values.shape[1]).to(values.dtype).unsqueeze(-1)
    return (values * mask).sum(dim=1) / (
        lengths.clamp_min(1).to(values.dtype).unsqueeze(1)
    )


def _native_evidence(
    model: Exp101Net, trajectory: dict[str, Any]
) -> torch.Tensor:
    l2 = trajectory["l2_evidence"]
    if model.spec.objective == OBJECTIVE_JOINT:
        l1 = trajectory["l1_evidence"]
        if l1 is None:
            raise RuntimeError("Joint objective requires L1 evidence")
        return l1 + l2
    return l2


def _l1_timestep_ce(
    trajectory: dict[str, Any],
    lengths: torch.Tensor,
    y: torch.Tensor,
) -> torch.Tensor:
    logits = trajectory["l1_evidence"]
    if logits is None:
        raise RuntimeError("L1 TSCE requires L1 evidence")
    batch, steps, _ = logits.shape
    valid = _valid_mask(lengths, steps)
    targets = y[:, None].expand(batch, steps)
    return F.cross_entropy(logits[valid], targets[valid])


def _loss_terms(
    model: Exp101Net,
    trajectory: dict[str, Any],
    lengths: torch.Tensor,
    y: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor]:
    scores = _valid_mean(_native_evidence(model, trajectory), lengths)
    main_ce = F.cross_entropy(scores, y)
    aux_ce: torch.Tensor | None = None
    total = main_ce
    if model.spec.objective == OBJECTIVE_L1_TSCE:
        aux_ce = _l1_timestep_ce(trajectory, lengths, y)
        total = main_ce + L1_TSCE_LAMBDA * aux_ce
    return total, main_ce, aux_ce, scores


def _evaluate_native(
    model: Exp101Net,
    loader: Iterable,
    device: torch.device,
) -> dict[str, float]:
    ys: list[np.ndarray] = []
    preds: list[np.ndarray] = []
    total_sum = 0.0
    main_sum = 0.0
    aux_sum = 0.0
    n_total = 0

    model.eval()
    with torch.no_grad():
        for X, y, lengths in loader:
            X = X.to(device=device, dtype=torch.float32)
            y = y.to(device)
            lengths = lengths.to(device)
            trajectory = model.forward_trajectory(X)
            total, main_ce, aux_ce, scores = _loss_terms(
                model, trajectory, lengths, y
            )
            ys.append(y.cpu().numpy())
            preds.append(scores.argmax(dim=1).cpu().numpy())
            total_sum += float(total) * len(y)
            main_sum += float(main_ce) * len(y)
            if aux_ce is not None:
                aux_sum += float(aux_ce) * len(y)
            n_total += len(y)

    metrics = exp72._metrics(np.concatenate(ys), np.concatenate(preds))
    metrics["objective_loss"] = total_sum / max(n_total, 1)
    metrics["main_ce"] = main_sum / max(n_total, 1)
    if model.spec.objective == OBJECTIVE_L1_TSCE:
        metrics["l1_tsce"] = aux_sum / max(n_total, 1)
    return metrics


def _evaluate_lif_transfer(
    model: Exp101Net,
    loader: Iterable,
    device: torch.device,
) -> dict[str, float]:
    ys: list[np.ndarray] = []
    preds: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for X, y, lengths in loader:
            X = X.to(device=device, dtype=torch.float32)
            lengths_d = lengths.to(device)
            trajectory = model.forward_trajectory(X)
            evidence = _native_evidence(model, trajectory)
            spikes = exp81._output_lif_spikes(evidence)
            valid = _valid_mask(lengths_d, spikes.shape[1]).to(
                spikes.dtype
            ).unsqueeze(-1)
            scores = (spikes * valid).sum(dim=1)
            ys.append(y.numpy())
            preds.append(scores.argmax(dim=1).cpu().numpy())
    return exp72._metrics(np.concatenate(ys), np.concatenate(preds))


def _branch_removal_metrics(
    model: Exp101Net,
    loader: Iterable,
    device: torch.device,
) -> dict[str, dict[str, float]] | None:
    if model.spec.objective != OBJECTIVE_JOINT:
        return None

    ys: list[np.ndarray] = []
    full_preds: list[np.ndarray] = []
    l1_preds: list[np.ndarray] = []
    l2_preds: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for X, y, lengths in loader:
            X = X.to(device=device, dtype=torch.float32)
            lengths_d = lengths.to(device)
            trajectory = model.forward_trajectory(X)
            l1 = trajectory["l1_evidence"]
            if l1 is None:
                raise RuntimeError("Joint branch missing L1 evidence")
            l2 = trajectory["l2_evidence"]
            l1_scores = _valid_mean(l1, lengths_d)
            l2_scores = _valid_mean(l2, lengths_d)
            ys.append(y.numpy())
            l1_preds.append(l1_scores.argmax(dim=1).cpu().numpy())
            l2_preds.append(l2_scores.argmax(dim=1).cpu().numpy())
            full_preds.append((l1_scores + l2_scores).argmax(dim=1).cpu().numpy())

    y_true = np.concatenate(ys)
    return {
        "full": exp72._metrics(y_true, np.concatenate(full_preds)),
        "l1_only": exp72._metrics(y_true, np.concatenate(l1_preds)),
        "l2_only": exp72._metrics(y_true, np.concatenate(l2_preds)),
    }


def _fit_state_probes(
    state_splits: dict[
        str,
        dict[
            str,
            dict[str, tuple[torch.Tensor, np.ndarray, np.ndarray]],
        ],
    ],
    seed: int,
    bin_steps: int,
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for layer in LAYERS:
        result[layer] = {}
        for state in PROBE_STATES:
            result[layer][state] = {}
            for aggregation in PROBE_AGGREGATIONS:
                features = {
                    split: (
                        exp81._probe_features(
                            values, lengths, aggregation, bin_steps
                        ),
                        y,
                    )
                    for split, (values, y, lengths) in state_splits[
                        layer
                    ][state].items()
                }
                result[layer][state][aggregation] = exp01._fit_linear_probe(
                    features["train"][0],
                    features["train"][1],
                    features["val"][0],
                    features["val"][1],
                    features["test"][0],
                    features["test"][1],
                    seed=exp3.dseed(
                        seed,
                        EXPERIMENT_ID,
                        "probe",
                        layer,
                        state,
                        aggregation,
                    ),
                )
    return result


def _collect_activity(
    spec: RunSpec,
    state_splits: dict[
        str,
        dict[
            str,
            dict[str, tuple[torch.Tensor, np.ndarray, np.ndarray]],
        ],
    ],
) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for layer, coding in (
        ("l1", spec.l1_coding),
        ("l2", spec.l2_coding),
    ):
        values, _, lengths = state_splits[layer]["communication"]["test"]
        output[layer] = exp81._communication_stats(
            values,
            lengths,
            exp811._groups(coding),
            1,
        )
    return output


def _probe_ba(
    probes: dict[str, Any],
    layer: str,
    state: str,
    aggregation: str,
) -> float:
    return exp81._probe_ba(probes, layer, state, aggregation)


def run_one(
    spec: RunSpec,
    config: Config,
    force: bool = False,
) -> dict[str, Any]:
    validate_spec(spec)
    evaluation_path = _path(
        config.results_dir, "evaluations", spec.key, ".json"
    )
    checkpoint_path = _path(
        config.results_dir, "checkpoints", spec.key, ".pt"
    )
    history_path = _path(
        config.results_dir, "histories", spec.key, ".csv"
    )

    if (
        not force
        and evaluation_path.exists()
        and checkpoint_path.exists()
        and history_path.exists()
    ):
        return json.loads(evaluation_path.read_text(encoding="utf-8"))

    data = _prepare_run_data(config, spec.variant)
    torch.set_num_threads(config.threads)
    device = torch.device(config.device)

    # Shared parameters use the exact same initialization stream across
    # D0/D1, BB/MM, and all three objectives for a given seed.
    exp3.seed_all(exp73._e2e_pair_seed(spec.seed, "model_init"))
    model = Exp101Net(spec, len(data.labels), data.fs).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=exp72.LR,
        weight_decay=exp72.WEIGHT_DECAY,
    )
    train_loader = exp73._raw_loaders(
        data, spec.seed, config.batch_size, True
    )["train"]
    eval_loaders = exp73._raw_loaders(
        data, spec.seed, config.batch_size, False
    )

    best_state: dict[str, torch.Tensor] | None = None
    best_epoch = -1
    best_ba = -1.0
    best_loss = float("inf")
    stopped_epoch = config.max_epochs
    history: list[dict[str, float]] = []

    for epoch in range(1, config.max_epochs + 1):
        model.train()
        total_sum = 0.0
        main_sum = 0.0
        aux_sum = 0.0
        n_total = 0

        for X, y, lengths in train_loader:
            X = X.to(device=device, dtype=torch.float32)
            y = y.to(device)
            lengths = lengths.to(device)
            optimizer.zero_grad(set_to_none=True)
            trajectory = model.forward_trajectory(X)
            total, main_ce, aux_ce, _ = _loss_terms(
                model, trajectory, lengths, y
            )
            total.backward()
            optimizer.step()

            total_sum += float(total.detach()) * len(y)
            main_sum += float(main_ce.detach()) * len(y)
            if aux_ce is not None:
                aux_sum += float(aux_ce.detach()) * len(y)
            n_total += len(y)

        train_metrics = _evaluate_native(
            model, eval_loaders["train"], device
        )
        val_metrics = _evaluate_native(
            model, eval_loaders["val"], device
        )
        row: dict[str, float] = {
            "epoch": float(epoch),
            "train_ba": float(train_metrics["balanced_accuracy"]),
            "val_ba": float(val_metrics["balanced_accuracy"]),
            "train_loss": total_sum / max(n_total, 1),
            "val_loss": float(val_metrics["objective_loss"]),
            "train_main_ce": main_sum / max(n_total, 1),
            "val_main_ce": float(val_metrics["main_ce"]),
        }
        if spec.objective == OBJECTIVE_L1_TSCE:
            row["train_l1_tsce"] = aux_sum / max(n_total, 1)
            row["val_l1_tsce"] = float(val_metrics["l1_tsce"])
        history.append(row)

        if exp73._checkpoint_improved(
            val_metrics, best_ba, best_loss
        ):
            best_ba = float(val_metrics["balanced_accuracy"])
            best_loss = float(val_metrics["objective_loss"])
            best_epoch = epoch
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }

        if (
            epoch >= MIN_EPOCHS
            and best_epoch > 0
            and epoch - best_epoch >= PATIENCE
        ):
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
            "rotation": ROTATION,
            "l1_tsce_lambda": L1_TSCE_LAMBDA,
            "threshold_multipliers": list(THRESHOLD_MULTIPLIERS),
            "best_epoch": best_epoch,
            "stopped_epoch": stopped_epoch,
            "best_val_ba": best_ba,
            "best_val_objective_loss": best_loss,
            "model_state_dict": best_state,
        },
        checkpoint_path,
    )
    history_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(history).to_csv(history_path, index=False)

    model.load_state_dict(best_state, strict=True)
    native_metrics = {
        split: _evaluate_native(model, loader, device)
        for split, loader in eval_loaders.items()
    }
    lif_metrics = {
        split: _evaluate_lif_transfer(model, loader, device)
        for split, loader in eval_loaders.items()
    }
    state_splits = exp81._extract_state_splits(
        model, eval_loaders, device
    )
    probes = _fit_state_probes(
        state_splits, spec.seed, data.bin_steps
    )
    activity = _collect_activity(spec, state_splits)
    branch_metrics = _branch_removal_metrics(
        model, eval_loaders["test"], device
    )

    total_parameters = int(
        sum(parameter.numel() for parameter in model.parameters())
    )
    inference_parameters = total_parameters
    if (
        spec.objective == OBJECTIVE_L1_TSCE
        and model.l1_output_linear is not None
    ):
        inference_parameters -= int(
            sum(
                parameter.numel()
                for parameter in model.l1_output_linear.parameters()
            )
        )

    payload = {
        "experiment_id": EXPERIMENT_ID,
        "protocol_version": PROTOCOL_VERSION,
        "spec": asdict(spec),
        "rotation": ROTATION,
        "l1_coding": spec.l1_coding,
        "l2_coding": spec.l2_coding,
        "threshold_multipliers": list(THRESHOLD_MULTIPLIERS),
        "l1_tsce_lambda": L1_TSCE_LAMBDA,
        "readout": "time_shared",
        "best_epoch": best_epoch,
        "stopped_epoch": stopped_epoch,
        "trainable_parameter_count": total_parameters,
        "inference_parameter_count": inference_parameters,
        "native_metrics": native_metrics,
        "lif_transfer_metrics": lif_metrics,
        "representation_probes": probes,
        "activity": activity,
        "joint_branch_metrics": branch_metrics,
    }
    _save_json(evaluation_path, payload)
    return payload


def _row(payload: dict[str, Any]) -> dict[str, Any]:
    spec = payload["spec"]
    probes = payload["representation_probes"]
    row: dict[str, Any] = {
        "variant": spec["variant"],
        "coding": spec["coding"],
        "objective": spec["objective"],
        "seed": int(spec["seed"]),
        "rotation": ROTATION,
        "trainable_parameter_count": payload[
            "trainable_parameter_count"
        ],
        "inference_parameter_count": payload[
            "inference_parameter_count"
        ],
        "best_epoch": int(payload["best_epoch"]),
        "stopped_epoch": int(payload["stopped_epoch"]),
    }
    for split in SPLITS:
        metrics = payload["native_metrics"][split]
        for metric in (
            "accuracy",
            "balanced_accuracy",
            "macro_f1",
            "objective_loss",
        ):
            row[f"native_{split}_{metric}"] = float(metrics[metric])
    row["lif_test_ba"] = float(
        payload["lif_transfer_metrics"]["test"]["balanced_accuracy"]
    )

    for layer in LAYERS:
        for state in PROBE_STATES:
            for aggregation in PROBE_AGGREGATIONS:
                short = (
                    "whole"
                    if aggregation == "whole_mean"
                    else "fixed250"
                )
                row[
                    f"{layer}_{state}_{short}_ba"
                ] = _probe_ba(
                    probes, layer, state, aggregation
                )

    row["l1_quantization_gap_fixed250"] = (
        row["l1_pre_reset_fixed250_ba"]
        - row["l1_communication_fixed250_ba"]
    )
    row["l1_to_l2_transform_delta_fixed250"] = (
        row["l2_pre_reset_fixed250_ba"]
        - row["l1_communication_fixed250_ba"]
    )
    row["l2_quantization_gap_fixed250"] = (
        row["l2_pre_reset_fixed250_ba"]
        - row["l2_communication_fixed250_ba"]
    )

    branches = payload.get("joint_branch_metrics")
    if branches is not None:
        row["joint_full_test_ba"] = float(
            branches["full"]["balanced_accuracy"]
        )
        row["joint_l1_only_test_ba"] = float(
            branches["l1_only"]["balanced_accuracy"]
        )
        row["joint_l2_only_test_ba"] = float(
            branches["l2_only"]["balanced_accuracy"]
        )
        row["joint_remove_l1_delta"] = (
            row["joint_full_test_ba"]
            - row["joint_l2_only_test_ba"]
        )
    return row


def _activity_rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
    spec = payload["spec"]
    rows: list[dict[str, Any]] = []
    for layer, groups in payload["activity"].items():
        for group in groups:
            rows.append(
                {
                    "variant": spec["variant"],
                    "coding": spec["coding"],
                    "objective": spec["objective"],
                    "seed": int(spec["seed"]),
                    "layer": layer,
                    **group,
                }
            )
    return rows


def _paired_contrast_rows(runs: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        "native_test_balanced_accuracy",
        "l1_communication_fixed250_ba",
        "l2_communication_fixed250_ba",
        "l1_quantization_gap_fixed250",
        "l2_quantization_gap_fixed250",
    ]
    rows: list[dict[str, Any]] = []

    indexed = runs.set_index(
        ["variant", "coding", "objective", "seed"]
    )

    def emit(
        contrast: str,
        left: tuple[str, str, str, int],
        right: tuple[str, str, str, int],
    ) -> None:
        left_row = indexed.loc[left]
        right_row = indexed.loc[right]
        row: dict[str, Any] = {
            "contrast": contrast,
            "variant": left[0]
            if left[0] == right[0]
            else "paired_D1_vs_D0",
            "coding": left[1]
            if left[1] == right[1]
            else "paired_MM_vs_BB",
            "objective": left[2]
            if left[2] == right[2]
            else f"{left[2]}_vs_{right[2]}",
            "seed": left[3],
        }
        for metric in metrics:
            row[f"delta_{metric}"] = float(
                left_row[metric] - right_row[metric]
            )
        rows.append(row)

    for seed in MODEL_SEEDS:
        for variant in VARIANTS:
            for coding in CODINGS:
                emit(
                    "joint_minus_l2_wcce",
                    (variant, coding, OBJECTIVE_JOINT, seed),
                    (variant, coding, OBJECTIVE_L2, seed),
                )
                emit(
                    "l1_tsce_minus_l2_wcce",
                    (variant, coding, OBJECTIVE_L1_TSCE, seed),
                    (variant, coding, OBJECTIVE_L2, seed),
                )
        for variant in VARIANTS:
            for objective in OBJECTIVES:
                emit(
                    "mm_minus_bb",
                    (variant, "mm", objective, seed),
                    (variant, "bb", objective, seed),
                )
        for coding in CODINGS:
            for objective in OBJECTIVES:
                emit(
                    "d1_minus_d0",
                    (
                        VARIANT_POSTENCODE,
                        coding,
                        objective,
                        seed,
                    ),
                    (
                        VARIANT_ORIGINAL,
                        coding,
                        objective,
                        seed,
                    ),
                )

    return pd.DataFrame(rows)


def _interaction_rows(runs: pd.DataFrame) -> pd.DataFrame:
    indexed = runs.set_index(
        ["variant", "coding", "objective", "seed"]
    )
    rows: list[dict[str, Any]] = []
    metrics = [
        "native_test_balanced_accuracy",
        "l1_communication_fixed250_ba",
        "l2_communication_fixed250_ba",
    ]
    for variant in VARIANTS:
        for seed in MODEL_SEEDS:
            for objective, name in (
                (OBJECTIVE_JOINT, "mt_x_joint"),
                (OBJECTIVE_L1_TSCE, "mt_x_l1_tsce"),
            ):
                row: dict[str, Any] = {
                    "interaction": name,
                    "variant": variant,
                    "seed": seed,
                }
                for metric in metrics:
                    treated = float(
                        indexed.loc[
                            (variant, "mm", objective, seed),
                            metric,
                        ]
                        - indexed.loc[
                            (variant, "bb", objective, seed),
                            metric,
                        ]
                    )
                    baseline = float(
                        indexed.loc[
                            (variant, "mm", OBJECTIVE_L2, seed),
                            metric,
                        ]
                        - indexed.loc[
                            (variant, "bb", OBJECTIVE_L2, seed),
                            metric,
                        ]
                    )
                    row[f"interaction_{metric}"] = (
                        treated - baseline
                    )
                rows.append(row)
    return pd.DataFrame(rows)


def finalize(config: Config) -> dict[str, Any]:
    payloads: list[dict[str, Any]] = []
    for spec in run_specs():
        path = _path(
            config.results_dir, "evaluations", spec.key, ".json"
        )
        if not path.exists():
            raise FileNotFoundError(path)
        payloads.append(
            json.loads(path.read_text(encoding="utf-8"))
        )

    runs = pd.DataFrame([_row(payload) for payload in payloads])
    if len(runs) != EXPECTED_RUNS:
        raise RuntimeError(
            f"Expected {EXPECTED_RUNS} runs, got {len(runs)}"
        )
    runs.to_csv(config.results_dir / "run_metrics.csv", index=False)

    group_cols = ["variant", "coding", "objective"]
    metric_cols = [
        "native_train_balanced_accuracy",
        "native_val_balanced_accuracy",
        "native_test_balanced_accuracy",
        "lif_test_ba",
        "l1_pre_reset_fixed250_ba",
        "l1_communication_fixed250_ba",
        "l2_pre_reset_fixed250_ba",
        "l2_communication_fixed250_ba",
        "l1_quantization_gap_fixed250",
        "l1_to_l2_transform_delta_fixed250",
        "l2_quantization_gap_fixed250",
    ]
    summary = (
        runs.groupby(group_cols, sort=False)[metric_cols]
        .agg(["mean", "std"])
        .reset_index()
    )
    summary.columns = [
        "_".join(str(part) for part in column if str(part))
        if isinstance(column, tuple)
        else str(column)
        for column in summary.columns
    ]
    summary.to_csv(
        config.results_dir / "method_summary.csv", index=False
    )

    paired = _paired_contrast_rows(runs)
    paired.to_csv(
        config.results_dir / "paired_contrasts.csv", index=False
    )
    delta_cols = [
        column for column in paired.columns if column.startswith("delta_")
    ]
    paired_summary = (
        paired.groupby(["contrast"], sort=False)[delta_cols]
        .agg(["count", "mean", "std"])
        .reset_index()
    )
    paired_summary.columns = [
        "_".join(str(part) for part in column if str(part))
        if isinstance(column, tuple)
        else str(column)
        for column in paired_summary.columns
    ]
    paired_summary.to_csv(
        config.results_dir / "paired_contrast_summary.csv",
        index=False,
    )

    interactions = _interaction_rows(runs)
    interactions.to_csv(
        config.results_dir / "interaction_runs.csv", index=False
    )
    interaction_metric_cols = [
        column
        for column in interactions.columns
        if column.startswith("interaction_")
    ]
    interaction_summary = (
        interactions.groupby(
            ["interaction", "variant"], sort=False
        )[interaction_metric_cols]
        .agg(["count", "mean", "std"])
        .reset_index()
    )
    interaction_summary.columns = [
        "_".join(str(part) for part in column if str(part))
        if isinstance(column, tuple)
        else str(column)
        for column in interaction_summary.columns
    ]
    interaction_summary.to_csv(
        config.results_dir / "interaction_summary.csv",
        index=False,
    )

    activity = pd.DataFrame(
        [
            row
            for payload in payloads
            for row in _activity_rows(payload)
        ]
    )
    activity.to_csv(
        config.results_dir / "activity_runs.csv", index=False
    )

    joint_rows = runs[runs.objective == OBJECTIVE_JOINT][
        [
            "variant",
            "coding",
            "seed",
            "joint_full_test_ba",
            "joint_l1_only_test_ba",
            "joint_l2_only_test_ba",
            "joint_remove_l1_delta",
        ]
    ]
    joint_rows.to_csv(
        config.results_dir / "joint_branch_removal.csv",
        index=False,
    )

    manifest = {
        "experiment_id": EXPERIMENT_ID,
        "protocol_version": PROTOCOL_VERSION,
        "status": "PASS",
        "rotation": ROTATION,
        "variants": list(VARIANTS),
        "codings": list(CODINGS),
        "objectives": list(OBJECTIVES),
        "model_seeds": list(MODEL_SEEDS),
        "run_count": int(len(runs)),
        "expected_run_count": EXPECTED_RUNS,
        "readout": "time_shared_only",
        "phase_aware_readout": False,
        "l1_tsce_lambda": L1_TSCE_LAMBDA,
        "primary_metric": "native_test_balanced_accuracy",
        "statistical_scope": (
            "single fixed cross-user split; three paired model seeds are "
            "optimization replicates, not independent user splits"
        ),
        "primary_outputs": [
            "run_metrics.csv",
            "method_summary.csv",
            "paired_contrasts.csv",
            "paired_contrast_summary.csv",
            "interaction_runs.csv",
            "interaction_summary.csv",
            "activity_runs.csv",
            "joint_branch_removal.csv",
        ],
    }
    _save_json(config.results_dir / "manifest.json", manifest)
    return manifest


def _resolve_config(args: argparse.Namespace) -> Config:
    repo_root = (
        Path(args.repo_root).resolve()
        if args.repo_root
        else find_repo_root()
    )
    output = (
        Path(args.results_dir).resolve()
        if args.results_dir
        else results_dir(repo_root)
    )
    return Config(
        repo_root=repo_root,
        results_dir=output,
        device=args.device,
        threads=args.threads,
        batch_size=args.batch_size,
        max_epochs=args.max_epochs,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Exp10.1 one-split D0/D1 x BB/MM x "
            "L2-WCCE/Joint/L1-TSCE backbone ablation"
        )
    )
    parser.add_argument("--repo-root", default=None)
    parser.add_argument("--results-dir", default=None)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--max-epochs", type=int, default=MAX_EPOCHS)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("prepare")
    sub.add_parser("list-runs")
    run = sub.add_parser("run-one")
    run.add_argument("--array-task-id", type=int, required=True)
    run.add_argument("--force", action="store_true")
    sub.add_parser("finalize")
    return parser


def main() -> None:
    args = _parser().parse_args()
    config = _resolve_config(args)

    if args.command == "prepare":
        print(
            json.dumps(
                prepare_all(config),
                indent=2,
                sort_keys=True,
            )
        )
        return

    if args.command == "list-runs":
        for index, spec in enumerate(run_specs()):
            print(index, spec.key)
        return

    if args.command == "run-one":
        specs = run_specs()
        if not 0 <= args.array_task_id < len(specs):
            raise IndexError(args.array_task_id)
        spec = specs[args.array_task_id]
        payload = run_one(spec, config, force=args.force)
        print(
            json.dumps(
                {
                    "key": spec.key,
                    "best_epoch": payload["best_epoch"],
                    "native_test_ba": payload["native_metrics"][
                        "test"
                    ]["balanced_accuracy"],
                },
                indent=2,
                sort_keys=True,
            )
        )
        return

    if args.command == "finalize":
        print(
            json.dumps(
                finalize(config),
                indent=2,
                sort_keys=True,
            )
        )
        return

    raise ValueError(args.command)


if __name__ == "__main__":
    main()
