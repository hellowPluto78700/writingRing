from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd
import torch
from torch import nn
import torch.nn.functional as F

from scripts import experiment_3_0_1_single_tau_objectives as exp3
from scripts import experiment_7_2_two_layer_tau_training as exp72
from scripts import experiment_7_3_training_strategy_decomposition as exp73
from scripts import experiment_7_3_5_hidden_state_information_loss as exp735
from scripts import experiment_8_1_hidden_quantization_ablation as exp81
from scripts import experiment_8_1_1_two_layer_mt_factorial as exp811
from scripts import experiment_9_0_within_user_generalization as exp90
from scripts import experiment_10_0_airborne_motion_ablation as exp10


EXPERIMENT_ID = "experiment_10_1_a2_backbone_coding_supervision"
PROTOCOL_VERSION = "d0_d1_bb_mm_supervision_single_split_v1"

VARIANT_ORIGINAL = exp10.VARIANT_ORIGINAL
VARIANT_POSTENCODE = exp10.VARIANT_POSTENCODE
VARIANTS = (VARIANT_ORIGINAL, VARIANT_POSTENCODE)

CODING_BB = "bb"
CODING_MM = "mm"
CODINGS = (CODING_BB, CODING_MM)

OBJECTIVE_BASELINE = "l2_wcce"
OBJECTIVE_JOINT = "l1_l2_joint"
OBJECTIVE_L1_TSCE = "l2_wcce_plus_0p1_l1_tsce"
OBJECTIVES = (OBJECTIVE_BASELINE, OBJECTIVE_JOINT, OBJECTIVE_L1_TSCE)
TSCE_LAMBDA = 0.1

ROTATIONS = (0,)
MODEL_SEEDS = (11, 23, 37)
EXPECTED_RUNS = (
    len(VARIANTS) * len(CODINGS) * len(OBJECTIVES) * len(ROTATIONS) * len(MODEL_SEEDS)
)

WIDTH = exp811.WIDTH
SHIFTS = exp811.SHIFTS
THRESHOLD = exp811.THRESHOLD
THRESHOLD_MULTIPLIERS = exp811.THRESHOLD_MULTIPLIERS
MAX_EPOCHS = exp73.MAX_EPOCHS
MIN_EPOCHS = exp73.MIN_EPOCHS
PATIENCE = exp73.PATIENCE
BATCH_SIZE = exp72.BATCH_SIZE
SPLITS = ("train", "val", "test")
HIDDEN_LAYERS = ("l1", "l2")
HIDDEN_STATES = ("syn_current", "pre_reset", "spike", "post_reset")


@dataclass(frozen=True)
class RunSpec:
    variant: str
    coding: str
    objective: str
    rotation: int
    seed: int

    @property
    def key(self) -> str:
        return (
            f"{self.variant}__{self.coding}__{self.objective}__"
            f"rotation{self.rotation}__seed{self.seed}"
        )

    @property
    def l1_coding(self) -> str:
        return "binary" if self.coding == CODING_BB else "hetero3"

    @property
    def l2_coding(self) -> str:
        return "binary" if self.coding == CODING_BB else "hetero3"


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
    return repo_root / "notebooks" / "artifacts" / EXPERIMENT_ID / PROTOCOL_VERSION


def run_specs() -> list[RunSpec]:
    return [
        RunSpec(variant, coding, objective, rotation, seed)
        for variant in VARIANTS
        for coding in CODINGS
        for objective in OBJECTIVES
        for rotation in ROTATIONS
        for seed in MODEL_SEEDS
    ]


def validate_spec(spec: RunSpec) -> None:
    if spec.variant not in VARIANTS:
        raise ValueError(spec.variant)
    if spec.coding not in CODINGS:
        raise ValueError(spec.coding)
    if spec.objective not in OBJECTIVES:
        raise ValueError(spec.objective)
    if spec.rotation not in ROTATIONS:
        raise ValueError(spec.rotation)
    if spec.seed not in MODEL_SEEDS:
        raise ValueError(spec.seed)


def _save_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _path(root: Path, kind: str, key: str, suffix: str) -> Path:
    return root / kind / f"{key}{suffix}"


def _sha_rows(values: Iterable[str]) -> str:
    digest = hashlib.sha256()
    for value in values:
        digest.update(str(value).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def _variant_roots(repo_root: Path, variant: str) -> list[Path]:
    if variant not in VARIANTS:
        raise ValueError(variant)
    return exp10._variant_roots(repo_root, variant)


def _load_variant_manifest(
    repo_root: Path,
    variant: str,
) -> tuple[Any, pd.DataFrame, tuple[str, ...]]:
    if variant not in VARIANTS:
        raise ValueError(variant)
    return exp10._load_variant_manifest(repo_root, variant)


def _fold_assignment_path(config: Config) -> Path:
    return config.results_dir / "split" / "cross_user_fold_assignment.csv"


def _rotation_manifest_path(config: Config, rotation: int) -> Path:
    return config.results_dir / "split" / f"rotation{rotation}.csv"


def prepare_all(config: Config) -> dict[str, Any]:
    manifests: dict[str, pd.DataFrame] = {}
    labels_by_variant: dict[str, tuple[str, ...]] = {}
    for variant in VARIANTS:
        loaded, manifest, labels = _load_variant_manifest(config.repo_root, variant)
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
    if labels_by_variant[VARIANT_POSTENCODE] != labels_by_variant[VARIANT_ORIGINAL]:
        raise RuntimeError("D0/D1 class mapping mismatch")

    config.results_dir.mkdir(parents=True, exist_ok=True)
    split_root = config.results_dir / "split"
    split_root.mkdir(parents=True, exist_ok=True)

    canonical = (
        reference.drop(columns=["pi", "si"])
        .sort_values("sample_id")
        .reset_index(drop=True)
    )
    canonical.to_csv(config.results_dir / "canonical_sample_manifest.csv", index=False)

    assignment = exp90._assign_cross_user_folds(reference)
    assignment.to_csv(_fold_assignment_path(config), index=False)
    (
        assignment[["user", "cv_fold"]]
        .drop_duplicates()
        .sort_values(["cv_fold", "user"])
        .to_csv(split_root / "cross_user_fold_users.csv", index=False)
    )

    rotation_rows: list[dict[str, Any]] = []
    for rotation in ROTATIONS:
        split_manifest = exp90._apply_rotation(assignment, rotation)
        split_manifest.drop(columns=["pi", "si"]).to_csv(
            _rotation_manifest_path(config, rotation),
            index=False,
        )
        counts = split_manifest.split.value_counts()
        rotation_rows.append(
            {
                "rotation": rotation,
                "test_fold": rotation,
                "val_fold": (rotation + 1) % exp90.N_FOLDS,
                "train_samples": int(counts.get("train", 0)),
                "val_samples": int(counts.get("val", 0)),
                "test_samples": int(counts.get("test", 0)),
                "train_users": int(
                    split_manifest.loc[split_manifest.split == "train", "user"].nunique()
                ),
                "val_users": int(
                    split_manifest.loc[split_manifest.split == "val", "user"].nunique()
                ),
                "test_users": int(
                    split_manifest.loc[split_manifest.split == "test", "user"].nunique()
                ),
            }
        )
    pd.DataFrame(rotation_rows).to_csv(
        config.results_dir / "rotation_summary.csv", index=False
    )

    audit = {
        "experiment_id": EXPERIMENT_ID,
        "protocol_version": PROTOCOL_VERSION,
        "question": (
            "Factorially test D0 versus D1, binary versus two-layer MT3 coding, "
            "and baseline/joint/L1-TSCE supervision while keeping a time-shared readout."
        ),
        "variants": list(VARIANTS),
        "variant_definitions": {
            VARIANT_ORIGINAL: "D0 = Encoder(a)",
            VARIANT_POSTENCODE: "D1 = m * Encoder(a)",
        },
        "codings": {
            CODING_BB: {"l1": "binary", "l2": "binary"},
            CODING_MM: {"l1": "MT3", "l2": "MT3"},
        },
        "threshold_multipliers": list(THRESHOLD_MULTIPLIERS),
        "objectives": {
            OBJECTIVE_BASELINE: "CE(mean_t(W2 z2_t), y)",
            OBJECTIVE_JOINT: "CE(mean_t(W1 z1_t + W2 z2_t), y)",
            OBJECTIVE_L1_TSCE: (
                "CE(mean_t(W2 z2_t), y) + 0.1 * mean_valid_t CE(W1 z1_t, y); "
                "W1 is training-only at inference"
            ),
        },
        "tsce_lambda": TSCE_LAMBDA,
        "readout_scope": "time-shared only; no phase-aware readout",
        "combined_actions": [0, 1],
        "total_samples": int(len(reference)),
        "total_users": int(reference.user.nunique()),
        "labels": list(labels_by_variant[VARIANT_ORIGINAL]),
        "rotations": list(ROTATIONS),
        "model_seeds": list(MODEL_SEEDS),
        "expected_runs": EXPECTED_RUNS,
        "cross_user_split_unit": "user",
        "split_protocol": "Exp10.0/Exp9.0 cross-user fold assignment; rotation0 only (test fold0, validation fold1, train folds2/3/4)",
        "architecture": {
            "input_channels": exp72.EXPECTED_CHANNELS,
            "l1_width": WIDTH,
            "l2_width": WIDTH,
            "l1_shifts": list(SHIFTS),
            "l2_shifts": list(SHIFTS),
        },
        "model_init_seed_contract": (
            "exp73._e2e_pair_seed(seed, 'model_init'); "
            "variant/coding/objective/rotation excluded"
        ),
        "loader_seed_contract": (
            "exp73._raw_loaders(data, seed, ...); "
            "variant/coding/objective/rotation excluded"
        ),
        "probe_inventory": list(exp10.probe_names()),
        "probe_count_per_run": len(exp10.probe_names()),
    }
    _save_json(config.results_dir / "audit.json", audit)
    return audit


def _load_saved_assignment(config: Config, manifest: pd.DataFrame) -> pd.DataFrame:
    path = _fold_assignment_path(config)
    if not path.exists():
        raise FileNotFoundError(f"Missing Exp10.1 split assignment {path}; run prepare first")
    saved = pd.read_csv(path, usecols=["sample_id", "cv_fold"])
    if saved.sample_id.duplicated().any():
        raise RuntimeError(f"Duplicate sample_id in {path}")
    merged = manifest.merge(saved, on="sample_id", how="inner", validate="one_to_one")
    if len(merged) != len(manifest):
        raise RuntimeError("Saved fold assignment does not match current dataset")
    merged["cv_fold"] = merged.cv_fold.astype(int)
    return merged


def _prepare_run_data(
    config: Config,
    variant: str,
    rotation: int,
) -> tuple[exp3.Data, dict[str, pd.DataFrame], pd.DataFrame]:
    loaded, manifest, labels = _load_variant_manifest(config.repo_root, variant)
    canonical_path = config.results_dir / "canonical_sample_manifest.csv"
    if not canonical_path.exists():
        raise FileNotFoundError(f"Missing {canonical_path}; run prepare first")
    canonical = pd.read_csv(canonical_path)
    exp10._assert_paired_geometry(
        canonical,
        manifest,
        reference_name="prepared_canonical",
        candidate_name=variant,
    )
    assignment = _load_saved_assignment(config, manifest)
    split_manifest = exp90._apply_rotation(assignment, rotation)
    data, frames = exp90._to_data(loaded, split_manifest, labels)
    return data, frames, split_manifest


def _split_hashes(frames: Mapping[str, pd.DataFrame]) -> dict[str, str]:
    return {
        split: _sha_rows(frames[split].sample_id.astype(str).tolist())
        for split in SPLITS
    }


def _valid_mask(lengths: torch.Tensor, steps: int) -> torch.Tensor:
    return torch.arange(steps, device=lengths.device)[None, :] < lengths[:, None]


def _valid_mean(values: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
    mask = _valid_mask(lengths, values.shape[1]).to(values.dtype).unsqueeze(-1)
    return (values * mask).sum(dim=1) / lengths.clamp_min(1).to(values.dtype).unsqueeze(1)


def _l1_tsce(
    evidence: torch.Tensor,
    lengths: torch.Tensor,
    y: torch.Tensor,
) -> torch.Tensor:
    mask = _valid_mask(lengths, evidence.shape[1])
    targets = y[:, None].expand(-1, evidence.shape[1])
    if not bool(mask.any()):
        raise RuntimeError("L1 TSCE received a batch with no valid timesteps")
    return F.cross_entropy(evidence[mask], targets[mask])


class Exp101Net(nn.Module):
    """A2 234x234 backbone with BB/MM coding and time-shared L1/L2 heads."""

    def __init__(self, spec: RunSpec, n_classes: int, fs: float) -> None:
        super().__init__()
        validate_spec(spec)
        self.spec = spec
        self.n_classes = int(n_classes)
        self.fs = float(fs)

        # Create all shared learned layers before the optional L1 head so paired
        # seeds preserve identical A2 shared initialization across objectives.
        self.hidden_linears = nn.ModuleList(
            [
                nn.Linear(exp72.EXPECTED_CHANNELS, WIDTH, bias=False),
                nn.Linear(WIDTH, WIDTH, bias=False),
            ]
        )
        self.output_linear = nn.Linear(WIDTH, n_classes, bias=False)
        if spec.objective in {OBJECTIVE_JOINT, OBJECTIVE_L1_TSCE}:
            self.l1_output_linear: nn.Linear | None = nn.Linear(
                WIDTH, n_classes, bias=False
            )
        else:
            self.l1_output_linear = None

        beta_hidden = math.exp(-(1000.0 / fs) / exp72.TAU_MEM_MS)
        self.l1_lif = exp811._lif(spec.l1_coding, beta_hidden)
        self.l2_lif = exp811._lif(spec.l2_coding, beta_hidden)
        self.register_buffer("alpha_0", exp811._alpha_vector(spec.l1_coding))
        self.register_buffer("alpha_1", exp811._alpha_vector(spec.l2_coding))

    def forward_trajectory(self, x: torch.Tensor) -> dict[str, Any]:
        batch, steps, channels = x.shape
        if channels != exp72.EXPECTED_CHANNELS:
            raise ValueError(channels)

        syn1 = torch.zeros(batch, WIDTH, device=x.device, dtype=x.dtype)
        mem1 = torch.zeros_like(syn1)
        syn2 = torch.zeros(batch, WIDTH, device=x.device, dtype=x.dtype)
        mem2 = torch.zeros_like(syn2)

        states: dict[str, dict[str, list[torch.Tensor]]] = {
            layer: {state: [] for state in HIDDEN_STATES}
            for layer in HIDDEN_LAYERS
        }
        l2_evidence: list[torch.Tensor] = []
        l1_evidence: list[torch.Tensor] = []

        for timestep in range(steps):
            syn1 = self.alpha_0 * syn1 + self.hidden_linears[0](x[:, timestep])
            out1, post1, pre1 = self.l1_lif(syn1, mem1)
            mem1 = post1

            syn2 = self.alpha_1 * syn2 + self.hidden_linears[1](out1)
            out2, post2, pre2 = self.l2_lif(syn2, mem2)
            mem2 = post2

            states["l1"]["syn_current"].append(syn1)
            states["l1"]["pre_reset"].append(pre1)
            states["l1"]["spike"].append(out1)
            states["l1"]["post_reset"].append(post1)
            states["l2"]["syn_current"].append(syn2)
            states["l2"]["pre_reset"].append(pre2)
            states["l2"]["spike"].append(out2)
            states["l2"]["post_reset"].append(post2)

            l2_evidence.append(self.output_linear(out2))
            if self.l1_output_linear is not None:
                l1_evidence.append(self.l1_output_linear(out1))

        hidden = {
            layer: {
                state: torch.stack(values, dim=1)
                for state, values in state_map.items()
            }
            for layer, state_map in states.items()
        }
        return {
            "hidden": hidden,
            "l2_evidence": torch.stack(l2_evidence, dim=1),
            "l1_evidence": (
                torch.stack(l1_evidence, dim=1)
                if self.l1_output_linear is not None
                else None
            ),
        }


def _native_evidence(
    model: Exp101Net,
    trajectory: dict[str, Any],
) -> torch.Tensor:
    l2 = trajectory["l2_evidence"]
    if model.spec.objective == OBJECTIVE_JOINT:
        l1 = trajectory["l1_evidence"]
        if l1 is None:
            raise RuntimeError("Joint objective requires L1 evidence")
        return l1 + l2
    return l2


def _loss_terms(
    model: Exp101Net,
    trajectory: dict[str, Any],
    lengths: torch.Tensor,
    y: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    main_scores = _valid_mean(_native_evidence(model, trajectory), lengths)
    main_ce = F.cross_entropy(main_scores, y)
    aux: torch.Tensor | None = None
    total = main_ce
    if model.spec.objective == OBJECTIVE_L1_TSCE:
        l1 = trajectory["l1_evidence"]
        if l1 is None:
            raise RuntimeError("L1 TSCE objective requires L1 evidence")
        aux = _l1_tsce(l1, lengths, y)
        total = main_ce + TSCE_LAMBDA * aux
    return total, main_ce, aux


def _evaluate_native(
    model: Exp101Net,
    loader: Iterable,
    device: torch.device,
) -> tuple[dict[str, float], np.ndarray, np.ndarray]:
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
            scores = _valid_mean(_native_evidence(model, trajectory), lengths)
            total, main_ce, aux = _loss_terms(model, trajectory, lengths, y)
            ys.append(y.cpu().numpy())
            preds.append(scores.argmax(dim=1).cpu().numpy())
            total_sum += float(total) * len(y)
            main_sum += float(main_ce) * len(y)
            if aux is not None:
                aux_sum += float(aux) * len(y)
            n_total += len(y)

    y_true = np.concatenate(ys)
    y_pred = np.concatenate(preds)
    metrics = exp10._classification_metrics(y_true, y_pred)
    metrics["objective_loss"] = total_sum / max(n_total, 1)
    metrics["main_ce"] = main_sum / max(n_total, 1)
    if model.spec.objective == OBJECTIVE_L1_TSCE:
        metrics["l1_tsce"] = aux_sum / max(n_total, 1)
    return metrics, y_true, y_pred


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
            lengths_device = lengths.to(device)
            trajectory = model.forward_trajectory(X)
            evidence = _native_evidence(model, trajectory)
            spikes = exp81._output_lif_spikes(evidence)
            mask = _valid_mask(lengths_device, spikes.shape[1]).to(spikes.dtype).unsqueeze(-1)
            scores = (spikes * mask).sum(dim=1)
            ys.append(y.numpy())
            preds.append(scores.argmax(dim=1).cpu().numpy())
    return exp10._classification_metrics(np.concatenate(ys), np.concatenate(preds))


def _evaluate_joint_branches(
    model: Exp101Net,
    loader: Iterable,
    device: torch.device,
) -> dict[str, dict[str, float]] | None:
    if model.spec.objective != OBJECTIVE_JOINT:
        return None
    y_parts: list[np.ndarray] = []
    pred_parts: dict[str, list[np.ndarray]] = {
        "full": [],
        "l1_only": [],
        "l2_only": [],
    }
    model.eval()
    with torch.no_grad():
        for X, y, lengths in loader:
            X = X.to(device=device, dtype=torch.float32)
            ld = lengths.to(device)
            trajectory = model.forward_trajectory(X)
            l1 = trajectory["l1_evidence"]
            if l1 is None:
                raise RuntimeError("Joint branch evaluation requires L1 evidence")
            l2 = trajectory["l2_evidence"]
            scores = {
                "full": _valid_mean(l1 + l2, ld),
                "l1_only": _valid_mean(l1, ld),
                "l2_only": _valid_mean(l2, ld),
            }
            y_parts.append(y.numpy())
            for name, value in scores.items():
                pred_parts[name].append(value.argmax(dim=1).cpu().numpy())
    y_true = np.concatenate(y_parts)
    return {
        name: exp10._classification_metrics(y_true, np.concatenate(parts))
        for name, parts in pred_parts.items()
    }


def _collect_probe_features_and_activity(
    model: Exp101Net,
    data: exp3.Data,
    spec: RunSpec,
    config: Config,
) -> tuple[
    dict[str, dict[str, np.ndarray]],
    dict[str, np.ndarray],
    dict[str, Any],
]:
    device = torch.device(config.device)
    loaders = exp73._raw_loaders(data, spec.seed, config.batch_size, False)
    names = exp10.probe_names()
    feature_parts: dict[str, dict[str, list[np.ndarray]]] = {
        split: {name: [] for name in names} for split in SPLITS
    }
    label_parts: dict[str, list[np.ndarray]] = {split: [] for split in SPLITS}
    test_comm: dict[str, list[torch.Tensor]] = {"l1": [], "l2": []}
    test_lengths: list[np.ndarray] = []

    model.eval()
    with torch.no_grad():
        for split in SPLITS:
            for X, y, lengths in loaders[split]:
                Xd = X.to(device=device, dtype=torch.float32)
                ld = lengths.to(device=device, dtype=torch.long)
                trajectory = model.forward_trajectory(Xd)
                hidden = trajectory["hidden"]

                feature_parts[split]["input__events__whole_count"].append(
                    exp10._masked_sum(Xd, ld).cpu().numpy().astype(np.float32, copy=False)
                )
                feature_parts[split]["input__events__fixed250_count"].append(
                    exp3.fixed_counts(Xd, ld, data.bin_steps)
                    .flatten(start_dim=1)
                    .cpu()
                    .numpy()
                    .astype(np.float32, copy=False)
                )

                for layer in HIDDEN_LAYERS:
                    for state in HIDDEN_STATES:
                        values = hidden[layer][state]
                        for aggregation in exp10.STATE_AGGREGATIONS:
                            key = f"{layer}__{state}__{aggregation}"
                            aggregated = exp735._aggregate_tensor(
                                values, ld, aggregation, data.bin_steps
                            )
                            feature_parts[split][key].append(
                                aggregated.cpu().numpy().astype(np.float32, copy=False)
                            )
                    spikes = hidden[layer]["spike"]
                    feature_parts[split][f"{layer}__spike__whole_count"].append(
                        exp10._masked_sum(spikes, ld)
                        .cpu()
                        .numpy()
                        .astype(np.float32, copy=False)
                    )
                    feature_parts[split][f"{layer}__spike__fixed250_count"].append(
                        exp3.fixed_counts(spikes, ld, data.bin_steps)
                        .flatten(start_dim=1)
                        .cpu()
                        .numpy()
                        .astype(np.float32, copy=False)
                    )
                    if split == "test":
                        test_comm[layer].append(spikes.cpu())

                if split == "test":
                    test_lengths.append(lengths.numpy().astype(np.int64, copy=False))
                label_parts[split].append(y.numpy().astype(np.int64, copy=False))

    features = {
        split: {
            name: np.concatenate(parts, axis=0)
            for name, parts in feature_parts[split].items()
        }
        for split in SPLITS
    }
    labels = {
        split: np.concatenate(label_parts[split], axis=0)
        for split in SPLITS
    }

    lengths_np = np.concatenate(test_lengths, axis=0)
    activity = {
        "l1": exp81._communication_stats(
            torch.cat(test_comm["l1"], dim=0),
            lengths_np,
            exp811._groups(spec.l1_coding),
            1,
        ),
        "l2": exp81._communication_stats(
            torch.cat(test_comm["l2"], dim=0),
            lengths_np,
            exp811._groups(spec.l2_coding),
            1,
        ),
    }
    return features, labels, activity


def _fit_probes(
    spec: RunSpec,
    features: dict[str, dict[str, np.ndarray]],
    labels: dict[str, np.ndarray],
    test_actions: np.ndarray,
) -> pd.DataFrame:
    frame = exp10._fit_all_probes(spec, features, labels, test_actions)
    frame.insert(1, "coding", spec.coding)
    frame.insert(2, "objective", spec.objective)
    return frame


def _run_artifacts(config: Config, spec: RunSpec) -> dict[str, Path]:
    return {
        "checkpoint": _path(config.results_dir, "checkpoints", spec.key, ".pt"),
        "history": _path(config.results_dir, "histories", spec.key, ".csv"),
        "evaluation": _path(config.results_dir, "evaluations", spec.key, ".json"),
        "probes": _path(config.results_dir, "probe_evaluations", spec.key, ".csv"),
    }


def run_one(
    spec: RunSpec,
    config: Config,
    force: bool = False,
) -> dict[str, Any]:
    validate_spec(spec)
    artifacts = _run_artifacts(config, spec)
    if not force and all(path.exists() for path in artifacts.values()):
        return json.loads(artifacts["evaluation"].read_text(encoding="utf-8"))

    data, frames, split_manifest = _prepare_run_data(config, spec.variant, spec.rotation)
    del split_manifest
    split_hashes = _split_hashes(frames)

    torch.set_num_threads(config.threads)
    device = torch.device(config.device)
    model_init_seed = exp73._e2e_pair_seed(spec.seed, "model_init")
    exp3.seed_all(model_init_seed)
    model = Exp101Net(spec, len(data.labels), data.fs).to(device)

    resume_checkpoint = (
        not force and artifacts["checkpoint"].exists() and artifacts["history"].exists()
    )
    if resume_checkpoint:
        checkpoint_payload = torch.load(
            artifacts["checkpoint"], map_location="cpu", weights_only=False
        )
        if checkpoint_payload.get("experiment_id") != EXPERIMENT_ID:
            raise RuntimeError(f"{spec.key}: checkpoint experiment mismatch")
        if checkpoint_payload.get("protocol_version") != PROTOCOL_VERSION:
            raise RuntimeError(f"{spec.key}: checkpoint protocol mismatch")
        if checkpoint_payload.get("spec") != asdict(spec):
            raise RuntimeError(f"{spec.key}: checkpoint spec mismatch")
        if checkpoint_payload.get("split_sample_hashes") != split_hashes:
            raise RuntimeError(f"{spec.key}: checkpoint split geometry mismatch")
        if int(checkpoint_payload.get("model_init_seed", -1)) != int(model_init_seed):
            raise RuntimeError(f"{spec.key}: checkpoint model-init seed mismatch")
        best_state = checkpoint_payload["model_state_dict"]
        best_epoch = int(checkpoint_payload["best_epoch"])
        stopped_epoch = int(checkpoint_payload["stopped_epoch"])
        best_ba = float(checkpoint_payload["best_val_ba"])
        best_loss = float(checkpoint_payload["best_val_objective_loss"])
    else:
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=exp72.LR,
            weight_decay=exp72.WEIGHT_DECAY,
        )
        train_loader = exp73._raw_loaders(
            data, spec.seed, config.batch_size, True
        )["train"]
        eval_loaders = exp73._raw_loaders(data, spec.seed, config.batch_size, False)

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
                loss, main_ce, aux = _loss_terms(model, trajectory, lengths, y)
                loss.backward()
                optimizer.step()
                total_sum += float(loss.detach()) * len(y)
                main_sum += float(main_ce.detach()) * len(y)
                if aux is not None:
                    aux_sum += float(aux.detach()) * len(y)
                n_total += len(y)

            train_metrics, _, _ = _evaluate_native(
                model, eval_loaders["train"], device
            )
            val_metrics, _, _ = _evaluate_native(
                model, eval_loaders["val"], device
            )
            history.append(
                {
                    "epoch": float(epoch),
                    "train_ba": float(train_metrics["balanced_accuracy"]),
                    "val_ba": float(val_metrics["balanced_accuracy"]),
                    "train_loss": total_sum / max(n_total, 1),
                    "val_loss": float(val_metrics["objective_loss"]),
                    "train_main_ce": main_sum / max(n_total, 1),
                    "val_main_ce": float(val_metrics["main_ce"]),
                    "train_l1_tsce": (
                        aux_sum / max(n_total, 1)
                        if spec.objective == OBJECTIVE_L1_TSCE
                        else np.nan
                    ),
                    "val_l1_tsce": (
                        float(val_metrics["l1_tsce"])
                        if spec.objective == OBJECTIVE_L1_TSCE
                        else np.nan
                    ),
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
            if (
                epoch >= MIN_EPOCHS
                and best_epoch > 0
                and epoch - best_epoch >= PATIENCE
            ):
                stopped_epoch = epoch
                break

        if best_state is None:
            raise RuntimeError(f"No checkpoint selected for {spec.key}")

        artifacts["checkpoint"].parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "experiment_id": EXPERIMENT_ID,
                "protocol_version": PROTOCOL_VERSION,
                "spec": asdict(spec),
                "model_init_seed": int(model_init_seed),
                "split_sample_hashes": split_hashes,
                "best_epoch": best_epoch,
                "stopped_epoch": stopped_epoch,
                "best_val_ba": best_ba,
                "best_val_objective_loss": best_loss,
                "tsce_lambda": TSCE_LAMBDA,
                "threshold_multipliers": list(THRESHOLD_MULTIPLIERS),
                "model_state_dict": best_state,
            },
            artifacts["checkpoint"],
        )
        artifacts["history"].parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(history).to_csv(artifacts["history"], index=False)

    model.load_state_dict(best_state, strict=True)
    eval_loaders = exp73._raw_loaders(data, spec.seed, config.batch_size, False)

    native_metrics: dict[str, dict[str, float]] = {}
    native_arrays: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    lif_metrics: dict[str, dict[str, float]] = {}
    for split in SPLITS:
        metrics, y_true, y_pred = _evaluate_native(
            model, eval_loaders[split], device
        )
        native_metrics[split] = metrics
        native_arrays[split] = (y_true, y_pred)
        lif_metrics[split] = _evaluate_lif_transfer(
            model, eval_loaders[split], device
        )

    branch_test = _evaluate_joint_branches(model, eval_loaders["test"], device)
    test_actions = frames["test"].action.to_numpy(dtype=np.int64, copy=True)
    native_test_by_action = exp10._subgroup_test_metrics(
        native_arrays["test"][0],
        native_arrays["test"][1],
        test_actions,
    )

    features, probe_labels, activity = _collect_probe_features_and_activity(
        model, data, spec, config
    )
    for split in SPLITS:
        if not np.array_equal(probe_labels[split], native_arrays[split][0]):
            raise RuntimeError(f"{spec.key}/{split}: probe label order mismatch")
    probe_frame = _fit_probes(spec, features, probe_labels, test_actions)
    artifacts["probes"].parent.mkdir(parents=True, exist_ok=True)
    probe_frame.to_csv(artifacts["probes"], index=False)

    payload = {
        "experiment_id": EXPERIMENT_ID,
        "protocol_version": PROTOCOL_VERSION,
        "spec": asdict(spec),
        "contract": {
            "architecture": "30->128->128->12",
            "shifts": [list(SHIFTS), list(SHIFTS)],
            "coding": spec.coding,
            "threshold_multipliers": (
                list(THRESHOLD_MULTIPLIERS) if spec.coding == CODING_MM else [1.0]
            ),
            "objective": spec.objective,
            "tsce_lambda": TSCE_LAMBDA,
            "readout": "time_shared",
            "phase_aware": False,
            "bias": False,
            "max_epochs": config.max_epochs,
            "min_epochs": MIN_EPOCHS,
            "patience": PATIENCE,
            "checkpoint_metric": "native validation balanced accuracy",
            "checkpoint_tiebreak": "native validation objective loss",
        },
        "dataset_roots": [
            str(path.resolve()) for path in _variant_roots(config.repo_root, spec.variant)
        ],
        "labels": list(data.labels),
        "labels_hash": _sha_rows(data.labels),
        "fs_hz": float(data.fs),
        "timesteps": int(data.T),
        "bin_steps": int(data.bin_steps),
        "n_bins": int(data.n_bins),
        "split": {key: list(value) for key, value in data.split.items()},
        "split_sample_hashes": split_hashes,
        "model_init_seed": int(model_init_seed),
        "best_epoch": best_epoch,
        "stopped_epoch": stopped_epoch,
        "best_val_ba": best_ba,
        "best_val_objective_loss": best_loss,
        "native_metrics": native_metrics,
        "native_test_by_action": native_test_by_action,
        "lif_transfer_metrics": lif_metrics,
        "joint_branch_test_metrics": branch_test,
        "activity": activity,
        "probe_count": int(len(probe_frame)),
        "probe_names": list(exp10.probe_names()),
        "parameter_count": int(sum(p.numel() for p in model.parameters())),
        "checkpoint": str(artifacts["checkpoint"].relative_to(config.repo_root)),
        "history": str(artifacts["history"].relative_to(config.repo_root)),
        "probe_evaluation_csv": str(artifacts["probes"].relative_to(config.repo_root)),
    }
    _save_json(artifacts["evaluation"], payload)
    return payload


def _run_row(payload: Mapping[str, Any]) -> dict[str, Any]:
    spec = payload["spec"]
    native = payload["native_metrics"]
    lif = payload["lif_transfer_metrics"]
    branch = payload.get("joint_branch_test_metrics")
    row: dict[str, Any] = {
        "variant": spec["variant"],
        "coding": spec["coding"],
        "objective": spec["objective"],
        "rotation": int(spec["rotation"]),
        "seed": int(spec["seed"]),
        "best_epoch": int(payload["best_epoch"]),
        "stopped_epoch": int(payload["stopped_epoch"]),
        "parameter_count": int(payload["parameter_count"]),
        "model_init_seed": int(payload["model_init_seed"]),
        "labels_hash": payload["labels_hash"],
        "train_sample_hash": payload["split_sample_hashes"]["train"],
        "val_sample_hash": payload["split_sample_hashes"]["val"],
        "test_sample_hash": payload["split_sample_hashes"]["test"],
    }
    for split in SPLITS:
        for metric in ("accuracy", "balanced_accuracy", "macro_f1", "objective_loss"):
            if metric in native[split]:
                row[f"native_{split}_{metric}"] = float(native[split][metric])
        for metric in ("accuracy", "balanced_accuracy", "macro_f1"):
            row[f"lif_{split}_{metric}"] = float(lif[split][metric])
    for action in (0, 1):
        values = payload["native_test_by_action"][f"action{action}"]
        for metric in ("accuracy", "balanced_accuracy", "macro_f1"):
            row[f"native_test_action{action}_{metric}"] = (
                float(values[metric]) if values is not None else np.nan
            )
    if branch is not None:
        full = float(branch["full"]["balanced_accuracy"])
        l1 = float(branch["l1_only"]["balanced_accuracy"])
        l2 = float(branch["l2_only"]["balanced_accuracy"])
        row["joint_full_test_ba"] = full
        row["joint_l1_only_test_ba"] = l1
        row["joint_l2_only_test_ba"] = l2
        row["joint_remove_l1_delta"] = full - l2
        row["joint_remove_l2_delta"] = full - l1
    else:
        for name in (
            "joint_full_test_ba",
            "joint_l1_only_test_ba",
            "joint_l2_only_test_ba",
            "joint_remove_l1_delta",
            "joint_remove_l2_delta",
        ):
            row[name] = np.nan
    return row


def _activity_rows(payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    spec = payload["spec"]
    rows: list[dict[str, Any]] = []
    for layer, groups in payload["activity"].items():
        for group in groups:
            rows.append(
                {
                    "variant": spec["variant"],
                    "coding": spec["coding"],
                    "objective": spec["objective"],
                    "rotation": int(spec["rotation"]),
                    "seed": int(spec["seed"]),
                    "layer": layer,
                    **group,
                }
            )
    return rows


def _validate_pairing(runs: pd.DataFrame) -> None:
    for rotation in ROTATIONS:
        for seed in MODEL_SEEDS:
            cell = runs[(runs.rotation == rotation) & (runs.seed == seed)]
            if len(cell) != len(VARIANTS) * len(CODINGS) * len(OBJECTIVES):
                raise RuntimeError(f"Incomplete factor cell rotation={rotation}, seed={seed}")
            for field in (
                "model_init_seed",
                "labels_hash",
                "train_sample_hash",
                "val_sample_hash",
                "test_sample_hash",
            ):
                if cell[field].nunique(dropna=False) != 1:
                    raise RuntimeError(
                        f"Pairing mismatch {field} at rotation={rotation}, seed={seed}"
                    )


def _native_contrast_rows(runs: pd.DataFrame) -> pd.DataFrame:
    metric_columns = (
        "native_test_balanced_accuracy",
        "native_test_accuracy",
        "native_test_macro_f1",
    )
    rows: list[dict[str, Any]] = []
    for rotation in ROTATIONS:
        for seed in MODEL_SEEDS:
            cell = runs[(runs.rotation == rotation) & (runs.seed == seed)]
            indexed = cell.set_index(["variant", "coding", "objective"])

            def add(
                contrast: str,
                left: tuple[str, str, str],
                right: tuple[str, str, str],
                **meta: Any,
            ) -> None:
                row: dict[str, Any] = {
                    "contrast": contrast,
                    "rotation": rotation,
                    "seed": seed,
                    **meta,
                }
                for metric in metric_columns:
                    row[f"{metric}_delta"] = float(
                        indexed.loc[left, metric] - indexed.loc[right, metric]
                    )
                rows.append(row)

            for coding in CODINGS:
                for objective in OBJECTIVES:
                    add(
                        "D1_minus_D0",
                        (VARIANT_POSTENCODE, coding, objective),
                        (VARIANT_ORIGINAL, coding, objective),
                        coding=coding,
                        objective=objective,
                    )
            for variant in VARIANTS:
                for objective in OBJECTIVES:
                    add(
                        "MM_minus_BB",
                        (variant, CODING_MM, objective),
                        (variant, CODING_BB, objective),
                        variant=variant,
                        objective=objective,
                    )
                for coding in CODINGS:
                    add(
                        "Joint_minus_Baseline",
                        (variant, coding, OBJECTIVE_JOINT),
                        (variant, coding, OBJECTIVE_BASELINE),
                        variant=variant,
                        coding=coding,
                    )
                    add(
                        "L1TSCE_minus_Baseline",
                        (variant, coding, OBJECTIVE_L1_TSCE),
                        (variant, coding, OBJECTIVE_BASELINE),
                        variant=variant,
                        coding=coding,
                    )

            for variant in VARIANTS:
                for objective, name in (
                    (OBJECTIVE_JOINT, "MT_x_Joint"),
                    (OBJECTIVE_L1_TSCE, "MT_x_L1TSCE"),
                ):
                    row = {
                        "contrast": name,
                        "variant": variant,
                        "rotation": rotation,
                        "seed": seed,
                    }
                    for metric in metric_columns:
                        treatment = (
                            float(indexed.loc[(variant, CODING_MM, objective), metric])
                            - float(indexed.loc[(variant, CODING_BB, objective), metric])
                        )
                        baseline = (
                            float(indexed.loc[(variant, CODING_MM, OBJECTIVE_BASELINE), metric])
                            - float(indexed.loc[(variant, CODING_BB, OBJECTIVE_BASELINE), metric])
                        )
                        row[f"{metric}_delta"] = treatment - baseline
                    rows.append(row)
    return pd.DataFrame(rows)



def _information_contrast_rows(info: pd.DataFrame) -> pd.DataFrame:
    metric_columns = (
        "l1_pre_reset_fixed250_ba",
        "l1_spike_fixed250_ba",
        "l2_pre_reset_fixed250_ba",
        "l2_spike_fixed250_ba",
        "l1_quantization_delta",
        "l1_to_l2_transform_delta",
        "l2_quantization_delta",
    )
    rows: list[dict[str, Any]] = []
    for rotation in ROTATIONS:
        for seed in MODEL_SEEDS:
            cell = info[(info.rotation == rotation) & (info.seed == seed)]
            indexed = cell.set_index(["variant", "coding", "objective"])

            def add(
                contrast: str,
                left: tuple[str, str, str],
                right: tuple[str, str, str],
                **meta: Any,
            ) -> None:
                row: dict[str, Any] = {
                    "contrast": contrast,
                    "rotation": rotation,
                    "seed": seed,
                    **meta,
                }
                for metric in metric_columns:
                    row[f"{metric}_delta"] = float(
                        indexed.loc[left, metric] - indexed.loc[right, metric]
                    )
                rows.append(row)

            for coding in CODINGS:
                for objective in OBJECTIVES:
                    add(
                        "D1_minus_D0",
                        (VARIANT_POSTENCODE, coding, objective),
                        (VARIANT_ORIGINAL, coding, objective),
                        coding=coding,
                        objective=objective,
                    )
            for variant in VARIANTS:
                for objective in OBJECTIVES:
                    add(
                        "MM_minus_BB",
                        (variant, CODING_MM, objective),
                        (variant, CODING_BB, objective),
                        variant=variant,
                        objective=objective,
                    )
                for coding in CODINGS:
                    add(
                        "Joint_minus_Baseline",
                        (variant, coding, OBJECTIVE_JOINT),
                        (variant, coding, OBJECTIVE_BASELINE),
                        variant=variant,
                        coding=coding,
                    )
                    add(
                        "L1TSCE_minus_Baseline",
                        (variant, coding, OBJECTIVE_L1_TSCE),
                        (variant, coding, OBJECTIVE_BASELINE),
                        variant=variant,
                        coding=coding,
                    )
            for variant in VARIANTS:
                for objective, name in (
                    (OBJECTIVE_JOINT, "MT_x_Joint"),
                    (OBJECTIVE_L1_TSCE, "MT_x_L1TSCE"),
                ):
                    row = {
                        "contrast": name,
                        "variant": variant,
                        "rotation": rotation,
                        "seed": seed,
                    }
                    for metric in metric_columns:
                        treatment = (
                            float(indexed.loc[(variant, CODING_MM, objective), metric])
                            - float(indexed.loc[(variant, CODING_BB, objective), metric])
                        )
                        baseline = (
                            float(indexed.loc[(variant, CODING_MM, OBJECTIVE_BASELINE), metric])
                            - float(indexed.loc[(variant, CODING_BB, OBJECTIVE_BASELINE), metric])
                        )
                        row[f"{metric}_delta"] = treatment - baseline
                    rows.append(row)
    return pd.DataFrame(rows)


def _information_path_runs(probes: pd.DataFrame) -> pd.DataFrame:
    target = {
        "l1_pre_reset_fixed250_ba": "l1__pre_reset__fixed250_ordered_mean",
        "l1_spike_fixed250_ba": "l1__spike__fixed250_count",
        "l2_pre_reset_fixed250_ba": "l2__pre_reset__fixed250_ordered_mean",
        "l2_spike_fixed250_ba": "l2__spike__fixed250_count",
    }
    keys = ["variant", "coding", "objective", "rotation", "seed"]
    rows: list[dict[str, Any]] = []
    for key_values, cell in probes.groupby(keys, sort=False):
        row = dict(zip(keys, key_values, strict=True))
        by_probe = cell.set_index("probe")
        for name, probe in target.items():
            row[name] = float(by_probe.loc[probe, "test_balanced_accuracy"])
        row["l1_quantization_delta"] = (
            row["l1_spike_fixed250_ba"] - row["l1_pre_reset_fixed250_ba"]
        )
        row["l1_to_l2_transform_delta"] = (
            row["l2_pre_reset_fixed250_ba"] - row["l1_spike_fixed250_ba"]
        )
        row["l2_quantization_delta"] = (
            row["l2_spike_fixed250_ba"] - row["l2_pre_reset_fixed250_ba"]
        )
        rows.append(row)
    return pd.DataFrame(rows)


def _flatten_columns(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out.columns = [
        "_".join(str(part) for part in column if str(part))
        if isinstance(column, tuple)
        else str(column)
        for column in out.columns
    ]
    return out


def finalize(config: Config) -> dict[str, Any]:
    payloads: list[dict[str, Any]] = []
    probe_frames: list[pd.DataFrame] = []
    missing: list[str] = []

    for spec in run_specs():
        artifacts = _run_artifacts(config, spec)
        for name, path in artifacts.items():
            if not path.exists():
                missing.append(f"{spec.key}:{name}:{path}")
        if artifacts["evaluation"].exists():
            payloads.append(
                json.loads(artifacts["evaluation"].read_text(encoding="utf-8"))
            )
        if artifacts["probes"].exists():
            frame = pd.read_csv(artifacts["probes"])
            if len(frame) != len(exp10.probe_names()):
                raise RuntimeError(
                    f"{spec.key}: expected {len(exp10.probe_names())} probes, got {len(frame)}"
                )
            probe_frames.append(frame)

    if missing:
        raise FileNotFoundError(
            f"Exp10.1 incomplete; missing {len(missing)} artifacts:\n"
            + "\n".join(missing[:30])
        )
    if len(payloads) != EXPECTED_RUNS:
        raise RuntimeError(f"Expected {EXPECTED_RUNS} evaluations, got {len(payloads)}")

    runs = pd.DataFrame([_run_row(payload) for payload in payloads]).sort_values(
        ["variant", "coding", "objective", "rotation", "seed"]
    )
    _validate_pairing(runs)
    runs.to_csv(config.results_dir / "run_metrics.csv", index=False)

    primary_metrics = [
        "native_test_balanced_accuracy",
        "native_test_accuracy",
        "native_test_macro_f1",
        "native_train_balanced_accuracy",
        "native_val_balanced_accuracy",
        "lif_test_balanced_accuracy",
    ]
    run_summary = (
        runs.groupby(["variant", "coding", "objective"], sort=False)[primary_metrics]
        .agg(["count", "mean", "std"])
        .reset_index()
    )
    _flatten_columns(run_summary).to_csv(
        config.results_dir / "condition_run_summary.csv", index=False
    )

    split_level = (
        runs.groupby(["variant", "coding", "objective", "rotation"], sort=False)[
            primary_metrics
        ]
        .mean()
        .reset_index()
    )
    split_level.to_csv(config.results_dir / "condition_split_level.csv", index=False)
    # With one locked user split, the three model seeds are optimization
    # replicates. Summaries therefore aggregate paired seeds directly rather
    # than treating the single rotation as a population-level replicate.
    _flatten_columns(run_summary).to_csv(
        config.results_dir / "condition_summary.csv", index=False
    )

    contrasts = _native_contrast_rows(runs)
    contrasts.to_csv(config.results_dir / "contrast_runs.csv", index=False)
    contrast_metric_cols = [
        column for column in contrasts.columns if column.endswith("_delta")
    ]
    contrast_split = (
        contrasts.groupby(
            [
                column
                for column in ("contrast", "variant", "coding", "objective", "rotation")
                if column in contrasts.columns
            ],
            dropna=False,
            sort=False,
        )[contrast_metric_cols]
        .mean()
        .reset_index()
    )
    contrast_split.to_csv(config.results_dir / "contrast_split_level.csv", index=False)
    contrast_group_cols = [
        column
        for column in ("contrast", "variant", "coding", "objective")
        if column in contrasts.columns
    ]
    contrast_summary = (
        contrasts.groupby(
            contrast_group_cols,
            dropna=False,
            sort=False,
        )[contrast_metric_cols]
        .agg(["count", "mean", "std"])
        .reset_index()
    )
    _flatten_columns(contrast_summary).to_csv(
        config.results_dir / "contrast_summary.csv", index=False
    )

    probes = pd.concat(probe_frames, ignore_index=True).sort_values(
        ["probe", "variant", "coding", "objective", "rotation", "seed"]
    )
    expected_probe_rows = EXPECTED_RUNS * len(exp10.probe_names())
    if len(probes) != expected_probe_rows:
        raise RuntimeError(
            f"Expected {expected_probe_rows} probe rows, got {len(probes)}"
        )
    probes.to_csv(config.results_dir / "probe_runs.csv", index=False)
    probe_split = (
        probes.groupby(
            ["variant", "coding", "objective", "rotation", "probe"], sort=False
        )["test_balanced_accuracy"]
        .mean()
        .reset_index()
    )
    probe_split.to_csv(config.results_dir / "probe_split_level.csv", index=False)
    probe_summary = (
        probes.groupby(
            ["variant", "coding", "objective", "probe"], sort=False
        )["test_balanced_accuracy"]
        .agg(["count", "mean", "std"])
        .reset_index()
    )
    probe_summary.to_csv(config.results_dir / "probe_summary.csv", index=False)

    info = _information_path_runs(probes)
    info.to_csv(config.results_dir / "information_path_runs.csv", index=False)
    info_metrics = [
        "l1_pre_reset_fixed250_ba",
        "l1_spike_fixed250_ba",
        "l2_pre_reset_fixed250_ba",
        "l2_spike_fixed250_ba",
        "l1_quantization_delta",
        "l1_to_l2_transform_delta",
        "l2_quantization_delta",
    ]
    info_split = (
        info.groupby(["variant", "coding", "objective", "rotation"], sort=False)[
            info_metrics
        ]
        .mean()
        .reset_index()
    )
    info_split.to_csv(config.results_dir / "information_path_split_level.csv", index=False)
    info_summary = (
        info.groupby(["variant", "coding", "objective"], sort=False)[info_metrics]
        .agg(["count", "mean", "std"])
        .reset_index()
    )
    _flatten_columns(info_summary).to_csv(
        config.results_dir / "information_path_summary.csv", index=False
    )

    info_contrasts = _information_contrast_rows(info)
    info_contrasts.to_csv(
        config.results_dir / "information_path_contrast_runs.csv", index=False
    )
    info_contrast_metrics = [
        column for column in info_contrasts.columns if column.endswith("_delta")
    ]
    info_contrast_split = (
        info_contrasts.groupby(
            [
                column
                for column in ("contrast", "variant", "coding", "objective", "rotation")
                if column in info_contrasts.columns
            ],
            dropna=False,
            sort=False,
        )[info_contrast_metrics]
        .mean()
        .reset_index()
    )
    info_contrast_split.to_csv(
        config.results_dir / "information_path_contrast_split_level.csv",
        index=False,
    )
    info_contrast_group_cols = [
        column
        for column in ("contrast", "variant", "coding", "objective")
        if column in info_contrasts.columns
    ]
    info_contrast_summary = (
        info_contrasts.groupby(
            info_contrast_group_cols,
            dropna=False,
            sort=False,
        )[info_contrast_metrics]
        .agg(["count", "mean", "std"])
        .reset_index()
    )
    _flatten_columns(info_contrast_summary).to_csv(
        config.results_dir / "information_path_contrast_summary.csv",
        index=False,
    )

    activity = pd.DataFrame(
        [row for payload in payloads for row in _activity_rows(payload)]
    )
    activity.to_csv(config.results_dir / "activity_runs.csv", index=False)
    activity_summary = (
        activity.groupby(
            [
                "variant",
                "coding",
                "objective",
                "layer",
                "threshold_multiplier",
                "shift",
            ],
            sort=False,
        )[
            [
                "mean_value_per_neuron_step",
                "fraction_nonzero",
                "fraction_gt1",
                "fraction_at_cap",
            ]
        ]
        .agg(["mean", "std"])
        .reset_index()
    )
    _flatten_columns(activity_summary).to_csv(
        config.results_dir / "activity_summary.csv", index=False
    )

    manifest = {
        "experiment_id": EXPERIMENT_ID,
        "protocol_version": PROTOCOL_VERSION,
        "status": "PASS",
        "variants": list(VARIANTS),
        "codings": list(CODINGS),
        "objectives": list(OBJECTIVES),
        "tsce_lambda": TSCE_LAMBDA,
        "rotations": list(ROTATIONS),
        "model_seeds": list(MODEL_SEEDS),
        "run_count": int(len(runs)),
        "expected_run_count": EXPECTED_RUNS,
        "probe_count_per_run": len(exp10.probe_names()),
        "probe_run_count": int(len(probes)),
        "readout": "time-shared only",
        "phase_aware": False,
        "primary_metric": "native_test_balanced_accuracy",
        "statistical_scope": (
            "one locked cross-user split (rotation0); seeds 11/23/37 are paired "
            "optimization replicates, not independent user splits"
        ),
        "primary_contrasts": [
            "D1_minus_D0",
            "MM_minus_BB",
            "Joint_minus_Baseline",
            "L1TSCE_minus_Baseline",
            "MT_x_Joint",
            "MT_x_L1TSCE",
        ],
        "primary_outputs": [
            "run_metrics.csv",
            "condition_split_level.csv",
            "condition_summary.csv",
            "contrast_runs.csv",
            "contrast_split_level.csv",
            "contrast_summary.csv",
            "probe_runs.csv",
            "probe_split_level.csv",
            "probe_summary.csv",
            "information_path_runs.csv",
            "information_path_split_level.csv",
            "information_path_summary.csv",
            "information_path_contrast_runs.csv",
            "information_path_contrast_split_level.csv",
            "information_path_contrast_summary.csv",
            "activity_runs.csv",
            "activity_summary.csv",
        ],
    }
    _save_json(config.results_dir / "manifest.json", manifest)
    return manifest


def _resolve_config(args: argparse.Namespace) -> Config:
    repo_root = Path(args.repo_root).resolve() if args.repo_root else find_repo_root()
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
            "Exp10.1 D0/D1 x BB/MM x baseline/joint/L1-TSCE "
            "A2 backbone ablation with time-shared readout"
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
        print(json.dumps(prepare_all(config), indent=2, sort_keys=True))
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
                    "native_test_ba": payload["native_metrics"]["test"][
                        "balanced_accuracy"
                    ],
                },
                indent=2,
            )
        )
        return
    if args.command == "finalize":
        print(json.dumps(finalize(config), indent=2, sort_keys=True))
        return
    raise ValueError(args.command)


if __name__ == "__main__":
    main()
