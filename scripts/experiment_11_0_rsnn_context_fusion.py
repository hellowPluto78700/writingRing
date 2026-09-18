from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import torch
from torch import nn
import torch.nn.functional as F

from scripts import experiment_3_0_1_single_tau_objectives as exp3
from scripts import experiment_4_0_1_multispike_macro_lif as exp401
from scripts import experiment_7_2_two_layer_tau_training as exp72
from scripts import experiment_7_3_training_strategy_decomposition as exp73
from scripts import experiment_8_1_1_two_layer_mt_factorial as exp811
from scripts import experiment_10_0_airborne_motion_ablation as exp10
from scripts import experiment_10_1_a2_backbone_coding_supervision as exp101
from scripts import experiment_10_2_l1_membrane_memory as exp102
from scripts import experiment_10_2_1_l2_membrane_weighted as exp1021
from scripts import experiment_10_2_2_valid_window_probes as exp1022


EXPERIMENT_ID = "experiment_11_0_rsnn_context_fusion"
PROTOCOL_VERSION = "d1_l1init2_recurrence3_rsnn128_fusion128_v1"

VARIANT = exp101.VARIANT_POSTENCODE
ROTATION = 0
MODEL_SEEDS = (11, 23, 37)

L1_INIT_DYNAMICS = "dynamics_only"
L1_INIT_PRETRAINED = "pretrained_input"
L1_INITS = (L1_INIT_DYNAMICS, L1_INIT_PRETRAINED)

RECURRENCE_NONE = "none"
RECURRENCE_DIAGONAL = "diagonal"
RECURRENCE_DENSE = "dense"
RECURRENCES = (RECURRENCE_NONE, RECURRENCE_DIAGONAL, RECURRENCE_DENSE)

WIDTH = 128
N_INPUTS = exp72.EXPECTED_CHANNELS
L1_MEM_SHIFT = 2
SOURCE_L2_MEM_SHIFT = 1
L1_SYN_SHIFTS = exp101.SHIFTS

RSNN_TAU_SYN_MS = exp102.tau_mem_ms_from_shift(2)
RSNN_TAU_MEM_MS = exp102.tau_mem_ms_from_shift(2)
FUSION_TAU_SYN_MS = exp102.tau_mem_ms_from_shift(2)
FUSION_TAU_MEM_MS = exp102.tau_mem_ms_from_shift(2)
RSNN_ALPHA = 0.75
RSNN_BETA = 0.75
FUSION_ALPHA = 0.75
FUSION_BETA = 0.75

RECURRENT_INIT_BOUND = 0.02
GRAD_CLIP_NORM = 1.0

MAX_EPOCHS = exp101.MAX_EPOCHS
MIN_EPOCHS = exp101.MIN_EPOCHS
PATIENCE = exp101.PATIENCE
BATCH_SIZE = exp101.BATCH_SIZE
SPLITS = exp101.SPLITS

LAYERS = ("l1", "rsnn", "fusion")
ANALOG_STATES = exp1022.ANALOG_STATES
COMMUNICATION_STATE = exp1022.COMMUNICATION_STATE
SUPPORTS = exp1022.SUPPORTS
ANALOG_AGGREGATIONS = exp1022.ANALOG_AGGREGATIONS
COMMUNICATION_AGGREGATIONS = exp1022.COMMUNICATION_AGGREGATIONS
C_GRID = exp1022.C_GRID
PROBE_MAX_ITER = exp1022.PROBE_MAX_ITER

EXPECTED_RUNS = len(L1_INITS) * len(RECURRENCES) * len(MODEL_SEEDS)
EXPECTED_PROBES_PER_RUN = (
    len(LAYERS)
    * (
        len(ANALOG_STATES) * len(SUPPORTS) * len(ANALOG_AGGREGATIONS)
        + len(SUPPORTS) * len(COMMUNICATION_AGGREGATIONS)
    )
)


@dataclass(frozen=True)
class RunSpec:
    l1_init: str
    recurrence: str
    seed: int

    @property
    def key(self) -> str:
        return f"{self.l1_init}__{self.recurrence}__seed{self.seed}"


@dataclass(frozen=True)
class Config:
    repo_root: Path
    results_dir: Path
    device: str = "cpu"
    threads: int = 1
    batch_size: int = BATCH_SIZE
    max_epochs: int = MAX_EPOCHS


def find_repo_root(start: Path | None = None) -> Path:
    return exp1021.find_repo_root(start)


def results_dir(repo_root: Path) -> Path:
    return repo_root / "notebooks" / "artifacts" / EXPERIMENT_ID / PROTOCOL_VERSION


def source_exp1021_results_dir(repo_root: Path) -> Path:
    return exp1021.results_dir(repo_root)


def source_exp1022_results_dir(repo_root: Path) -> Path:
    return exp1022.results_dir(repo_root)


def source_config(config: Config) -> exp1021.Config:
    return exp1021.Config(
        repo_root=config.repo_root,
        results_dir=source_exp1021_results_dir(config.repo_root),
        device=config.device,
        threads=config.threads,
        batch_size=config.batch_size,
        max_epochs=exp1021.MAX_EPOCHS,
    )


def run_specs() -> list[RunSpec]:
    return [
        RunSpec(l1_init=l1_init, recurrence=recurrence, seed=seed)
        for l1_init in L1_INITS
        for recurrence in RECURRENCES
        for seed in MODEL_SEEDS
    ]


def validate_spec(spec: RunSpec) -> None:
    if spec.l1_init not in L1_INITS:
        raise ValueError(spec.l1_init)
    if spec.recurrence not in RECURRENCES:
        raise ValueError(spec.recurrence)
    if spec.seed not in MODEL_SEEDS:
        raise ValueError(spec.seed)


def _save_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _path(root: Path, kind: str, key: str, suffix: str) -> Path:
    return root / kind / f"{key}{suffix}"


def _source_spec(seed: int) -> exp1021.RunSpec:
    return exp1021.RunSpec(
        coding=exp1021.CODING_BINARY,
        l2_mem_shift=SOURCE_L2_MEM_SHIFT,
        seed=seed,
    )


def _source_checkpoint_path(config: Config, seed: int) -> Path:
    return exp1021._run_artifacts(source_config(config), _source_spec(seed))["checkpoint"]


def _require_source_manifests(config: Config) -> tuple[dict[str, Any], dict[str, Any]]:
    p1021 = source_exp1021_results_dir(config.repo_root) / "manifest.json"
    p1022 = source_exp1022_results_dir(config.repo_root) / "manifest.json"
    if not p1021.exists():
        raise FileNotFoundError(f"Missing Exp10.2.1 manifest: {p1021}")
    if not p1022.exists():
        raise FileNotFoundError(f"Missing Exp10.2.2 manifest: {p1022}")

    m1021 = json.loads(p1021.read_text(encoding="utf-8"))
    m1022 = json.loads(p1022.read_text(encoding="utf-8"))
    expected1021 = {
        "experiment_id": exp1021.EXPERIMENT_ID,
        "protocol_version": exp1021.PROTOCOL_VERSION,
        "status": "PASS",
        "run_count": exp1021.EXPECTED_RUNS,
    }
    expected1022 = {
        "experiment_id": exp1022.EXPERIMENT_ID,
        "protocol_version": exp1022.PROTOCOL_VERSION,
        "status": "PASS",
    }
    for key, value in expected1021.items():
        if m1021.get(key) != value:
            raise RuntimeError(
                f"Exp10.2.1 manifest mismatch for {key}: "
                f"expected {value!r}, got {m1021.get(key)!r}"
            )
    for key, value in expected1022.items():
        if m1022.get(key) != value:
            raise RuntimeError(
                f"Exp10.2.2 manifest mismatch for {key}: "
                f"expected {value!r}, got {m1022.get(key)!r}"
            )
    return m1021, m1022


def _prepare_data(
    config: Config,
) -> tuple[exp3.Data, dict[str, pd.DataFrame], pd.DataFrame]:
    return exp1021._prepare_data(source_config(config))


def _split_hashes(frames: Mapping[str, pd.DataFrame]) -> dict[str, str]:
    return exp1021._split_hashes(frames)


def _load_source_checkpoint(
    config: Config,
    seed: int,
    split_hashes: Mapping[str, str],
) -> tuple[dict[str, Any], Path]:
    path = _source_checkpoint_path(config, seed)
    if not path.exists():
        raise FileNotFoundError(path)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    spec = _source_spec(seed)
    if payload.get("experiment_id") != exp1021.EXPERIMENT_ID:
        raise RuntimeError(f"{path}: source experiment mismatch")
    if payload.get("protocol_version") != exp1021.PROTOCOL_VERSION:
        raise RuntimeError(f"{path}: source protocol mismatch")
    if payload.get("spec") != asdict(spec):
        raise RuntimeError(f"{path}: source spec mismatch")
    if payload.get("split_sample_hashes") != dict(split_hashes):
        raise RuntimeError(f"{path}: source split geometry mismatch")
    state = payload.get("model_state_dict")
    if not isinstance(state, dict) or "hidden_linears.0.weight" not in state:
        raise RuntimeError(f"{path}: missing source L1 input weight")
    return payload, path


def prepare_all(config: Config) -> dict[str, Any]:
    m1021, m1022 = _require_source_manifests(config)
    data, frames, split_manifest = _prepare_data(config)
    split_hashes = _split_hashes(frames)

    source_rows: list[dict[str, Any]] = []
    for seed in MODEL_SEEDS:
        checkpoint, path = _load_source_checkpoint(config, seed, split_hashes)
        source_rows.append(
            {
                "seed": seed,
                "source_checkpoint": str(path.relative_to(config.repo_root)),
                "source_best_epoch": int(checkpoint["best_epoch"]),
                "source_spec": checkpoint["spec"],
            }
        )

    config.results_dir.mkdir(parents=True, exist_ok=True)
    split_root = config.results_dir / "split"
    split_root.mkdir(parents=True, exist_ok=True)
    split_manifest.drop(columns=["pi", "si"], errors="ignore").to_csv(
        split_root / "rotation0.csv", index=False
    )
    pd.concat(
        [
            frame.assign(split=split_name)
            for split_name, frame in frames.items()
        ],
        ignore_index=True,
    ).drop(columns=["pi", "si"], errors="ignore").to_csv(
        split_root / "split_samples.csv", index=False
    )

    audit = {
        "experiment_id": EXPERIMENT_ID,
        "protocol_version": PROTOCOL_VERSION,
        "dataset": VARIANT,
        "rotation": ROTATION,
        "labels": list(data.labels),
        "fs_hz": float(data.fs),
        "timesteps": int(data.T),
        "fixed250_bin_steps": int(data.bin_steps),
        "split_sample_hashes": split_hashes,
        "source_exp10_2_1_manifest": m1021,
        "source_exp10_2_2_manifest": m1022,
        "source_checkpoints": source_rows,
        "l1_inits": list(L1_INITS),
        "recurrences": list(RECURRENCES),
        "model_seeds": list(MODEL_SEEDS),
        "expected_runs": EXPECTED_RUNS,
        "expected_probes_per_run": EXPECTED_PROBES_PER_RUN,
        "training": "end-to-end for all trainable layers after initialization",
        "pretrained_transfer": (
            "seed-matched Exp10.2.1 binary l1mem2/l2mem1 checkpoint; "
            "copy hidden_linears.0.weight only"
        ),
        "statistical_scope": (
            "one locked cross-user split; seeds 11/23/37 are paired "
            "optimization replicates, not independent user splits"
        ),
    }
    _save_json(config.results_dir / "audit.json", audit)
    return audit


def _recurrence_mask(mode: str) -> torch.Tensor:
    if mode == RECURRENCE_NONE:
        return torch.zeros(WIDTH, WIDTH)
    if mode == RECURRENCE_DIAGONAL:
        return torch.eye(WIDTH)
    if mode == RECURRENCE_DENSE:
        return torch.ones(WIDTH, WIDTH)
    raise ValueError(mode)


class Exp110Net(nn.Module):
    """L1 local SNN -> RSNN context -> local/context fusion SNN -> shared Linear."""

    def __init__(self, spec: RunSpec, n_classes: int, fs: float) -> None:
        super().__init__()
        validate_spec(spec)
        self.spec = spec
        self.n_classes = int(n_classes)
        self.fs = float(fs)

        self.l1_input = nn.Linear(N_INPUTS, WIDTH, bias=False)
        self.rsnn_input = nn.Linear(WIDTH, WIDTH, bias=False)
        self.recurrent_weight = nn.Parameter(torch.empty(WIDTH, WIDTH))
        self.fusion_local = nn.Linear(WIDTH, WIDTH, bias=False)
        self.fusion_context = nn.Linear(WIDTH, WIDTH, bias=False)
        self.output_linear = nn.Linear(WIDTH, n_classes, bias=False)

        nn.init.uniform_(
            self.recurrent_weight,
            -RECURRENT_INIT_BOUND,
            RECURRENT_INIT_BOUND,
        )
        self.register_buffer("recurrence_mask", _recurrence_mask(spec.recurrence))
        self.register_buffer("alpha_l1", exp811._alpha_vector("binary"))
        self.register_buffer("alpha_rsnn", torch.tensor(RSNN_ALPHA, dtype=torch.float32))
        self.register_buffer(
            "alpha_fusion", torch.tensor(FUSION_ALPHA, dtype=torch.float32)
        )

        self.l1_lif = exp401.MacroMultiSpikeLIF(
            beta=exp1021.beta_from_mem_shift(L1_MEM_SHIFT, fs),
            threshold=float(exp73.THRESHOLD),
            max_spikes_per_dt=1,
            surrogate_slope=exp72.SURROGATE_SLOPE,
        )
        self.rsnn_lif = exp401.MacroMultiSpikeLIF(
            beta=RSNN_BETA,
            threshold=float(exp73.THRESHOLD),
            max_spikes_per_dt=1,
            surrogate_slope=exp72.SURROGATE_SLOPE,
        )
        self.fusion_lif = exp401.MacroMultiSpikeLIF(
            beta=FUSION_BETA,
            threshold=float(exp73.THRESHOLD),
            max_spikes_per_dt=1,
            surrogate_slope=exp72.SURROGATE_SLOPE,
        )

    def effective_recurrent_weight(self) -> torch.Tensor:
        return self.recurrent_weight * self.recurrence_mask

    def effective_recurrent_parameter_count(self) -> int:
        return int(self.recurrence_mask.sum().item())

    def forward_trajectory(
        self,
        x: torch.Tensor,
        *,
        collect_states: bool = True,
    ) -> dict[str, Any]:
        batch, steps, channels = x.shape
        if channels != N_INPUTS:
            raise ValueError(f"Expected {N_INPUTS} channels, got {channels}")

        syn_l1 = torch.zeros(batch, WIDTH, device=x.device, dtype=x.dtype)
        mem_l1 = torch.zeros_like(syn_l1)
        syn_rsnn = torch.zeros_like(syn_l1)
        mem_rsnn = torch.zeros_like(syn_l1)
        syn_fusion = torch.zeros_like(syn_l1)
        mem_fusion = torch.zeros_like(syn_l1)
        previous_rsnn_spike = torch.zeros_like(syn_l1)

        hidden_lists: dict[str, dict[str, list[torch.Tensor]]] | None = None
        if collect_states:
            hidden_lists = {
                layer: {
                    "syn_current": [],
                    "pre_reset": [],
                    "spike": [],
                    "post_reset": [],
                }
                for layer in LAYERS
            }

        evidence: list[torch.Tensor] = []
        rsnn_external_inputs: list[torch.Tensor] = []
        rsnn_recurrent_inputs: list[torch.Tensor] = []
        effective_recurrent = self.effective_recurrent_weight()

        for timestep in range(steps):
            syn_l1 = self.alpha_l1 * syn_l1 + self.l1_input(x[:, timestep])
            spike_l1, post_l1, pre_l1 = self.l1_lif(syn_l1, mem_l1)
            mem_l1 = post_l1

            rsnn_external = self.rsnn_input(spike_l1)
            rsnn_recurrent = F.linear(previous_rsnn_spike, effective_recurrent)
            syn_rsnn = (
                self.alpha_rsnn * syn_rsnn
                + rsnn_external
                + rsnn_recurrent
            )
            spike_rsnn, post_rsnn, pre_rsnn = self.rsnn_lif(
                syn_rsnn, mem_rsnn
            )
            mem_rsnn = post_rsnn
            previous_rsnn_spike = spike_rsnn

            fusion_drive = (
                self.fusion_local(spike_l1)
                + self.fusion_context(spike_rsnn)
            )
            syn_fusion = self.alpha_fusion * syn_fusion + fusion_drive
            spike_fusion, post_fusion, pre_fusion = self.fusion_lif(
                syn_fusion, mem_fusion
            )
            mem_fusion = post_fusion

            evidence.append(self.output_linear(spike_fusion))
            if collect_states:
                assert hidden_lists is not None
                values = {
                    "l1": (syn_l1, pre_l1, spike_l1, post_l1),
                    "rsnn": (syn_rsnn, pre_rsnn, spike_rsnn, post_rsnn),
                    "fusion": (
                        syn_fusion,
                        pre_fusion,
                        spike_fusion,
                        post_fusion,
                    ),
                }
                for layer, (
                    syn_value,
                    pre_value,
                    spike_value,
                    post_value,
                ) in values.items():
                    hidden_lists[layer]["syn_current"].append(syn_value)
                    hidden_lists[layer]["pre_reset"].append(pre_value)
                    hidden_lists[layer]["spike"].append(spike_value)
                    hidden_lists[layer]["post_reset"].append(post_value)
                rsnn_external_inputs.append(rsnn_external)
                rsnn_recurrent_inputs.append(rsnn_recurrent)

        payload: dict[str, Any] = {
            "evidence": torch.stack(evidence, dim=1),
        }
        if collect_states:
            assert hidden_lists is not None
            payload["hidden"] = {
                layer: {
                    state: torch.stack(values, dim=1)
                    for state, values in state_map.items()
                }
                for layer, state_map in hidden_lists.items()
            }
            payload["diagnostics"] = {
                "rsnn_external_input": torch.stack(
                    rsnn_external_inputs, dim=1
                ),
                "rsnn_recurrent_input": torch.stack(
                    rsnn_recurrent_inputs, dim=1
                ),
            }
        return payload


def _initialize_l1_from_source(
    model: Exp110Net,
    spec: RunSpec,
    config: Config,
    split_hashes: Mapping[str, str],
) -> dict[str, Any]:
    if spec.l1_init == L1_INIT_DYNAMICS:
        return {
            "mode": L1_INIT_DYNAMICS,
            "source_checkpoint": None,
            "copied_parameter": None,
            "initial_max_abs_delta": None,
        }

    checkpoint, path = _load_source_checkpoint(
        config, spec.seed, split_hashes
    )
    source_weight = checkpoint["model_state_dict"]["hidden_linears.0.weight"]
    if tuple(source_weight.shape) != tuple(model.l1_input.weight.shape):
        raise RuntimeError(
            f"{spec.key}: L1 source shape {tuple(source_weight.shape)} "
            f"!= target {tuple(model.l1_input.weight.shape)}"
        )
    with torch.no_grad():
        model.l1_input.weight.copy_(
            source_weight.to(
                device=model.l1_input.weight.device,
                dtype=model.l1_input.weight.dtype,
            )
        )
    delta = float(
        (
            model.l1_input.weight.detach().cpu()
            - source_weight.detach().cpu()
        )
        .abs()
        .max()
        .item()
    )
    if delta != 0.0:
        raise RuntimeError(f"{spec.key}: failed exact L1 input-weight transfer")
    return {
        "mode": L1_INIT_PRETRAINED,
        "source_checkpoint": str(path.relative_to(config.repo_root)),
        "source_best_epoch": int(checkpoint["best_epoch"]),
        "source_spec": checkpoint["spec"],
        "copied_parameter": "hidden_linears.0.weight -> l1_input.weight",
        "initial_max_abs_delta": delta,
    }


def _valid_mean(values: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
    return exp101._valid_mean(values, lengths)


def _valid_mask(lengths: torch.Tensor, steps: int) -> torch.Tensor:
    return exp101._valid_mask(lengths, steps)


def _native_scores(
    model: Exp110Net,
    X: torch.Tensor,
    lengths: torch.Tensor,
) -> torch.Tensor:
    evidence = model.forward_trajectory(X, collect_states=False)["evidence"]
    return _valid_mean(evidence, lengths)


def _evaluate_native(
    model: Exp110Net,
    loader: Iterable,
    device: torch.device,
) -> tuple[dict[str, float], np.ndarray, np.ndarray]:
    ys: list[np.ndarray] = []
    preds: list[np.ndarray] = []
    loss_sum = 0.0
    n_total = 0

    model.eval()
    with torch.no_grad():
        for X, y, lengths in loader:
            Xd = X.to(device=device, dtype=torch.float32)
            yd = y.to(device)
            ld = lengths.to(device=device, dtype=torch.long)
            scores = _native_scores(model, Xd, ld)
            loss = F.cross_entropy(scores, yd)
            ys.append(y.numpy())
            preds.append(scores.argmax(dim=1).cpu().numpy())
            loss_sum += float(loss) * len(y)
            n_total += len(y)

    y_true = np.concatenate(ys)
    y_pred = np.concatenate(preds)
    metrics = exp10._classification_metrics(y_true, y_pred)
    metrics["objective_loss"] = loss_sum / max(n_total, 1)
    return metrics, y_true, y_pred


def probe_names() -> tuple[str, ...]:
    names: list[str] = []
    for layer in LAYERS:
        for state in ANALOG_STATES:
            for support in SUPPORTS:
                for aggregation in ANALOG_AGGREGATIONS:
                    names.append(
                        f"{layer}__{state}__{support}__{aggregation}"
                    )
        for support in SUPPORTS:
            for aggregation in COMMUNICATION_AGGREGATIONS:
                names.append(
                    f"{layer}__{COMMUNICATION_STATE}__"
                    f"{support}__{aggregation}"
                )
    if len(names) != EXPECTED_PROBES_PER_RUN:
        raise RuntimeError(
            f"Expected {EXPECTED_PROBES_PER_RUN} probes, got {len(names)}"
        )
    if len(names) != len(set(names)):
        raise RuntimeError("Duplicate Exp11.0 probe names")
    return tuple(names)


def _descriptor(name: str) -> tuple[str, str, str, str]:
    layer, state, support, aggregation = name.split("__", 3)
    return layer, state, support, aggregation


def _aggregate(
    values: torch.Tensor,
    lengths: torch.Tensor,
    support: str,
    aggregation: str,
    bin_steps: int,
) -> torch.Tensor:
    return exp1022._aggregate(
        values,
        lengths,
        support,
        aggregation,
        bin_steps,
    )


def _collect_probe_features(
    model: Exp110Net,
    data: exp3.Data,
    spec: RunSpec,
    config: Config,
) -> tuple[
    dict[str, dict[str, np.ndarray]],
    dict[str, np.ndarray],
]:
    loaders = exp73._raw_loaders(
        data, spec.seed, config.batch_size, False
    )
    device = torch.device(config.device)
    names = probe_names()
    feature_parts: dict[str, dict[str, list[np.ndarray]]] = {
        split: {name: [] for name in names}
        for split in SPLITS
    }
    label_parts: dict[str, list[np.ndarray]] = {
        split: [] for split in SPLITS
    }

    model.eval()
    with torch.no_grad():
        for split in SPLITS:
            for X, y, lengths in loaders[split]:
                Xd = X.to(device=device, dtype=torch.float32)
                ld = lengths.to(device=device, dtype=torch.long)
                hidden = model.forward_trajectory(
                    Xd, collect_states=True
                )["hidden"]

                for layer in LAYERS:
                    for state in ANALOG_STATES:
                        values = hidden[layer][state]
                        for support in SUPPORTS:
                            for aggregation in ANALOG_AGGREGATIONS:
                                name = (
                                    f"{layer}__{state}__{support}__"
                                    f"{aggregation}"
                                )
                                feature_parts[split][name].append(
                                    _aggregate(
                                        values,
                                        ld,
                                        support,
                                        aggregation,
                                        data.bin_steps,
                                    )
                                    .cpu()
                                    .numpy()
                                    .astype(np.float32, copy=False)
                                )

                    communication = hidden[layer]["spike"]
                    for support in SUPPORTS:
                        for aggregation in COMMUNICATION_AGGREGATIONS:
                            name = (
                                f"{layer}__{COMMUNICATION_STATE}__"
                                f"{support}__{aggregation}"
                            )
                            feature_parts[split][name].append(
                                _aggregate(
                                    communication,
                                    ld,
                                    support,
                                    aggregation,
                                    data.bin_steps,
                                )
                                .cpu()
                                .numpy()
                                .astype(np.float32, copy=False)
                            )

                label_parts[split].append(
                    y.numpy().astype(np.int64, copy=False)
                )

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
    return features, labels


def _probe_seed(spec: RunSpec, probe_name: str) -> int:
    return int(
        exp3.dseed(
            spec.seed,
            EXPERIMENT_ID,
            "probe",
            spec.l1_init,
            spec.recurrence,
            probe_name,
        )
    )


def _fit_probes(
    spec: RunSpec,
    features: dict[str, dict[str, np.ndarray]],
    labels: dict[str, np.ndarray],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    for name in probe_names():
        train_x = features["train"][name].astype(np.float64, copy=False)
        val_x = features["val"][name].astype(np.float64, copy=False)
        test_x = features["test"][name].astype(np.float64, copy=False)

        scaler = StandardScaler().fit(train_x)
        transformed = {
            "train": scaler.transform(train_x),
            "val": scaler.transform(val_x),
            "test": scaler.transform(test_x),
        }

        random_state = _probe_seed(spec, name)
        best: tuple[float, float, LogisticRegression] | None = None
        candidates: list[dict[str, float]] = []
        for C in C_GRID:
            classifier = LogisticRegression(
                C=C,
                max_iter=PROBE_MAX_ITER,
                solver="lbfgs",
                random_state=random_state,
                fit_intercept=True,
            ).fit(transformed["train"], labels["train"])
            val_pred = classifier.predict(transformed["val"])
            val_metrics = exp10._classification_metrics(
                labels["val"], val_pred
            )
            val_ba = float(val_metrics["balanced_accuracy"])
            candidates.append(
                {
                    "C": float(C),
                    "val_balanced_accuracy": val_ba,
                    "val_macro_f1": float(val_metrics["macro_f1"]),
                    "val_accuracy": float(val_metrics["accuracy"]),
                }
            )
            if best is None or val_ba > best[0] + 1e-12:
                best = (val_ba, float(C), classifier)

        if best is None:
            raise RuntimeError(f"No probe selected for {spec.key}/{name}")

        selected_val_ba, selected_C, classifier = best
        metrics = {
            split: exp10._classification_metrics(
                labels[split],
                classifier.predict(transformed[split]),
            )
            for split in SPLITS
        }
        layer, state, support, aggregation = _descriptor(name)
        row: dict[str, Any] = {
            "experiment_id": EXPERIMENT_ID,
            "protocol_version": PROTOCOL_VERSION,
            "l1_init": spec.l1_init,
            "recurrence": spec.recurrence,
            "seed": spec.seed,
            "probe": name,
            "layer": layer,
            "state": state,
            "support": support,
            "aggregation": aggregation,
            "feature_dim": int(train_x.shape[1]),
            "selected_C": selected_C,
            "selected_val_balanced_accuracy": selected_val_ba,
            "probe_seed": random_state,
            "candidate_validation_json": json.dumps(
                candidates, sort_keys=True
            ),
        }
        for split in SPLITS:
            for metric, value in metrics[split].items():
                row[f"{split}_{metric}"] = float(value)
        rows.append(row)

    frame = pd.DataFrame(rows)
    if len(frame) != EXPECTED_PROBES_PER_RUN:
        raise RuntimeError(
            f"{spec.key}: expected {EXPECTED_PROBES_PER_RUN} probe rows, "
            f"got {len(frame)}"
        )
    return frame


def _dynamics_metrics(
    model: Exp110Net,
    loader: Iterable,
    device: torch.device,
    fs: float,
) -> dict[str, float]:
    valid_events = {layer: 0.0 for layer in LAYERS}
    valid_neuron_steps = {layer: 0.0 for layer in LAYERS}
    post_events = {layer: 0.0 for layer in LAYERS}
    post_neuron_steps = {layer: 0.0 for layer in LAYERS}
    external_abs = 0.0
    recurrent_abs = 0.0
    valid_rsnn_values = 0

    model.eval()
    with torch.no_grad():
        for X, _, lengths in loader:
            Xd = X.to(device=device, dtype=torch.float32)
            ld = lengths.to(device=device, dtype=torch.long)
            trajectory = model.forward_trajectory(
                Xd, collect_states=True
            )
            hidden = trajectory["hidden"]
            diagnostics = trajectory["diagnostics"]
            steps = Xd.shape[1]
            valid = _valid_mask(ld, steps)
            post = ~valid

            for layer in LAYERS:
                spikes = hidden[layer]["spike"]
                valid_mask = valid.to(spikes.dtype).unsqueeze(-1)
                post_mask = post.to(spikes.dtype).unsqueeze(-1)
                valid_events[layer] += float(
                    (spikes * valid_mask).sum().item()
                )
                post_events[layer] += float(
                    (spikes * post_mask).sum().item()
                )
                valid_neuron_steps[layer] += float(
                    valid.sum().item() * WIDTH
                )
                post_neuron_steps[layer] += float(
                    post.sum().item() * WIDTH
                )

            ext = diagnostics["rsnn_external_input"]
            rec = diagnostics["rsnn_recurrent_input"]
            valid_flat = valid.unsqueeze(-1).expand_as(ext)
            external_abs += float(ext[valid_flat].abs().sum().item())
            recurrent_abs += float(rec[valid_flat].abs().sum().item())
            valid_rsnn_values += int(valid_flat.sum().item())

    effective = model.effective_recurrent_weight().detach().cpu()
    if model.effective_recurrent_parameter_count() == 0:
        spectral_radius = 0.0
    else:
        eigvals = torch.linalg.eigvals(effective)
        spectral_radius = float(eigvals.abs().max().item())

    output: dict[str, float] = {
        "rsnn_external_mean_abs_valid": (
            external_abs / max(valid_rsnn_values, 1)
        ),
        "rsnn_recurrent_mean_abs_valid": (
            recurrent_abs / max(valid_rsnn_values, 1)
        ),
        "rsnn_recurrent_to_external_ratio": (
            recurrent_abs / max(external_abs, 1e-12)
        ),
        "effective_recurrent_weight_fro_norm": float(
            effective.norm().item()
        ),
        "effective_recurrent_spectral_radius": spectral_radius,
        "effective_recurrent_parameter_count": float(
            model.effective_recurrent_parameter_count()
        ),
    }
    for layer in LAYERS:
        output[f"{layer}_valid_events_per_neuron_s"] = (
            valid_events[layer]
            / max(valid_neuron_steps[layer], 1.0)
            * float(fs)
        )
        output[f"{layer}_postvalid_events_per_neuron_s"] = (
            post_events[layer]
            / max(post_neuron_steps[layer], 1.0)
            * float(fs)
        )
    return output


def _run_artifacts(config: Config, spec: RunSpec) -> dict[str, Path]:
    return {
        "checkpoint": _path(
            config.results_dir, "checkpoints", spec.key, ".pt"
        ),
        "history": _path(
            config.results_dir, "histories", spec.key, ".csv"
        ),
        "evaluation": _path(
            config.results_dir, "evaluations", spec.key, ".json"
        ),
        "probes": _path(
            config.results_dir, "probe_evaluations", spec.key, ".csv"
        ),
    }


def run_one(
    spec: RunSpec,
    config: Config,
    force: bool = False,
) -> dict[str, Any]:
    validate_spec(spec)
    artifacts = _run_artifacts(config, spec)
    if not force and all(path.exists() for path in artifacts.values()):
        return json.loads(
            artifacts["evaluation"].read_text(encoding="utf-8")
        )

    _require_source_manifests(config)
    data, frames, _ = _prepare_data(config)
    split_hashes = _split_hashes(frames)
    torch.set_num_threads(config.threads)
    device = torch.device(config.device)

    model_init_seed = exp73._e2e_pair_seed(spec.seed, "model_init")
    exp3.seed_all(model_init_seed)
    model = Exp110Net(spec, len(data.labels), data.fs).to(device)
    l1_init_provenance = _initialize_l1_from_source(
        model, spec, config, split_hashes
    )

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
        train_loss_sum = 0.0
        n_total = 0
        grad_norm_sum = 0.0
        n_batches = 0
        for X, y, lengths in train_loader:
            Xd = X.to(device=device, dtype=torch.float32)
            yd = y.to(device)
            ld = lengths.to(device=device, dtype=torch.long)
            optimizer.zero_grad(set_to_none=True)
            scores = _native_scores(model, Xd, ld)
            loss = F.cross_entropy(scores, yd)
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), GRAD_CLIP_NORM
            )
            optimizer.step()
            train_loss_sum += float(loss.detach()) * len(y)
            n_total += len(y)
            grad_norm_sum += float(grad_norm)
            n_batches += 1

        train_metrics, _, _ = _evaluate_native(
            model, eval_loaders["train"], device
        )
        val_metrics, _, _ = _evaluate_native(
            model, eval_loaders["val"], device
        )
        history.append(
            {
                "epoch": float(epoch),
                "train_ba": float(
                    train_metrics["balanced_accuracy"]
                ),
                "val_ba": float(val_metrics["balanced_accuracy"]),
                "train_loss": train_loss_sum / max(n_total, 1),
                "val_loss": float(val_metrics["objective_loss"]),
                "mean_preclip_grad_norm": (
                    grad_norm_sum / max(n_batches, 1)
                ),
            }
        )

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

    artifacts["checkpoint"].parent.mkdir(
        parents=True, exist_ok=True
    )
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
            "l1_init_provenance": l1_init_provenance,
            "model_state_dict": best_state,
        },
        artifacts["checkpoint"],
    )
    artifacts["history"].parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(history).to_csv(
        artifacts["history"], index=False
    )

    model.load_state_dict(best_state, strict=True)
    native_metrics: dict[str, dict[str, float]] = {}
    native_arrays: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for split in SPLITS:
        metrics, y_true, y_pred = _evaluate_native(
            model, eval_loaders[split], device
        )
        native_metrics[split] = metrics
        native_arrays[split] = (y_true, y_pred)

    features, probe_labels = _collect_probe_features(
        model, data, spec, config
    )
    for split in SPLITS:
        if not np.array_equal(
            probe_labels[split], native_arrays[split][0]
        ):
            raise RuntimeError(
                f"{spec.key}/{split}: probe label order mismatch"
            )
    probes = _fit_probes(spec, features, probe_labels)
    artifacts["probes"].parent.mkdir(parents=True, exist_ok=True)
    probes.to_csv(artifacts["probes"], index=False)

    dynamics = _dynamics_metrics(
        model, eval_loaders["test"], device, data.fs
    )
    payload = {
        "experiment_id": EXPERIMENT_ID,
        "protocol_version": PROTOCOL_VERSION,
        "spec": asdict(spec),
        "contract": {
            "dataset": VARIANT,
            "rotation": ROTATION,
            "architecture": "30->L1(128)->RSNN(128)->Fusion(128)->Linear(12)",
            "l1_mem_shift": L1_MEM_SHIFT,
            "l1_tau_mem_ms": exp1021.tau_mem_ms_from_shift(
                L1_MEM_SHIFT, data.fs
            ),
            "l1_synaptic_shifts": list(L1_SYN_SHIFTS),
            "rsnn_tau_syn_ms": RSNN_TAU_SYN_MS,
            "rsnn_tau_mem_ms": RSNN_TAU_MEM_MS,
            "fusion_tau_syn_ms": FUSION_TAU_SYN_MS,
            "fusion_tau_mem_ms": FUSION_TAU_MEM_MS,
            "rsnn_width": WIDTH,
            "fusion_width": WIDTH,
            "communication": "binary spikes",
            "recurrence_source": "previous RSNN binary spike",
            "fusion_inputs": "current L1 binary spike + current RSNN binary spike",
            "fusion_recurrent": False,
            "readout": "single shared bias-free Linear 128->12",
            "objective": "valid-length mean WCCE only",
            "gradient_clip_norm": GRAD_CLIP_NORM,
            "all_layers_trainable_after_initialization": True,
        },
        "l1_init_provenance": l1_init_provenance,
        "model_init_seed": int(model_init_seed),
        "split_sample_hashes": split_hashes,
        "best_epoch": best_epoch,
        "stopped_epoch": stopped_epoch,
        "native_metrics": native_metrics,
        "dynamics_metrics": dynamics,
        "probe_count": int(len(probes)),
        "parameter_count": int(
            sum(parameter.numel() for parameter in model.parameters())
        ),
        "effective_recurrent_parameter_count": (
            model.effective_recurrent_parameter_count()
        ),
    }
    _save_json(artifacts["evaluation"], payload)
    return payload


def _probe_ba(
    probes: pd.DataFrame,
    *,
    layer: str,
    state: str,
    support: str,
    aggregation: str,
) -> float:
    selected = probes[
        (probes.layer == layer)
        & (probes.state == state)
        & (probes.support == support)
        & (probes.aggregation == aggregation)
    ]
    if len(selected) != 1:
        raise RuntimeError(
            f"Expected one probe row for "
            f"{layer}/{state}/{support}/{aggregation}; got {len(selected)}"
        )
    return float(selected.iloc[0].test_balanced_accuracy)


def _run_row(
    payload: Mapping[str, Any],
    probes: pd.DataFrame,
) -> dict[str, Any]:
    spec = payload["spec"]
    row: dict[str, Any] = {
        "l1_init": spec["l1_init"],
        "recurrence": spec["recurrence"],
        "seed": int(spec["seed"]),
        "best_epoch": int(payload["best_epoch"]),
        "stopped_epoch": int(payload["stopped_epoch"]),
        "native_train_ba": float(
            payload["native_metrics"]["train"]["balanced_accuracy"]
        ),
        "native_val_ba": float(
            payload["native_metrics"]["val"]["balanced_accuracy"]
        ),
        "native_test_ba": float(
            payload["native_metrics"]["test"]["balanced_accuracy"]
        ),
        "parameter_count": int(payload["parameter_count"]),
        "effective_recurrent_parameter_count": int(
            payload["effective_recurrent_parameter_count"]
        ),
    }
    for key, value in payload["dynamics_metrics"].items():
        row[key] = float(value)

    for layer in LAYERS:
        whole = _probe_ba(
            probes,
            layer=layer,
            state=COMMUNICATION_STATE,
            support=exp1022.SUPPORT_VALID,
            aggregation="whole_count",
        )
        fixed = _probe_ba(
            probes,
            layer=layer,
            state=COMMUNICATION_STATE,
            support=exp1022.SUPPORT_VALID,
            aggregation="fixed250_count",
        )
        pre_whole = _probe_ba(
            probes,
            layer=layer,
            state="pre_reset",
            support=exp1022.SUPPORT_VALID,
            aggregation="whole_mean",
        )
        pre_fixed = _probe_ba(
            probes,
            layer=layer,
            state="pre_reset",
            support=exp1022.SUPPORT_VALID,
            aggregation="fixed250_ordered_mean",
        )
        row[f"{layer}_comm_valid_whole_ba"] = whole
        row[f"{layer}_comm_valid_fixed250_ba"] = fixed
        row[f"{layer}_comm_temporal_gap"] = fixed - whole
        row[f"{layer}_pre_valid_whole_ba"] = pre_whole
        row[f"{layer}_pre_valid_fixed250_ba"] = pre_fixed
        row[f"{layer}_pre_temporal_gap"] = pre_fixed - pre_whole

    row["fusion_comm_gap_reduction_vs_l1"] = (
        row["l1_comm_temporal_gap"]
        - row["fusion_comm_temporal_gap"]
    )
    row["fusion_comm_whole_gain_vs_l1"] = (
        row["fusion_comm_valid_whole_ba"]
        - row["l1_comm_valid_whole_ba"]
    )
    row["fusion_pre_gap_reduction_vs_l1"] = (
        row["l1_pre_temporal_gap"]
        - row["fusion_pre_temporal_gap"]
    )
    return row


def _contrast_metrics() -> tuple[str, ...]:
    return (
        "native_test_ba",
        "l1_comm_valid_whole_ba",
        "l1_comm_valid_fixed250_ba",
        "l1_comm_temporal_gap",
        "rsnn_comm_valid_whole_ba",
        "rsnn_comm_valid_fixed250_ba",
        "rsnn_comm_temporal_gap",
        "fusion_comm_valid_whole_ba",
        "fusion_comm_valid_fixed250_ba",
        "fusion_comm_temporal_gap",
        "fusion_comm_gap_reduction_vs_l1",
        "fusion_comm_whole_gain_vs_l1",
        "rsnn_recurrent_to_external_ratio",
        "rsnn_valid_events_per_neuron_s",
        "fusion_valid_events_per_neuron_s",
    )


def _paired_contrasts(runs: pd.DataFrame) -> pd.DataFrame:
    indexed = runs.set_index(["l1_init", "recurrence", "seed"])
    rows: list[dict[str, Any]] = []

    def emit(
        contrast: str,
        left: tuple[str, str, int],
        right: tuple[str, str, int],
    ) -> None:
        lrow = indexed.loc[left]
        rrow = indexed.loc[right]
        row: dict[str, Any] = {
            "contrast": contrast,
            "l1_init": (
                left[0]
                if left[0] == right[0]
                else "pretrained_vs_dynamics"
            ),
            "recurrence": (
                left[1]
                if left[1] == right[1]
                else f"{left[1]}_vs_{right[1]}"
            ),
            "seed": left[2],
        }
        for metric in _contrast_metrics():
            row[f"delta_{metric}"] = float(
                lrow[metric] - rrow[metric]
            )
        rows.append(row)

    for seed in MODEL_SEEDS:
        for l1_init in L1_INITS:
            emit(
                "diagonal_minus_none",
                (l1_init, RECURRENCE_DIAGONAL, seed),
                (l1_init, RECURRENCE_NONE, seed),
            )
            emit(
                "dense_minus_none",
                (l1_init, RECURRENCE_DENSE, seed),
                (l1_init, RECURRENCE_NONE, seed),
            )
            emit(
                "dense_minus_diagonal",
                (l1_init, RECURRENCE_DENSE, seed),
                (l1_init, RECURRENCE_DIAGONAL, seed),
            )
        for recurrence in RECURRENCES:
            emit(
                "pretrained_minus_dynamics",
                (L1_INIT_PRETRAINED, recurrence, seed),
                (L1_INIT_DYNAMICS, recurrence, seed),
            )
    return pd.DataFrame(rows)


def _interaction_rows(runs: pd.DataFrame) -> pd.DataFrame:
    indexed = runs.set_index(["l1_init", "recurrence", "seed"])
    rows: list[dict[str, Any]] = []
    for seed in MODEL_SEEDS:
        for recurrence in (RECURRENCE_DIAGONAL, RECURRENCE_DENSE):
            row: dict[str, Any] = {
                "interaction": (
                    f"pretraining_x_{recurrence}_vs_none"
                ),
                "recurrence": recurrence,
                "seed": seed,
            }
            for metric in _contrast_metrics():
                pretrained_effect = float(
                    indexed.loc[
                        (L1_INIT_PRETRAINED, recurrence, seed),
                        metric,
                    ]
                    - indexed.loc[
                        (L1_INIT_DYNAMICS, recurrence, seed),
                        metric,
                    ]
                )
                pretrained_none = float(
                    indexed.loc[
                        (
                            L1_INIT_PRETRAINED,
                            RECURRENCE_NONE,
                            seed,
                        ),
                        metric,
                    ]
                    - indexed.loc[
                        (
                            L1_INIT_DYNAMICS,
                            RECURRENCE_NONE,
                            seed,
                        ),
                        metric,
                    ]
                )
                row[f"interaction_{metric}"] = (
                    pretrained_effect - pretrained_none
                )
            rows.append(row)
    return pd.DataFrame(rows)


def _flatten_summary(frame: pd.DataFrame) -> pd.DataFrame:
    output = frame.copy()
    output.columns = [
        "_".join(str(part) for part in column if str(part))
        if isinstance(column, tuple)
        else str(column)
        for column in output.columns
    ]
    return output


def finalize(config: Config) -> dict[str, Any]:
    _require_source_manifests(config)
    payloads: list[dict[str, Any]] = []
    probe_frames: list[pd.DataFrame] = []
    missing: list[str] = []

    for spec in run_specs():
        artifacts = _run_artifacts(config, spec)
        for name in ("evaluation", "probes"):
            if not artifacts[name].exists():
                missing.append(
                    f"{spec.key}:{name}:{artifacts[name]}"
                )
        if artifacts["evaluation"].exists():
            payloads.append(
                json.loads(
                    artifacts["evaluation"].read_text(
                        encoding="utf-8"
                    )
                )
            )
        if artifacts["probes"].exists():
            probe_frames.append(
                pd.read_csv(artifacts["probes"])
            )

    if missing:
        raise FileNotFoundError(
            f"Exp11.0 incomplete; missing {len(missing)} artifacts:\n"
            + "\n".join(missing[:40])
        )
    if (
        len(payloads) != EXPECTED_RUNS
        or len(probe_frames) != EXPECTED_RUNS
    ):
        raise RuntimeError(
            f"Expected {EXPECTED_RUNS} runs, got "
            f"{len(payloads)} evaluations and "
            f"{len(probe_frames)} probe files"
        )

    runs = pd.DataFrame(
        [
            _run_row(payload, probes)
            for payload, probes in zip(
                payloads, probe_frames, strict=True
            )
        ]
    ).sort_values(["l1_init", "recurrence", "seed"])
    runs.to_csv(
        config.results_dir / "run_metrics.csv", index=False
    )

    metric_cols = [
        column
        for column in runs.columns
        if column
        not in {
            "l1_init",
            "recurrence",
            "seed",
            "best_epoch",
            "stopped_epoch",
        }
    ]
    summary = (
        runs.groupby(["l1_init", "recurrence"], sort=True)[
            metric_cols
        ]
        .agg(["count", "mean", "std"])
        .reset_index()
    )
    _flatten_summary(summary).to_csv(
        config.results_dir / "method_summary.csv", index=False
    )

    contrasts = _paired_contrasts(runs)
    contrasts.to_csv(
        config.results_dir / "paired_contrasts.csv", index=False
    )
    delta_cols = [
        column
        for column in contrasts.columns
        if column.startswith("delta_")
    ]
    contrast_summary = (
        contrasts.groupby(
            ["contrast", "l1_init", "recurrence"],
            sort=False,
        )[delta_cols]
        .agg(["count", "mean", "std"])
        .reset_index()
    )
    _flatten_summary(contrast_summary).to_csv(
        config.results_dir / "paired_contrast_summary.csv",
        index=False,
    )

    interactions = _interaction_rows(runs)
    interactions.to_csv(
        config.results_dir / "interaction_runs.csv", index=False
    )
    interaction_cols = [
        column
        for column in interactions.columns
        if column.startswith("interaction_")
    ]
    interaction_summary = (
        interactions.groupby(
            ["interaction", "recurrence"], sort=True
        )[interaction_cols]
        .agg(["count", "mean", "std"])
        .reset_index()
    )
    _flatten_summary(interaction_summary).to_csv(
        config.results_dir / "interaction_summary.csv",
        index=False,
    )

    all_probes = pd.concat(probe_frames, ignore_index=True)
    all_probes.to_csv(
        config.results_dir / "probe_runs.csv", index=False
    )
    probe_summary = (
        all_probes.groupby(
            [
                "l1_init",
                "recurrence",
                "layer",
                "state",
                "support",
                "aggregation",
                "probe",
            ],
            sort=False,
        )["test_balanced_accuracy"]
        .agg(["count", "mean", "std"])
        .reset_index()
    )
    probe_summary.to_csv(
        config.results_dir / "probe_summary.csv", index=False
    )

    mechanism_cols = [
        "native_test_ba",
        "l1_comm_valid_whole_ba",
        "l1_comm_valid_fixed250_ba",
        "l1_comm_temporal_gap",
        "rsnn_comm_valid_whole_ba",
        "rsnn_comm_valid_fixed250_ba",
        "rsnn_comm_temporal_gap",
        "fusion_comm_valid_whole_ba",
        "fusion_comm_valid_fixed250_ba",
        "fusion_comm_temporal_gap",
        "fusion_comm_gap_reduction_vs_l1",
        "fusion_comm_whole_gain_vs_l1",
    ]
    mechanism_summary = (
        runs.groupby(["l1_init", "recurrence"], sort=True)[
            mechanism_cols
        ]
        .agg(["count", "mean", "std"])
        .reset_index()
    )
    _flatten_summary(mechanism_summary).to_csv(
        config.results_dir / "mechanism_summary.csv",
        index=False,
    )

    manifest = {
        "experiment_id": EXPERIMENT_ID,
        "protocol_version": PROTOCOL_VERSION,
        "status": "PASS",
        "dataset": VARIANT,
        "rotation": ROTATION,
        "l1_inits": list(L1_INITS),
        "recurrences": list(RECURRENCES),
        "model_seeds": list(MODEL_SEEDS),
        "run_count": int(len(runs)),
        "expected_run_count": EXPECTED_RUNS,
        "expected_probes_per_run": EXPECTED_PROBES_PER_RUN,
        "objective": "valid-length mean WCCE only",
        "supports": list(SUPPORTS),
        "main_mechanism_metric": (
            "communication Fixed250 BA - communication whole-count BA"
        ),
        "success_direction": (
            "fusion whole BA increases while fusion temporal gap shrinks "
            "relative to L1"
        ),
        "source_checkpoint_condition": (
            "Exp10.2.1 binary, L1 mem shift2, L2 mem shift1, "
            "seed matched; Exp10.2.2 dual-support evaluation required"
        ),
        "statistical_scope": (
            "one locked cross-user split; seeds 11/23/37 are paired "
            "optimization replicates, not independent user splits"
        ),
        "primary_outputs": [
            "run_metrics.csv",
            "method_summary.csv",
            "paired_contrasts.csv",
            "paired_contrast_summary.csv",
            "interaction_runs.csv",
            "interaction_summary.csv",
            "probe_runs.csv",
            "probe_summary.csv",
            "mechanism_summary.csv",
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
            "Exp11.0: internalize Fixed250 temporal decoding with "
            "L1 local features, RSNN context, and a fusion SNN"
        )
    )
    parser.add_argument("--repo-root", default=None)
    parser.add_argument("--results-dir", default=None)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument(
        "--batch-size", type=int, default=BATCH_SIZE
    )
    parser.add_argument(
        "--max-epochs", type=int, default=MAX_EPOCHS
    )
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("prepare")
    sub.add_parser("list-runs")
    run = sub.add_parser("run-one")
    run.add_argument(
        "--array-task-id", type=int, required=True
    )
    run.add_argument("--force", action="store_true")
    sub.add_parser("finalize")
    return parser


def main() -> None:
    args = _parser().parse_args()
    config = _resolve_config(args)

    if args.command == "prepare":
        print(
            json.dumps(
                prepare_all(config), indent=2, sort_keys=True
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
                finalize(config), indent=2, sort_keys=True
            )
        )
        return
    raise ValueError(args.command)


if __name__ == "__main__":
    main()
