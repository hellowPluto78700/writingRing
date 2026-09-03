from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import torch
from torch import nn
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from scripts import experiment_4_0_5_temporal_resolution_event_capacity as exp405
from scripts import experiment_4_0_6_local_mem_raw64 as exp406


EXPERIMENT_ID = "experiment_4_3_long_term_memory_validation"
PROTOCOL_VERSION = "stage2_recurrence_memory_validation_v1"
LOCAL_CONDITION = "shift34"
MODEL_KINDS = ("ff", "rsnn")
SEEDS = exp406.SEEDS
STATE_RESET_MS = (250, 500, 1000)
SILENT_DELAY_MS = (0, 250, 500, 1000, 2000)
PROBE_MAX_ITER = exp406.PROBE_MAX_ITER
BATCH_SIZE = exp406.BATCH_SIZE
EPOCHS = exp406.EPOCHS
LR = exp406.LR
WEIGHT_DECAY = exp406.WEIGHT_DECAY

_MULTI_HO = next(item for item in exp405.VARIANTS if item[0] == "multi_ho")
VARIANT, HIDDEN_CAP, OUTPUT_CAP = _MULTI_HO

ModelKind = Literal["ff", "rsnn"]


@dataclass(frozen=True)
class RunSpec:
    model_kind: ModelKind
    seed: int

    @property
    def key(self) -> str:
        return f"{self.model_kind}__{LOCAL_CONDITION}__{VARIANT}__seed{self.seed}"


@dataclass(frozen=True)
class Config:
    repo_root: Path
    results_dir: Path
    device: str = "cpu"
    epochs: int = EPOCHS
    batch_size: int = BATCH_SIZE
    threads: int = 1


@dataclass
class ConditionFeatures:
    output_logits: np.ndarray
    hidden_count: np.ndarray
    uend: np.ndarray
    labels: np.ndarray


def results_dir(repo_root: Path) -> Path:
    return repo_root / "notebooks" / "artifacts" / EXPERIMENT_ID / PROTOCOL_VERSION


def run_specs(model_kind: ModelKind | None = None) -> list[RunSpec]:
    kinds = MODEL_KINDS if model_kind is None else (model_kind,)
    return [RunSpec(kind, seed) for kind in kinds for seed in SEEDS]


def source_spec(seed: int) -> exp406.RunSpec:
    return exp406.RunSpec(
        condition=LOCAL_CONDITION,
        variant=VARIANT,
        hidden_cap=HIDDEN_CAP,
        output_cap=OUTPUT_CAP,
        seed=seed,
    )


def paired_seed(spec: RunSpec, role: str) -> int:
    return exp406.paired_seed(source_spec(spec.seed), role)


def _ms_to_steps(ms: int, dt_ms: float) -> int:
    steps = int(round(float(ms) / float(dt_ms)))
    if steps < 1:
        raise ValueError(f"Expected positive interval, got {ms} ms")
    reconstructed = steps * float(dt_ms)
    if not np.isclose(reconstructed, float(ms), atol=1e-6):
        raise ValueError(
            f"{ms} ms is not an integer number of timesteps for dt={dt_ms} ms"
        )
    return steps


def reset_schedule(dt_ms: float) -> dict[int, int]:
    return {ms: _ms_to_steps(ms, dt_ms) for ms in STATE_RESET_MS}


def delay_schedule(dt_ms: float) -> dict[int, int]:
    return {0: 0, **{ms: _ms_to_steps(ms, dt_ms) for ms in SILENT_DELAY_MS if ms}}


class Stage2ValidationDecoder(exp406.LocalMemoryRaw64Decoder):
    """Exp4.0.6 shift34 decoder with explicit Stage-2 interventions.

    Local128 is never reset or ablated by the Stage-2 controls. The FF control
    has exactly the same feed-forward layers and passive state membrane as the
    RSNN, but the recurrent contribution is functionally disabled.
    """

    def __init__(
        self,
        model_kind: ModelKind,
        n_classes: int,
        fs: float = 64.0,
    ) -> None:
        if model_kind not in MODEL_KINDS:
            raise ValueError(f"Unknown model kind: {model_kind}")
        super().__init__(
            condition=LOCAL_CONDITION,
            n_classes=n_classes,
            hidden_cap=HIDDEN_CAP,
            output_cap=OUTPUT_CAP,
            fs=fs,
        )
        self.model_kind = model_kind

    def forward_trajectory(
        self,
        x: torch.Tensor,
        *,
        recurrence_enabled: bool | None = None,
        state_reset_interval_steps: int | None = None,
    ) -> dict[str, torch.Tensor]:
        batch, n_steps, channels = x.shape
        if channels != exp405.exp40.EVENT_CHANNELS:
            raise ValueError(
                f"Expected {exp405.exp40.EVENT_CHANNELS} channels, got {channels}"
            )
        if state_reset_interval_steps is not None and state_reset_interval_steps < 1:
            raise ValueError("state_reset_interval_steps must be >= 1")

        use_recurrence = self.model_kind == "rsnn"
        if recurrence_enabled is not None:
            if self.model_kind == "ff" and recurrence_enabled:
                raise ValueError("FF control cannot enable recurrence")
            use_recurrence = bool(recurrence_enabled)

        local_mem = torch.zeros(
            batch, exp405.LOCAL_WIDTH, device=x.device, dtype=x.dtype
        )
        state_mem = torch.zeros(
            batch, exp405.STATE_WIDTH, device=x.device, dtype=x.dtype
        )
        prev_state_spikes = torch.zeros_like(state_mem)
        output_mem = torch.zeros(
            batch, self.n_classes, device=x.device, dtype=x.dtype
        )

        local_spikes_seq: list[torch.Tensor] = []
        local_pre_seq: list[torch.Tensor] = []
        state_spikes_seq: list[torch.Tensor] = []
        state_mems_seq: list[torch.Tensor] = []
        state_pre_seq: list[torch.Tensor] = []
        output_spikes_seq: list[torch.Tensor] = []
        output_mems_seq: list[torch.Tensor] = []
        output_pre_seq: list[torch.Tensor] = []

        for step in range(n_steps):
            # A periodic reset destroys only Stage-2 recurrent state. Local128
            # and output membrane are intentionally left untouched.
            if (
                state_reset_interval_steps is not None
                and step > 0
                and step % state_reset_interval_steps == 0
            ):
                state_mem = torch.zeros_like(state_mem)
                prev_state_spikes = torch.zeros_like(prev_state_spikes)

            local_current = self.input_local(x[:, step])
            local_spikes, local_mem, local_pre = self.local_lif(
                local_current, local_mem
            )
            state_current = self.local_state(local_spikes)
            if use_recurrence:
                state_current = state_current + self.recurrent(prev_state_spikes)
            state_spikes, state_mem, state_pre = self.state_lif(
                state_current, state_mem
            )
            output_current = self.state_output(state_spikes)
            output_spikes, output_mem, output_pre = self.output_lif(
                output_current, output_mem
            )

            local_spikes_seq.append(local_spikes)
            local_pre_seq.append(local_pre)
            state_spikes_seq.append(state_spikes)
            state_mems_seq.append(state_mem)
            state_pre_seq.append(state_pre)
            output_spikes_seq.append(output_spikes)
            output_mems_seq.append(output_mem)
            output_pre_seq.append(output_pre)
            prev_state_spikes = state_spikes

        return {
            "local_spikes": torch.stack(local_spikes_seq, dim=1),
            "local_pre_reset": torch.stack(local_pre_seq, dim=1),
            "state_spikes": torch.stack(state_spikes_seq, dim=1),
            "state_membranes": torch.stack(state_mems_seq, dim=1),
            "state_pre_reset": torch.stack(state_pre_seq, dim=1),
            "output_spikes": torch.stack(output_spikes_seq, dim=1),
            "output_membranes": torch.stack(output_mems_seq, dim=1),
            "output_pre_reset": torch.stack(output_pre_seq, dim=1),
        }


def prepare_data(repo_root: Path) -> exp405.TemporalData:
    return exp406.prepare_raw64_data(repo_root)


def make_loaders(
    data: exp405.TemporalData,
    spec: RunSpec,
    batch_size: int,
    train_shuffle: bool,
) -> dict[str, torch.utils.data.DataLoader]:
    return exp406.make_loaders(
        data,
        source_spec(spec.seed),
        batch_size,
        train_shuffle,
    )


def checkpoint_path(root: Path, spec: RunSpec) -> Path:
    return root / "checkpoints" / f"{spec.key}.pt"


def evaluation_path(root: Path, spec: RunSpec) -> Path:
    return root / "evaluations" / f"{spec.key}.json"


def history_path(root: Path, spec: RunSpec) -> Path:
    return root / "histories" / f"{spec.key}.csv"


def _save_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _scaled_linear_probe() -> Pipeline:
    return Pipeline(
        [
            ("scale", StandardScaler()),
            (
                "classifier",
                LogisticRegression(
                    max_iter=PROBE_MAX_ITER,
                    class_weight="balanced",
                    solver="lbfgs",
                ),
            ),
        ]
    )


def train_ff(
    spec: RunSpec,
    data: exp405.TemporalData,
    config: Config,
    force: bool,
) -> Path:
    if spec.model_kind != "ff":
        raise ValueError("Only the FF control is newly trained in Exp4.3")
    destination = checkpoint_path(config.results_dir, spec)
    if destination.exists() and not force:
        return destination

    device = torch.device(config.device)
    torch.set_num_threads(config.threads)
    exp405.exp40.base.seed_all(paired_seed(spec, "model_init"))
    model = Stage2ValidationDecoder(
        model_kind="ff", n_classes=len(data.labels), fs=data.fs
    ).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY
    )
    loaders = make_loaders(data, spec, config.batch_size, train_shuffle=True)
    train_loader = loaders["train"]
    val_loader = make_loaders(
        data, spec, config.batch_size, train_shuffle=False
    )["val"]

    best_state: dict[str, torch.Tensor] | None = None
    best_epoch = -1
    best_val_ba = -float("inf")
    best_val_loss = float("inf")
    history: list[dict[str, float | int]] = []

    for epoch in range(1, config.epochs + 1):
        model.train()
        train_loss_sum = 0.0
        train_n = 0
        train_true: list[np.ndarray] = []
        train_pred: list[np.ndarray] = []
        for X, y, valid_steps in train_loader:
            X = X.to(device)
            y = y.to(device)
            valid_steps = valid_steps.to(device)
            optimizer.zero_grad(set_to_none=True)
            traj = model.forward_trajectory(X)
            logits = exp405.normalized_output_evidence(
                traj["output_spikes"], valid_steps, model.output_cap
            )
            loss = F.cross_entropy(logits, y)
            loss.backward()
            optimizer.step()

            n = len(y)
            train_n += n
            train_loss_sum += float(loss.item()) * n
            train_true.append(y.detach().cpu().numpy())
            train_pred.append(logits.detach().argmax(dim=1).cpu().numpy())

        train_metrics = exp405.exp40.metrics(
            np.concatenate(train_true), np.concatenate(train_pred)
        )
        val_metrics = exp405.evaluate_model(model, val_loader, device)
        val_ba = float(val_metrics["valid_count"]["balanced_accuracy"])
        val_loss = float(val_metrics["valid_count_loss"])
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss_sum / max(train_n, 1),
                "train_balanced_accuracy": train_metrics["balanced_accuracy"],
                "val_loss": val_loss,
                "val_balanced_accuracy": val_ba,
            }
        )
        improved = val_ba > best_val_ba + 1e-12 or (
            abs(val_ba - best_val_ba) <= 1e-12 and val_loss < best_val_loss
        )
        if improved:
            best_val_ba = val_ba
            best_val_loss = val_loss
            best_epoch = epoch
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }

    if best_state is None:
        raise RuntimeError(f"No checkpoint selected for {spec.key}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "experiment_id": EXPERIMENT_ID,
            "protocol_version": PROTOCOL_VERSION,
            "spec": {"model_kind": spec.model_kind, "seed": spec.seed},
            "source_pair": source_spec(spec.seed).__dict__,
            "best_epoch": best_epoch,
            "best_val_balanced_accuracy": best_val_ba,
            "best_val_loss": best_val_loss,
            "model_state_dict": best_state,
            "channel_scale": data.channel_scale,
            "labels": data.labels,
            "split": data.split,
        },
        destination,
    )
    history_file = history_path(config.results_dir, spec)
    history_file.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(history).to_csv(history_file, index=False)
    return destination


def load_model(
    spec: RunSpec,
    data: exp405.TemporalData,
    repo_root: Path,
    root: Path,
    device: torch.device,
) -> tuple[Stage2ValidationDecoder, dict[str, object], str]:
    model = Stage2ValidationDecoder(
        model_kind=spec.model_kind,
        n_classes=len(data.labels),
        fs=data.fs,
    ).to(device)
    if spec.model_kind == "ff":
        path = checkpoint_path(root, spec)
        if not path.exists():
            raise FileNotFoundError(f"Missing Exp4.3 FF checkpoint: {path}")
        checkpoint = torch.load(path, map_location=device, weights_only=False)
        expected = {"model_kind": spec.model_kind, "seed": spec.seed}
        if checkpoint.get("spec") != expected:
            raise ValueError(f"Checkpoint identity mismatch for {spec.key}")
        source = str(path.relative_to(repo_root))
    else:
        source_root = exp406.results_dir(repo_root)
        source = str(
            exp406.checkpoint_path(source_root, source_spec(spec.seed)).relative_to(
                repo_root
            )
        )
        path = repo_root / source
        if not path.exists():
            raise FileNotFoundError(
                f"Missing frozen Exp4.0.6 shift34 Multi-HO checkpoint: {path}"
            )
        checkpoint = torch.load(path, map_location=device, weights_only=False)
        if checkpoint.get("spec") != source_spec(spec.seed).__dict__:
            raise ValueError(f"Exp4.0.6 checkpoint identity mismatch for seed {spec.seed}")
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, checkpoint, source


def _gather_at_indices(sequence: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    if sequence.ndim != 3:
        raise ValueError(f"Expected [B,T,D], got {tuple(sequence.shape)}")
    if indices.ndim != 1 or indices.shape[0] != sequence.shape[0]:
        raise ValueError("indices must have one entry per batch element")
    if bool((indices < 0).any()) or bool((indices >= sequence.shape[1]).any()):
        raise IndexError("Requested trajectory index outside available timesteps")
    batch_index = torch.arange(sequence.shape[0], device=sequence.device)
    return sequence[batch_index, indices]


def _masked_silent_input(
    X: torch.Tensor,
    valid_steps: torch.Tensor,
    extra_steps: int,
) -> torch.Tensor:
    if extra_steps < 0:
        raise ValueError("extra_steps must be non-negative")
    steps = torch.arange(X.shape[1], device=X.device).view(1, -1, 1)
    valid = valid_steps.view(-1, 1, 1)
    masked = torch.where(steps < valid, X, torch.zeros_like(X))
    if extra_steps == 0:
        return masked
    zeros = torch.zeros(
        X.shape[0], extra_steps, X.shape[2], device=X.device, dtype=X.dtype
    )
    return torch.cat([masked, zeros], dim=1)


def extract_condition_features(
    model: Stage2ValidationDecoder,
    data_loader: torch.utils.data.DataLoader,
    device: torch.device,
    *,
    recurrence_enabled: bool | None = None,
    state_reset_interval_steps: int | None = None,
) -> ConditionFeatures:
    output_parts: list[np.ndarray] = []
    hidden_parts: list[np.ndarray] = []
    endpoint_parts: list[np.ndarray] = []
    label_parts: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for X, y, valid_steps in data_loader:
            X = X.to(device)
            valid_steps = valid_steps.to(device)
            traj = model.forward_trajectory(
                X,
                recurrence_enabled=recurrence_enabled,
                state_reset_interval_steps=state_reset_interval_steps,
            )
            output_logits = exp405.normalized_output_evidence(
                traj["output_spikes"], valid_steps, model.output_cap
            )
            hidden_count = exp405.exp40.valid_whole_count(
                traj["state_spikes"], valid_steps
            )
            endpoint = exp405.exp40.valid_final_membrane(
                traj["state_membranes"], valid_steps
            )
            output_parts.append(output_logits.cpu().numpy())
            hidden_parts.append(hidden_count.cpu().numpy())
            endpoint_parts.append(endpoint.cpu().numpy())
            label_parts.append(y.numpy())
    return ConditionFeatures(
        output_logits=np.concatenate(output_parts),
        hidden_count=np.concatenate(hidden_parts),
        uend=np.concatenate(endpoint_parts),
        labels=np.concatenate(label_parts),
    )


def extract_silent_delay_uend(
    model: Stage2ValidationDecoder,
    data_loader: torch.utils.data.DataLoader,
    device: torch.device,
    delay_steps: int,
) -> tuple[np.ndarray, np.ndarray]:
    endpoint_parts: list[np.ndarray] = []
    label_parts: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for X, y, valid_steps in data_loader:
            X = X.to(device)
            valid_steps = valid_steps.to(device)
            silent_X = _masked_silent_input(X, valid_steps, delay_steps)
            traj = model.forward_trajectory(silent_X)
            endpoint_indices = valid_steps - 1 + int(delay_steps)
            endpoint = _gather_at_indices(
                traj["state_membranes"], endpoint_indices
            )
            endpoint_parts.append(endpoint.cpu().numpy())
            label_parts.append(y.numpy())
    return np.concatenate(endpoint_parts), np.concatenate(label_parts)


def _fit_frozen_probes(
    train_features: ConditionFeatures,
) -> tuple[Pipeline, Pipeline]:
    hidden_probe = _scaled_linear_probe()
    uend_probe = _scaled_linear_probe()
    hidden_probe.fit(train_features.hidden_count, train_features.labels)
    uend_probe.fit(train_features.uend, train_features.labels)
    return hidden_probe, uend_probe


def _condition_metrics(
    features: ConditionFeatures,
    hidden_probe: Pipeline,
    uend_probe: Pipeline,
) -> dict[str, object]:
    return {
        "output_whole_count": exp405.exp40.metrics(
            features.labels, features.output_logits.argmax(axis=1)
        ),
        "hidden_whole_count_linear": exp405.exp40.metrics(
            features.labels, hidden_probe.predict(features.hidden_count)
        ),
        "uend_linear": exp405.exp40.metrics(
            features.labels, uend_probe.predict(features.uend)
        ),
    }


def evaluate_bundle(
    spec: RunSpec,
    data: exp405.TemporalData,
    config: Config,
    force: bool,
) -> dict[str, object]:
    destination = evaluation_path(config.results_dir, spec)
    if destination.exists() and not force:
        return json.loads(destination.read_text(encoding="utf-8"))

    device = torch.device(config.device)
    torch.set_num_threads(config.threads)
    model, checkpoint, source = load_model(
        spec, data, config.repo_root, config.results_dir, device
    )
    loaders = make_loaders(data, spec, config.batch_size, train_shuffle=False)

    normal_features = {
        split: extract_condition_features(model, loader, device)
        for split, loader in loaders.items()
    }
    hidden_probe, uend_probe = _fit_frozen_probes(normal_features["train"])

    classification: dict[str, dict[str, object]] = {
        "normal": {
            split: _condition_metrics(features, hidden_probe, uend_probe)
            for split, features in normal_features.items()
        }
    }

    # V2: same trained RSNN, recurrent contribution removed only at inference.
    if spec.model_kind == "rsnn":
        classification["recurrent_off"] = {
            split: _condition_metrics(
                extract_condition_features(
                    model, loader, device, recurrence_enabled=False
                ),
                hidden_probe,
                uend_probe,
            )
            for split, loader in loaders.items()
        }

    # V3: destroy only Stage-2 state at a controlled periodic horizon.
    for reset_ms, reset_steps in reset_schedule(data.dt_ms).items():
        name = f"reset_{reset_ms}ms"
        classification[name] = {
            split: _condition_metrics(
                extract_condition_features(
                    model,
                    loader,
                    device,
                    state_reset_interval_steps=reset_steps,
                ),
                hidden_probe,
                uend_probe,
            )
            for split, loader in loaders.items()
        }

    # V4: after each sample's true endpoint, force zero input and decode the
    # resulting state using the exact probe fit at delay=0 above.
    retention: dict[str, dict[str, object]] = {}
    for delay_ms, delay_steps in delay_schedule(data.dt_ms).items():
        retention[str(delay_ms)] = {}
        for split, loader in loaders.items():
            if delay_ms == 0:
                endpoints = normal_features[split].uend
                labels = normal_features[split].labels
            else:
                endpoints, labels = extract_silent_delay_uend(
                    model, loader, device, delay_steps
                )
            retention[str(delay_ms)][split] = exp405.exp40.metrics(
                labels, uend_probe.predict(endpoints)
            )

    normal_diagnostics = {
        split: exp405.evaluate_model(model, loader, device)
        for split, loader in loaders.items()
    }
    payload: dict[str, object] = {
        "experiment_id": EXPERIMENT_ID,
        "protocol_version": PROTOCOL_VERSION,
        "spec": {"model_kind": spec.model_kind, "seed": spec.seed},
        "source_checkpoint": source,
        "source_best_epoch": int(checkpoint["best_epoch"]),
        "architecture": {
            "input": "Raw64 30-channel scaled weighted events",
            "local": exp406.local_condition_definition(LOCAL_CONDITION, data.fs),
            "local_recurrence": False,
            "stage2_width": exp405.STATE_WIDTH,
            "stage2_tau_mem_ms": exp405.STATE_TAU_MEM_MS,
            "stage2_recurrence": spec.model_kind == "rsnn",
            "output_tau_mem_ms": exp405.OUTPUT_TAU_MEM_MS,
            "hidden_cap": HIDDEN_CAP,
            "output_cap": OUTPUT_CAP,
            "variant": VARIANT,
            "threshold": exp405.THRESHOLD,
            "dt_ms": data.dt_ms,
        },
        "probe_protocol": {
            "fit_source": "normal train-user representations only",
            "classifier": (
                "train-only StandardScaler + balanced LogisticRegression(lbfgs)"
            ),
            "frozen_for_all_ablations": True,
            "primary_memory_metric": "uend_linear balanced accuracy",
        },
        "classification": classification,
        "silent_delay_retention": retention,
        "normal_diagnostics": normal_diagnostics,
        "interventions": {
            "recurrent_off": (
                "omit W_rec * S[t-1] at inference; RSNN only"
            ),
            "periodic_reset": (
                "zero Stage-2 membrane and previous Stage-2 spikes before each interval; "
                "Local128 and output membrane are not reset"
            ),
            "state_reset_ms": list(STATE_RESET_MS),
            "silent_delay_ms": list(SILENT_DELAY_MS),
            "silent_delay": (
                "zero input after each sample valid endpoint; Local/Stage-2/output dynamics "
                "continue naturally; decode Stage-2 membrane with the frozen delay-0 probe"
            ),
        },
        "provenance": {
            "split_seed": int(exp405.exp40.base.SPLIT_SEED),
            "train_users": data.split["train_users"],
            "val_users": data.split["val_users"],
            "test_users": data.split["test_users"],
            "labels": data.labels,
            "channel_scale_source": (
                "training Fixed250 valid bins only, inherited from Exp4.0.5/4.0.6"
            ),
            "training_objective": "valid normalized output WholeCount CE",
            "rsnn_reused_from_exp406": spec.model_kind == "rsnn",
        },
    }
    _save_json(destination, payload)
    return payload


def run_ff_one(
    spec: RunSpec,
    data: exp405.TemporalData,
    config: Config,
    force: bool,
) -> dict[str, object]:
    train_ff(spec, data, config, force)
    return evaluate_bundle(spec, data, config, force)


def _metric_ba(metrics: dict[str, object], readout: str) -> float:
    return float(metrics[readout]["balanced_accuracy"])  # type: ignore[index]


def _classification_rows(root: Path) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for spec in run_specs():
        path = evaluation_path(root, spec)
        if not path.exists():
            raise FileNotFoundError(f"Missing required Exp4.3 artifact: {path}")
        payload = json.loads(path.read_text(encoding="utf-8"))
        for condition, split_metrics in payload["classification"].items():
            for split, metrics in split_metrics.items():
                rows.append(
                    {
                        "model_kind": spec.model_kind,
                        "seed": spec.seed,
                        "condition": condition,
                        "split": split,
                        "output_whole_count_ba": _metric_ba(
                            metrics, "output_whole_count"
                        ),
                        "hidden_whole_count_linear_ba": _metric_ba(
                            metrics, "hidden_whole_count_linear"
                        ),
                        "uend_linear_ba": _metric_ba(metrics, "uend_linear"),
                    }
                )
    return pd.DataFrame(rows)


def _retention_rows(root: Path) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for spec in run_specs():
        payload = json.loads(
            evaluation_path(root, spec).read_text(encoding="utf-8")
        )
        for delay_ms, split_metrics in payload["silent_delay_retention"].items():
            for split, metrics in split_metrics.items():
                rows.append(
                    {
                        "model_kind": spec.model_kind,
                        "seed": spec.seed,
                        "delay_ms": int(delay_ms),
                        "split": split,
                        "uend_linear_ba": float(metrics["balanced_accuracy"]),
                        "uend_linear_macro_f1": float(metrics["macro_f1"]),
                    }
                )
    return pd.DataFrame(rows)


def _paired_effects(classification: pd.DataFrame, retention: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    test = classification[classification["split"] == "test"].copy()
    indexed = test.set_index(["model_kind", "seed", "condition"])
    readouts = {
        "output_whole_count": "output_whole_count_ba",
        "hidden_whole_count_linear": "hidden_whole_count_linear_ba",
        "uend_linear": "uend_linear_ba",
    }
    for seed in SEEDS:
        for readout, column in readouts.items():
            rsnn = float(indexed.loc[("rsnn", seed, "normal"), column])
            ff = float(indexed.loc[("ff", seed, "normal"), column])
            rows.append(
                {
                    "effect": "rsnn_minus_ff",
                    "model_kind": "",
                    "seed": seed,
                    "condition": "normal",
                    "readout": readout,
                    "delta_ba": rsnn - ff,
                }
            )
            recurrent_off = float(
                indexed.loc[("rsnn", seed, "recurrent_off"), column]
            )
            rows.append(
                {
                    "effect": "rsnn_normal_minus_recurrent_off",
                    "model_kind": "rsnn",
                    "seed": seed,
                    "condition": "recurrent_off",
                    "readout": readout,
                    "delta_ba": rsnn - recurrent_off,
                }
            )
        for model_kind in MODEL_KINDS:
            for reset_ms in STATE_RESET_MS:
                for readout, column in readouts.items():
                    normal = float(indexed.loc[(model_kind, seed, "normal"), column])
                    reset = float(
                        indexed.loc[
                            (model_kind, seed, f"reset_{reset_ms}ms"), column
                        ]
                    )
                    rows.append(
                        {
                            "effect": "normal_minus_periodic_reset",
                            "model_kind": model_kind,
                            "seed": seed,
                            "condition": f"reset_{reset_ms}ms",
                            "readout": readout,
                            "delta_ba": normal - reset,
                        }
                    )

    ret_test = retention[retention["split"] == "test"].set_index(
        ["model_kind", "seed", "delay_ms"]
    )
    for model_kind in MODEL_KINDS:
        for seed in SEEDS:
            ba0 = float(ret_test.loc[(model_kind, seed, 0), "uend_linear_ba"])
            for delay_ms in SILENT_DELAY_MS:
                if delay_ms == 0:
                    continue
                delayed = float(
                    ret_test.loc[(model_kind, seed, delay_ms), "uend_linear_ba"]
                )
                rows.append(
                    {
                        "effect": "silent_delay_drop_from_0ms",
                        "model_kind": model_kind,
                        "seed": seed,
                        "condition": f"delay_{delay_ms}ms",
                        "readout": "uend_linear",
                        "delta_ba": ba0 - delayed,
                    }
                )
    return pd.DataFrame(rows)


def finalize_experiment(repo_root: Path) -> dict[str, Path]:
    root = results_dir(repo_root)
    root.mkdir(parents=True, exist_ok=True)
    classification = _classification_rows(root)
    retention = _retention_rows(root)

    classification_file = root / "classification_runs.csv"
    retention_file = root / "retention_runs.csv"
    classification.to_csv(classification_file, index=False)
    retention.to_csv(retention_file, index=False)

    test = classification[classification["split"] == "test"]
    classification_summary = (
        test.groupby(["model_kind", "condition"])[
            [
                "output_whole_count_ba",
                "hidden_whole_count_linear_ba",
                "uend_linear_ba",
            ]
        ]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    classification_summary.columns = [
        "_".join(str(part) for part in col if str(part))
        if isinstance(col, tuple)
        else str(col)
        for col in classification_summary.columns
    ]
    classification_summary_file = root / "classification_summary.csv"
    classification_summary.to_csv(classification_summary_file, index=False)

    retention_test = retention[retention["split"] == "test"]
    retention_summary = (
        retention_test.groupby(["model_kind", "delay_ms"])[
            ["uend_linear_ba", "uend_linear_macro_f1"]
        ]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    retention_summary.columns = [
        "_".join(str(part) for part in col if str(part))
        if isinstance(col, tuple)
        else str(col)
        for col in retention_summary.columns
    ]
    retention_summary_file = root / "retention_summary.csv"
    retention_summary.to_csv(retention_summary_file, index=False)

    effects = _paired_effects(classification, retention)
    effects_file = root / "paired_effects.csv"
    effects.to_csv(effects_file, index=False)
    effects_summary = (
        effects.groupby(
            ["effect", "model_kind", "condition", "readout"], dropna=False
        )["delta_ba"]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    effects_summary_file = root / "paired_effects_summary.csv"
    effects_summary.to_csv(effects_summary_file, index=False)

    manifest = {
        "experiment_id": EXPERIMENT_ID,
        "protocol_version": PROTOCOL_VERSION,
        "question": (
            "Does the current Stage-2 state retain information beyond the local ~250-ms "
            "window, and is that memory specifically supported by recurrent feedback?"
        ),
        "new_training_runs": len(SEEDS),
        "frozen_rsnn_evaluations": len(SEEDS),
        "seeds": list(SEEDS),
        "fixed_architecture": {
            "local_condition": LOCAL_CONDITION,
            "stage2_tau_mem_ms": exp405.STATE_TAU_MEM_MS,
            "stage2_width": exp405.STATE_WIDTH,
            "variant": VARIANT,
            "hidden_cap": HIDDEN_CAP,
            "output_cap": OUTPUT_CAP,
        },
        "validations": {
            "v1": "independently trained FF vs frozen Exp4.0.6 RSNN",
            "v2": "same trained RSNN with recurrence disabled only at inference",
            "v3": f"periodic Stage-2-only resets at {list(STATE_RESET_MS)} ms",
            "v4": f"silent-delay Uend retention at {list(SILENT_DELAY_MS)} ms",
        },
        "primary_memory_metric": (
            "Uend + frozen train-only StandardScaler/LogisticRegression probe"
        ),
        "whole_count_note": (
            "Output WholeCount is secondary for reset tests because the accumulator itself "
            "retains earlier evidence."
        ),
        "files": {
            "classification_runs": classification_file.name,
            "classification_summary": classification_summary_file.name,
            "retention_runs": retention_file.name,
            "retention_summary": retention_summary_file.name,
            "paired_effects": effects_file.name,
            "paired_effects_summary": effects_summary_file.name,
        },
    }
    manifest_file = root / "manifest.json"
    _save_json(manifest_file, manifest)
    return {
        "classification_runs": classification_file,
        "classification_summary": classification_summary_file,
        "retention_runs": retention_file,
        "retention_summary": retention_summary_file,
        "paired_effects": effects_file,
        "paired_effects_summary": effects_summary_file,
        "manifest": manifest_file,
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    ff_parser = subparsers.add_parser("run-ff-one")
    ff_parser.add_argument("--array-task-id", type=int, required=True)
    ff_parser.add_argument("--device", default="cpu")
    ff_parser.add_argument("--threads", type=int, default=1)
    ff_parser.add_argument("--epochs", type=int, default=EPOCHS)
    ff_parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    ff_parser.add_argument("--force", action="store_true")

    rsnn_parser = subparsers.add_parser("eval-rsnn-one")
    rsnn_parser.add_argument("--array-task-id", type=int, required=True)
    rsnn_parser.add_argument("--device", default="cpu")
    rsnn_parser.add_argument("--threads", type=int, default=1)
    rsnn_parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    rsnn_parser.add_argument("--force", action="store_true")

    eval_parser = subparsers.add_parser("eval-one")
    eval_parser.add_argument("--model-kind", choices=MODEL_KINDS, required=True)
    eval_parser.add_argument("--seed", type=int, choices=SEEDS, required=True)
    eval_parser.add_argument("--device", default="cpu")
    eval_parser.add_argument("--threads", type=int, default=1)
    eval_parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    eval_parser.add_argument("--force", action="store_true")

    subparsers.add_parser("finalize")
    return parser


def _config_from_args(args: argparse.Namespace, repo_root: Path) -> Config:
    return Config(
        repo_root=repo_root,
        results_dir=results_dir(repo_root),
        device=getattr(args, "device", "cpu"),
        epochs=getattr(args, "epochs", EPOCHS),
        batch_size=getattr(args, "batch_size", BATCH_SIZE),
        threads=getattr(args, "threads", 1),
    )


def main() -> None:
    args = build_arg_parser().parse_args()
    repo_root = exp405.exp40.find_repo_root()
    if args.command == "finalize":
        for name, path in finalize_experiment(repo_root).items():
            print(f"{name}: {path}")
        return

    data = prepare_data(repo_root)
    config = _config_from_args(args, repo_root)
    if args.command == "run-ff-one":
        specs = run_specs("ff")
        if args.array_task_id < 0 or args.array_task_id >= len(specs):
            raise IndexError(
                f"array task {args.array_task_id} outside [0, {len(specs) - 1}]"
            )
        spec = specs[args.array_task_id]
        payload = run_ff_one(spec, data, config, args.force)
    elif args.command == "eval-rsnn-one":
        specs = run_specs("rsnn")
        if args.array_task_id < 0 or args.array_task_id >= len(specs):
            raise IndexError(
                f"array task {args.array_task_id} outside [0, {len(specs) - 1}]"
            )
        spec = specs[args.array_task_id]
        payload = evaluate_bundle(spec, data, config, args.force)
    else:
        spec = RunSpec(args.model_kind, args.seed)
        payload = evaluate_bundle(spec, data, config, args.force)

    test = payload["classification"]["normal"]["test"]
    print(
        f"completed {spec.key}: "
        f"output_count_BA={test['output_whole_count']['balanced_accuracy']:.6f} "
        f"hidden_count_linear_BA={test['hidden_whole_count_linear']['balanced_accuracy']:.6f} "
        f"uend_linear_BA={test['uend_linear']['balanced_accuracy']:.6f}"
    )


if __name__ == "__main__":
    main()
