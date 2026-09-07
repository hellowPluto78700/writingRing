from __future__ import annotations

import argparse
import copy
from dataclasses import asdict, dataclass
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

from scripts import experiment_5_3_2_when_representation_runtime as parent


base = parent.base
exp52 = parent.exp52

EXPERIMENT_ID = "experiment_5_3_2_1_when_objective_sweep"
PROTOCOL_VERSION = "rsnn_when_objective_v1"
SEEDS = parent.SEEDS
WHEN_CONDITION = "rsnn_shortmem"
EPS = 1e-12


@dataclass(frozen=True)
class Objective:
    name: str
    phase_weight: float
    progress_weight: float

    @property
    def formula(self) -> str:
        if self.phase_weight == 1.0 and self.progress_weight == 0.0:
            return "L_phase"
        if self.phase_weight == 0.0 and self.progress_weight == 1.0:
            return "L_progress"
        return f"{self.phase_weight:g} * L_phase + {self.progress_weight:g} * L_progress"


OBJECTIVES = (
    Objective("phase_only", 1.0, 0.0),
    Objective("phase_heavy", 5.0, 1.0),
    Objective("joint", 1.0, 1.0),
    Objective("progress_heavy", 0.2, 1.0),
    Objective("progress_only", 0.0, 1.0),
)
OBJECTIVE_BY_NAME = {objective.name: objective for objective in OBJECTIVES}
OBJECTIVE_NAMES = tuple(objective.name for objective in OBJECTIVES)


@dataclass(frozen=True)
class RunSpec:
    objective: str
    seed: int

    @property
    def key(self) -> str:
        return f"{self.objective}__seed{self.seed}"

    @property
    def definition(self) -> Objective:
        try:
            return OBJECTIVE_BY_NAME[self.objective]
        except KeyError as error:
            raise ValueError(f"Unknown Exp5.3.2.1 objective: {self.objective}") from error


@dataclass(frozen=True)
class Config:
    repo_root: Path
    results_dir: Path
    device: str = "cpu"
    epochs: int = parent.EPOCHS
    batch_size: int = parent.BATCH_SIZE
    threads: int = 1


def find_repo_root(start: Path | None = None) -> Path:
    return parent.find_repo_root(start)


def results_dir(repo_root: Path) -> Path:
    return (
        repo_root
        / "notebooks"
        / "artifacts"
        / EXPERIMENT_ID
        / PROTOCOL_VERSION
    )


def checkpoint_path(root: Path, spec: RunSpec) -> Path:
    return root / "checkpoints" / f"{spec.key}.pt"


def evaluation_path(root: Path, spec: RunSpec) -> Path:
    return root / "evaluations" / f"{spec.key}.json"


def history_path(root: Path, spec: RunSpec) -> Path:
    return root / "histories" / f"{spec.key}.csv"


def baseline_path(root: Path, seed: int) -> Path:
    return root / "baselines" / f"seed{seed}.json"


def _save_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def run_specs() -> list[RunSpec]:
    return [RunSpec(objective.name, seed) for objective in OBJECTIVES for seed in SEEDS]


def _parent_config(config: Config) -> parent.Config:
    return parent.Config(
        repo_root=config.repo_root,
        results_dir=parent.results_dir(config.repo_root),
        device=config.device,
        epochs=config.epochs,
        batch_size=config.batch_size,
        threads=config.threads,
    )


def _parent_spec(seed: int) -> parent.RunSpec:
    return parent.RunSpec(WHEN_CONDITION, seed)


def _objective_loss(
    objective: Objective,
    phase_loss: torch.Tensor,
    progress_loss: torch.Tensor,
) -> torch.Tensor:
    return (
        objective.phase_weight * phase_loss
        + objective.progress_weight * progress_loss
    )


def _shared_rsnn_parameters(model: parent.WhenBranchNet) -> list[torch.nn.Parameter]:
    parameters = list(model.input_projection.parameters())
    if model.recurrent is not None:
        parameters.extend(model.recurrent.parameters())
    return [parameter for parameter in parameters if parameter.requires_grad]


def _gradient_norm(
    loss: torch.Tensor,
    parameters: list[torch.nn.Parameter],
    retain_graph: bool,
) -> float:
    gradients = torch.autograd.grad(
        loss,
        parameters,
        retain_graph=retain_graph,
        allow_unused=True,
    )
    total = torch.zeros((), device=loss.device)
    for gradient in gradients:
        if gradient is not None:
            total = total + gradient.detach().pow(2).sum()
    return float(torch.sqrt(total).item())


def gradient_diagnostics(
    model: parent.WhenBranchNet,
    local: torch.Tensor,
    lengths: torch.Tensor,
    objective: Objective,
) -> dict[str, float]:
    model.eval()
    _, phase_loss, progress_loss, _, _, _ = model.loss_components(local, lengths)
    parameters = _shared_rsnn_parameters(model)
    phase_norm = _gradient_norm(phase_loss, parameters, retain_graph=True)
    progress_norm = _gradient_norm(progress_loss, parameters, retain_graph=False)
    weighted_phase = abs(objective.phase_weight) * phase_norm
    weighted_progress = abs(objective.progress_weight) * progress_norm
    weighted_total = weighted_phase + weighted_progress
    return {
        "raw_phase_grad_norm": phase_norm,
        "raw_progress_grad_norm": progress_norm,
        "weighted_phase_grad_norm": weighted_phase,
        "weighted_progress_grad_norm": weighted_progress,
        "weighted_phase_grad_fraction": weighted_phase / (weighted_total + EPS),
        "weighted_progress_grad_fraction": weighted_progress / (weighted_total + EPS),
        "weighted_phase_to_progress_grad_ratio": weighted_phase
        / (weighted_progress + EPS),
    }


def _evaluate_native_objective(
    model: parent.WhenBranchNet,
    loader: DataLoader,
    device: torch.device,
    objective: Objective,
    reset_state_each_step: bool = False,
) -> dict[str, float]:
    metrics = parent.evaluate_native(
        model,
        loader,
        device,
        reset_state_each_step=reset_state_each_step,
    )
    metrics = dict(metrics)
    metrics["objective_loss"] = (
        objective.phase_weight * float(metrics["phase_ce"])
        + objective.progress_weight * float(metrics["progress_loss"])
    )
    return metrics


def _rankdata_average(values: np.ndarray) -> np.ndarray:
    return pd.Series(np.asarray(values, dtype=float)).rank(
        method="average"
    ).to_numpy(dtype=float)


def _spearman_against_time(prediction: np.ndarray) -> float:
    prediction = np.asarray(prediction, dtype=float)
    if prediction.size < 2:
        return 1.0
    ranks = _rankdata_average(prediction)
    if float(np.std(ranks)) <= EPS:
        return 0.0
    time = np.arange(prediction.size, dtype=float)
    return float(np.corrcoef(time, ranks)[0, 1])


def _monotonic_violation_rate(prediction: np.ndarray) -> float:
    prediction = np.asarray(prediction, dtype=float)
    if prediction.size < 2:
        return 0.0
    return float(np.mean(np.diff(prediction) < -1e-8))


def _true_progress(length: int) -> np.ndarray:
    if length <= 0:
        raise ValueError("Progress sequence length must be positive")
    if length == 1:
        return np.zeros(1, dtype=np.float32)
    return np.arange(length, dtype=np.float32) / float(length - 1)


def _trajectory_metrics(
    sequences: list[np.ndarray],
    scaler: StandardScaler,
    regressor: Ridge,
) -> dict[str, float | int]:
    sample_mae: list[float] = []
    sample_spearman: list[float] = []
    sample_violation: list[float] = []
    for sequence in sequences:
        if sequence.ndim != 2 or sequence.shape[0] <= 0:
            raise ValueError("Expected non-empty [time, feature] sequence")
        prediction = np.clip(
            regressor.predict(scaler.transform(sequence)),
            0.0,
            1.0,
        )
        truth = _true_progress(len(sequence))
        sample_mae.append(float(mean_absolute_error(truth, prediction)))
        sample_spearman.append(_spearman_against_time(prediction))
        sample_violation.append(_monotonic_violation_rate(prediction))
    return {
        "n_gestures": len(sequences),
        "sample_balanced_mae_mean": float(np.mean(sample_mae)),
        "sample_balanced_mae_sd": float(np.std(sample_mae, ddof=1))
        if len(sample_mae) > 1
        else 0.0,
        "spearman_mean": float(np.mean(sample_spearman)),
        "spearman_sd": float(np.std(sample_spearman, ddof=1))
        if len(sample_spearman) > 1
        else 0.0,
        "monotonic_violation_rate_mean": float(np.mean(sample_violation)),
        "monotonic_violation_rate_sd": float(np.std(sample_violation, ddof=1))
        if len(sample_violation) > 1
        else 0.0,
    }


def _fit_progress_probe_model(
    features: dict[str, dict[str, np.ndarray]],
    feature_name: str,
) -> tuple[StandardScaler, Ridge, dict[str, float]]:
    train_x = features["train"][feature_name]
    val_x = features["val"][feature_name]
    test_x = features["test"][feature_name]
    train_y = features["train"]["progress"]
    val_y = features["val"]["progress"]
    test_y = features["test"]["progress"]

    scaler = StandardScaler().fit(train_x)
    train_z = scaler.transform(train_x)
    val_z = scaler.transform(val_x)
    test_z = scaler.transform(test_x)

    best: tuple[float, float, Ridge] | None = None
    for alpha in parent.PROGRESS_ALPHA_GRID:
        regressor = Ridge(alpha=alpha).fit(train_z, train_y)
        val_prediction = np.clip(regressor.predict(val_z), 0.0, 1.0)
        val_mae = float(mean_absolute_error(val_y, val_prediction))
        if best is None or val_mae < best[0] - 1e-12:
            best = (val_mae, float(alpha), regressor)
    if best is None:
        raise RuntimeError("No progress probe candidate selected")
    val_mae, alpha, regressor = best
    test_prediction = np.clip(regressor.predict(test_z), 0.0, 1.0)
    return scaler, regressor, {
        "progress_probe_alpha": alpha,
        "progress_probe_val_mae": val_mae,
        "progress_probe_test_mae": float(
            mean_absolute_error(test_y, test_prediction)
        ),
    }


def _collect_membrane_sequences(
    model: parent.WhenBranchNet,
    data: base.Data,
    cache: dict[str, np.ndarray],
    seed: int,
    config: Config,
    reset_state_each_step: bool,
) -> dict[str, list[np.ndarray]]:
    pspec = _parent_spec(seed)
    pconfig = _parent_config(config)
    loaders = parent._make_loaders(
        data,
        cache,
        pspec,
        pconfig,
        train_shuffle=False,
    )
    device = torch.device(config.device)
    output: dict[str, list[np.ndarray]] = {}
    model.eval()
    with torch.no_grad():
        for split, loader in loaders.items():
            sequences: list[np.ndarray] = []
            for local, lengths in loader:
                local = local.to(device)
                lengths = lengths.to(device)
                trajectory = model.forward_trajectory(
                    local,
                    lengths,
                    reset_state_each_step=reset_state_each_step,
                )
                for sample_index, raw_length in enumerate(lengths.tolist()):
                    length = int(raw_length)
                    sequences.append(
                        trajectory.membranes[
                            sample_index, :length
                        ].detach().cpu().numpy()
                    )
            output[split] = sequences
    return output


def _augment_membrane_probe_with_trajectory_metrics(
    probe: dict[str, object],
    features: dict[str, dict[str, np.ndarray]],
    sequences: dict[str, list[np.ndarray]],
) -> dict[str, object]:
    scaler, regressor, model_metrics = _fit_progress_probe_model(
        features,
        "membrane",
    )
    output = dict(probe)
    output.update(model_metrics)
    for split in ("val", "test"):
        metrics = _trajectory_metrics(sequences[split], scaler, regressor)
        for name, value in metrics.items():
            output[f"progress_probe_{split}_{name}"] = value
    return output


def _baseline_sequences(
    X: np.ndarray,
    lengths: np.ndarray,
    fs: float,
    feature_name: str,
) -> list[np.ndarray]:
    sequences: list[np.ndarray] = []
    for sample_index, raw_length in enumerate(lengths):
        length = int(raw_length)
        if feature_name == "what":
            sequences.append(
                np.asarray(X[sample_index, :length], dtype=np.float32)
            )
        elif feature_name == "elapsed":
            sequences.append(
                (
                    np.arange(length, dtype=np.float32) / float(fs)
                )[:, None]
            )
        else:
            raise ValueError(f"Unknown baseline feature: {feature_name}")
    return sequences


def prepare_local_seed(
    seed: int,
    data: base.Data,
    config: Config,
    force: bool = False,
) -> Path:
    if seed not in SEEDS:
        raise ValueError(f"Unknown Exp5.3.2.1 seed: {seed}")
    pconfig = _parent_config(config)
    parent.prepare_local_seed(seed, data, pconfig, force=False)
    cache = parent.load_local_cache(seed, data, pconfig)

    destination = baseline_path(config.results_dir, seed)
    if force or not destination.exists():
        flat = {
            split: parent._baseline_arrays(X, lengths, data.fs)
            for split, (X, lengths) in parent._partitions(data, cache).items()
        }
        baseline_entries: list[dict[str, object]] = []
        for feature_name, baseline_name in (
            ("what", "what_only"),
            ("elapsed", "elapsed_time_only"),
        ):
            probe = parent._fit_feature_probe(
                flat,
                feature_name,
                seed,
                "exp5_3_2_1_baseline",
            )
            scaler, regressor, model_metrics = _fit_progress_probe_model(
                flat,
                feature_name,
            )
            entry = {
                "baseline": baseline_name,
                **probe,
                **model_metrics,
            }
            for split, (X, lengths) in parent._partitions(data, cache).items():
                if split not in ("val", "test"):
                    continue
                sequences = _baseline_sequences(
                    X,
                    lengths,
                    data.fs,
                    feature_name,
                )
                trajectory = _trajectory_metrics(sequences, scaler, regressor)
                for name, value in trajectory.items():
                    entry[f"progress_probe_{split}_{name}"] = value
            baseline_entries.append(entry)
        _save_json(
            destination,
            {
                "experiment_id": EXPERIMENT_ID,
                "protocol_version": PROTOCOL_VERSION,
                "seed": seed,
                "sampling_rate_hz": float(data.fs),
                "elapsed_time_input": (
                    "t / fs only; final duration T is never an input"
                ),
                "what_input": "frozen Local-SNN L2 spike vector z_t only",
                "baselines": baseline_entries,
            },
        )
    return exp52.local_cache_path(parent.source_results_dir(config.repo_root), seed)


def _initialize_model(
    seed: int,
    fs: float,
    device: torch.device,
) -> parent.WhenBranchNet:
    return parent._initialize_model(_parent_spec(seed), fs, device)


def _provenance(
    spec: RunSpec,
    data: base.Data,
    config: Config,
    model: parent.WhenBranchNet,
) -> dict[str, object]:
    pconfig = _parent_config(config)
    provenance = dict(
        parent.provenance(
            _parent_spec(spec.seed),
            data,
            pconfig,
            model,
        )
    )
    objective = spec.definition
    provenance.update(
        {
            "experiment_id": EXPERIMENT_ID,
            "protocol_version": PROTOCOL_VERSION,
            "source_when_experiment": parent.EXPERIMENT_ID,
            "source_when_protocol": parent.PROTOCOL_VERSION,
            "fixed_when_architecture": WHEN_CONDITION,
            "objective": objective.name,
            "objective_formula": objective.formula,
            "phase_weight": objective.phase_weight,
            "progress_weight": objective.progress_weight,
            "phase_head_trained": objective.phase_weight > 0.0,
            "progress_head_trained": objective.progress_weight > 0.0,
            "checkpoint_selection": (
                "min validation weighted objective loss; "
                "tie lower validation progress MAE; "
                "tie higher validation phase BA"
            ),
            "primary_cross_objective_readout": (
                "common post-hoc probes on frozen membrane U_t; "
                "native heads are not comparable when an objective weight is zero"
            ),
            "gradient_diagnostic": (
                "one fixed train diagnostic batch per epoch; "
                "L2 gradient norms measured on input_projection + recurrent weights"
            ),
        }
    )
    return provenance


def train_one(
    spec: RunSpec,
    data: base.Data,
    config: Config,
    force: bool = False,
) -> Path:
    destination = checkpoint_path(config.results_dir, spec)
    if destination.exists() and not force:
        return destination

    pconfig = _parent_config(config)
    pspec = _parent_spec(spec.seed)
    cache = parent.load_local_cache(spec.seed, data, pconfig)
    train_loaders = parent._make_loaders(
        data,
        cache,
        pspec,
        pconfig,
        train_shuffle=True,
    )
    eval_loaders = parent._make_loaders(
        data,
        cache,
        pspec,
        pconfig,
        train_shuffle=False,
    )
    diagnostic_local, diagnostic_lengths = next(iter(eval_loaders["train"]))
    device = torch.device(config.device)
    diagnostic_local = diagnostic_local.to(device)
    diagnostic_lengths = diagnostic_lengths.to(device)

    torch.set_num_threads(config.threads)
    model = _initialize_model(spec.seed, data.fs, device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=parent.LR,
        weight_decay=parent.WEIGHT_DECAY,
    )
    objective = spec.definition

    best_val_objective = np.inf
    best_val_progress_mae = np.inf
    best_val_phase_ba = -np.inf
    best_epoch = -1
    best_state: dict[str, torch.Tensor] | None = None
    history: list[dict[str, float | int]] = []

    for epoch in range(1, config.epochs + 1):
        model.train()
        train_objective_sum = 0.0
        train_phase_sum = 0.0
        train_progress_sum = 0.0
        train_n = 0

        for local, lengths in train_loaders["train"]:
            local = local.to(device)
            lengths = lengths.to(device)
            optimizer.zero_grad(set_to_none=True)
            _, phase_loss, progress_loss, _, _, _ = model.loss_components(
                local,
                lengths,
            )
            objective_loss = _objective_loss(
                objective,
                phase_loss,
                progress_loss,
            )
            objective_loss.backward()
            optimizer.step()

            n = len(local)
            train_n += n
            train_objective_sum += float(objective_loss.item()) * n
            train_phase_sum += float(phase_loss.item()) * n
            train_progress_sum += float(progress_loss.item()) * n

        val_metrics = _evaluate_native_objective(
            model,
            eval_loaders["val"],
            device,
            objective,
        )
        gradients = gradient_diagnostics(
            model,
            diagnostic_local,
            diagnostic_lengths,
            objective,
        )
        val_objective = float(val_metrics["objective_loss"])
        val_progress_mae = float(
            val_metrics["progress_sample_balanced_mae"]
        )
        val_phase_ba = float(val_metrics["phase_balanced_accuracy"])
        history.append(
            {
                "epoch": epoch,
                "train_objective_loss": train_objective_sum
                / max(train_n, 1),
                "train_phase_ce": train_phase_sum / max(train_n, 1),
                "train_progress_loss": train_progress_sum
                / max(train_n, 1),
                "val_objective_loss": val_objective,
                "val_phase_ce": val_metrics["phase_ce"],
                "val_progress_loss": val_metrics["progress_loss"],
                "val_phase_balanced_accuracy": val_phase_ba,
                "val_progress_sample_balanced_mae": val_progress_mae,
                **gradients,
            }
        )

        improved = (
            val_objective < best_val_objective - 1e-12
            or (
                abs(val_objective - best_val_objective) <= 1e-12
                and val_progress_mae < best_val_progress_mae - 1e-12
            )
            or (
                abs(val_objective - best_val_objective) <= 1e-12
                and abs(val_progress_mae - best_val_progress_mae) <= 1e-12
                and val_phase_ba > best_val_phase_ba + 1e-12
            )
        )
        if improved:
            best_val_objective = val_objective
            best_val_progress_mae = val_progress_mae
            best_val_phase_ba = val_phase_ba
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())

    if best_state is None:
        raise RuntimeError(f"No checkpoint selected for {spec.key}")

    model.load_state_dict(best_state)
    native = {
        split: _evaluate_native_objective(
            model,
            loader,
            device,
            objective,
        )
        for split, loader in eval_loaders.items()
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "experiment_id": EXPERIMENT_ID,
            "protocol_version": PROTOCOL_VERSION,
            "spec": asdict(spec),
            "provenance": _provenance(spec, data, config, model),
            "result": {
                "best_epoch": best_epoch,
                "best_val_objective_loss": best_val_objective,
                "best_val_progress_mae": best_val_progress_mae,
                "best_val_phase_ba": best_val_phase_ba,
                "native": native,
            },
            "state_dict": best_state,
        },
        destination,
    )
    hpath = history_path(config.results_dir, spec)
    hpath.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(history).to_csv(hpath, index=False)
    return destination


def load_model(
    spec: RunSpec,
    data: base.Data,
    config: Config,
) -> tuple[parent.WhenBranchNet, dict[str, object]]:
    path = checkpoint_path(config.results_dir, spec)
    if not path.exists():
        raise FileNotFoundError(f"Missing Exp5.3.2.1 checkpoint: {path}")
    payload = torch.load(
        path,
        map_location=config.device,
        weights_only=False,
    )
    if (
        payload.get("experiment_id") != EXPERIMENT_ID
        or payload.get("protocol_version") != PROTOCOL_VERSION
    ):
        raise ValueError(f"Wrong Exp5.3.2.1 checkpoint identity: {path}")
    if payload.get("spec") != asdict(spec):
        raise ValueError(f"Checkpoint spec mismatch: {path}")
    model = parent.WhenBranchNet(WHEN_CONDITION, data.fs).to(
        torch.device(config.device)
    )
    model.load_state_dict(payload["state_dict"], strict=True)
    model.eval()
    return model, payload


def _main_probe_rows(
    features: dict[str, dict[str, np.ndarray]],
    sequences: dict[str, list[np.ndarray]],
    seed: int,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for feature_name in (
        "membrane",
        "synaptic",
        "spike",
        "spike250",
        "spike500",
    ):
        probe = parent._fit_feature_probe(
            features,
            feature_name,
            seed,
            "exp5_3_2_1_common_probe",
        )
        if feature_name == "membrane":
            probe = _augment_membrane_probe_with_trajectory_metrics(
                probe,
                features,
                sequences,
            )
        rows.append(probe)
    return rows


def evaluate_one(
    spec: RunSpec,
    data: base.Data,
    config: Config,
    force: bool = False,
) -> dict[str, object]:
    destination = evaluation_path(config.results_dir, spec)
    if destination.exists() and not force:
        return json.loads(destination.read_text(encoding="utf-8"))

    pconfig = _parent_config(config)
    pspec = _parent_spec(spec.seed)
    cache = parent.load_local_cache(spec.seed, data, pconfig)
    model, checkpoint = load_model(spec, data, config)
    objective = spec.definition
    device = torch.device(config.device)
    loaders = parent._make_loaders(
        data,
        cache,
        pspec,
        pconfig,
        train_shuffle=False,
    )

    native = {
        split: _evaluate_native_objective(
            model,
            loader,
            device,
            objective,
        )
        for split, loader in loaders.items()
    }
    ordered_features = parent.collect_features(
        model,
        data,
        cache,
        pspec,
        pconfig,
        reset_state_each_step=False,
    )
    ordered_sequences = _collect_membrane_sequences(
        model,
        data,
        cache,
        spec.seed,
        config,
        reset_state_each_step=False,
    )
    probes = _main_probe_rows(
        ordered_features,
        ordered_sequences,
        spec.seed,
    )
    ordered_membrane = next(
        probe for probe in probes if probe["feature_type"] == "membrane"
    )

    reset_features = parent.collect_features(
        model,
        data,
        cache,
        pspec,
        pconfig,
        reset_state_each_step=True,
    )
    reset_sequences = _collect_membrane_sequences(
        model,
        data,
        cache,
        spec.seed,
        config,
        reset_state_each_step=True,
    )
    reset_probe = parent._fit_feature_probe(
        reset_features,
        "membrane",
        spec.seed,
        "exp5_3_2_1_state_reset",
    )
    reset_probe = _augment_membrane_probe_with_trajectory_metrics(
        reset_probe,
        reset_features,
        reset_sequences,
    )
    reset_native = {
        split: _evaluate_native_objective(
            model,
            loader,
            device,
            objective,
            reset_state_each_step=True,
        )
        for split, loader in loaders.items()
    }

    ablations: list[dict[str, object]] = [
        {
            "ablation": "state_reset",
            "replicate": 0,
            "native": reset_native,
            "membrane_probe": reset_probe,
        }
    ]

    for replicate in range(parent.SHUFFLE_REPLICATES):
        shuffled_cache = parent._shuffle_cache(
            data,
            cache,
            spec.seed,
            replicate,
        )
        shuffled_loaders = parent._make_loaders(
            data,
            shuffled_cache,
            pspec,
            pconfig,
            train_shuffle=False,
        )
        shuffled_native = {
            split: _evaluate_native_objective(
                model,
                loader,
                device,
                objective,
            )
            for split, loader in shuffled_loaders.items()
        }
        shuffled_features = parent.collect_features(
            model,
            data,
            shuffled_cache,
            pspec,
            pconfig,
            reset_state_each_step=False,
        )
        shuffled_sequences = _collect_membrane_sequences(
            model,
            data,
            shuffled_cache,
            spec.seed,
            config,
            reset_state_each_step=False,
        )
        shuffled_probe = parent._fit_feature_probe(
            shuffled_features,
            "membrane",
            spec.seed,
            "exp5_3_2_1_temporal_shuffle",
        )
        shuffled_probe = _augment_membrane_probe_with_trajectory_metrics(
            shuffled_probe,
            shuffled_features,
            shuffled_sequences,
        )
        ablations.append(
            {
                "ablation": "temporal_shuffle",
                "replicate": replicate,
                "native": shuffled_native,
                "membrane_probe": shuffled_probe,
            }
        )

    payload = {
        "experiment_id": EXPERIMENT_ID,
        "protocol_version": PROTOCOL_VERSION,
        "spec": asdict(spec),
        "objective": asdict(objective),
        "provenance": checkpoint["provenance"],
        "best_epoch": checkpoint["result"]["best_epoch"],
        "native": native,
        "probes": probes,
        "ordered_membrane_probe": ordered_membrane,
        "ablations": ablations,
    }
    _save_json(destination, payload)
    return payload


def run_one(
    spec: RunSpec,
    data: base.Data,
    config: Config,
    force: bool = False,
) -> dict[str, object]:
    train_one(spec, data, config, force=force)
    return evaluate_one(spec, data, config, force=force)


def _probe_trajectory_columns(probe: dict[str, object], split: str) -> dict[str, object]:
    prefix = f"progress_probe_{split}_"
    names = (
        "sample_balanced_mae_mean",
        "sample_balanced_mae_sd",
        "spearman_mean",
        "spearman_sd",
        "monotonic_violation_rate_mean",
        "monotonic_violation_rate_sd",
    )
    return {
        f"probe_{split}_{name}": probe.get(f"{prefix}{name}", np.nan)
        for name in names
    }


def finalize_experiment(repo_root: Path) -> dict[str, Path]:
    root = results_dir(repo_root)
    run_rows: list[dict[str, object]] = []
    probe_rows: list[dict[str, object]] = []
    ablation_rows: list[dict[str, object]] = []
    history_gain_rows: list[dict[str, object]] = []
    baseline_rows: list[dict[str, object]] = []
    local_rows: list[dict[str, object]] = []
    gradient_rows: list[dict[str, object]] = []
    history_parts: list[pd.DataFrame] = []

    for spec in run_specs():
        epath = evaluation_path(root, spec)
        if not epath.exists():
            raise FileNotFoundError(
                f"Missing Exp5.3.2.1 evaluation: {epath}"
            )
        payload = json.loads(epath.read_text(encoding="utf-8"))
        if (
            payload.get("experiment_id") != EXPERIMENT_ID
            or payload.get("protocol_version") != PROTOCOL_VERSION
            or payload.get("spec") != asdict(spec)
        ):
            raise ValueError(f"Evaluation identity mismatch: {epath}")

        objective = spec.definition
        common = {
            "objective": objective.name,
            "phase_weight": objective.phase_weight,
            "progress_weight": objective.progress_weight,
            "seed": spec.seed,
            "fixed_when_architecture": WHEN_CONDITION,
        }
        row: dict[str, object] = {
            **common,
            "best_epoch": payload["best_epoch"],
            "phase_head_trained": objective.phase_weight > 0.0,
            "progress_head_trained": objective.progress_weight > 0.0,
        }
        for split in ("train", "val", "test"):
            for metric, value in payload["native"][split].items():
                row[f"native_{split}_{metric}"] = value
        ordered_probe = payload["ordered_membrane_probe"]
        row.update(
            {
                "probe_u_test_phase_ba": ordered_probe["phase_probe_test_ba"],
                "probe_u_test_progress_mae": ordered_probe["progress_probe_test_mae"],
                **_probe_trajectory_columns(ordered_probe, "test"),
            }
        )
        run_rows.append(row)

        for probe in payload["probes"]:
            probe_rows.append({**common, **probe})

        def ablation_row(
            ablation: str,
            replicate: int,
            native_metrics: dict[str, dict[str, float]],
            probe: dict[str, object],
        ) -> dict[str, object]:
            return {
                **common,
                "ablation": ablation,
                "replicate": replicate,
                "native_test_phase_ba": native_metrics["test"][
                    "phase_balanced_accuracy"
                ],
                "native_test_progress_mae": native_metrics["test"][
                    "progress_sample_balanced_mae"
                ],
                "probe_test_phase_ba": probe["phase_probe_test_ba"],
                "probe_test_progress_mae": probe["progress_probe_test_mae"],
                **_probe_trajectory_columns(probe, "test"),
            }

        ordered_row = ablation_row(
            "ordered",
            0,
            payload["native"],
            ordered_probe,
        )
        ablation_rows.append(ordered_row)
        reset_entry = next(
            item for item in payload["ablations"]
            if item["ablation"] == "state_reset"
        )
        reset_row = ablation_row(
            "state_reset",
            int(reset_entry["replicate"]),
            reset_entry["native"],
            reset_entry["membrane_probe"],
        )
        ablation_rows.append(reset_row)

        shuffle_rows: list[dict[str, object]] = []
        for item in payload["ablations"]:
            if item["ablation"] != "temporal_shuffle":
                continue
            shuffle_row = ablation_row(
                "temporal_shuffle",
                int(item["replicate"]),
                item["native"],
                item["membrane_probe"],
            )
            ablation_rows.append(shuffle_row)
            shuffle_rows.append(shuffle_row)

        if len(shuffle_rows) != parent.SHUFFLE_REPLICATES:
            raise ValueError(
                f"Expected {parent.SHUFFLE_REPLICATES} shuffles for {spec.key}"
            )
        shuffle_phase_ba = float(
            np.mean([row["probe_test_phase_ba"] for row in shuffle_rows])
        )
        shuffle_progress_mae = float(
            np.mean([row["probe_test_sample_balanced_mae_mean"] for row in shuffle_rows])
        )
        shuffle_spearman = float(
            np.mean([row["probe_test_spearman_mean"] for row in shuffle_rows])
        )
        history_gain_rows.append(
            {
                **common,
                "ordered_phase_ba": ordered_row["probe_test_phase_ba"],
                "reset_phase_ba": reset_row["probe_test_phase_ba"],
                "shuffle_phase_ba": shuffle_phase_ba,
                "H_reset_phase_ba": ordered_row["probe_test_phase_ba"]
                - reset_row["probe_test_phase_ba"],
                "H_shuffle_phase_ba": ordered_row["probe_test_phase_ba"]
                - shuffle_phase_ba,
                "ordered_progress_mae": ordered_row[
                    "probe_test_sample_balanced_mae_mean"
                ],
                "reset_progress_mae": reset_row[
                    "probe_test_sample_balanced_mae_mean"
                ],
                "shuffle_progress_mae": shuffle_progress_mae,
                "H_reset_progress_mae": reset_row[
                    "probe_test_sample_balanced_mae_mean"
                ]
                - ordered_row["probe_test_sample_balanced_mae_mean"],
                "H_shuffle_progress_mae": shuffle_progress_mae
                - ordered_row["probe_test_sample_balanced_mae_mean"],
                "ordered_spearman": ordered_row["probe_test_spearman_mean"],
                "reset_spearman": reset_row["probe_test_spearman_mean"],
                "shuffle_spearman": shuffle_spearman,
                "H_reset_spearman": ordered_row["probe_test_spearman_mean"]
                - reset_row["probe_test_spearman_mean"],
                "H_shuffle_spearman": ordered_row["probe_test_spearman_mean"]
                - shuffle_spearman,
            }
        )

        hpath = history_path(root, spec)
        if not hpath.exists():
            raise FileNotFoundError(
                f"Missing Exp5.3.2.1 history: {hpath}"
            )
        history = pd.read_csv(hpath)
        history.insert(0, "seed", spec.seed)
        history.insert(0, "objective", objective.name)
        history_parts.append(history)

        best_epoch = int(payload["best_epoch"])
        best_row = history.loc[history["epoch"].eq(best_epoch)]
        if len(best_row) != 1:
            raise ValueError(f"Missing unique best epoch in history: {spec.key}")
        gradient_rows.append(
            {
                **common,
                "mean_raw_phase_grad_norm": history[
                    "raw_phase_grad_norm"
                ].mean(),
                "mean_raw_progress_grad_norm": history[
                    "raw_progress_grad_norm"
                ].mean(),
                "mean_weighted_phase_grad_norm": history[
                    "weighted_phase_grad_norm"
                ].mean(),
                "mean_weighted_progress_grad_norm": history[
                    "weighted_progress_grad_norm"
                ].mean(),
                "mean_weighted_phase_grad_fraction": history[
                    "weighted_phase_grad_fraction"
                ].mean(),
                "mean_weighted_progress_grad_fraction": history[
                    "weighted_progress_grad_fraction"
                ].mean(),
                "best_epoch_weighted_phase_grad_fraction": float(
                    best_row["weighted_phase_grad_fraction"].iloc[0]
                ),
                "best_epoch_weighted_progress_grad_fraction": float(
                    best_row["weighted_progress_grad_fraction"].iloc[0]
                ),
            }
        )

    for seed in SEEDS:
        bpath = baseline_path(root, seed)
        if not bpath.exists():
            raise FileNotFoundError(
                f"Missing Exp5.3.2.1 baseline: {bpath}"
            )
        baseline = json.loads(bpath.read_text(encoding="utf-8"))
        if (
            baseline.get("experiment_id") != EXPERIMENT_ID
            or baseline.get("protocol_version") != PROTOCOL_VERSION
            or baseline.get("seed") != seed
        ):
            raise ValueError(f"Baseline identity mismatch: {bpath}")
        for entry in baseline["baselines"]:
            baseline_rows.append({"seed": seed, **entry})

        reference_path = exp52.local_reference_path(
            parent.source_results_dir(repo_root),
            seed,
        )
        if not reference_path.exists():
            raise FileNotFoundError(
                f"Missing frozen Local reference: {reference_path}"
            )
        reference = json.loads(reference_path.read_text(encoding="utf-8"))
        for probe in reference["probes"]:
            local_rows.append(
                {
                    "seed": seed,
                    "probe_type": probe["probe_type"],
                    "val_ba": probe["probe_val_balanced_accuracy"],
                    "test_ba": probe["probe_test_balanced_accuracy"],
                    "test_accuracy": probe["probe_test_accuracy"],
                    "test_macro_f1": probe["probe_test_macro_f1"],
                }
            )

    outputs = {
        "runs": root / "runs.csv",
        "histories": root / "histories.csv",
        "probe_runs": root / "probe_runs.csv",
        "ablation_runs": root / "ablation_runs.csv",
        "history_gain_runs": root / "history_gain_runs.csv",
        "gradient_diagnostics": root / "gradient_diagnostics.csv",
        "baseline_runs": root / "baseline_runs.csv",
        "local_reference": root / "local_reference.csv",
        "manifest": root / "manifest.json",
    }
    root.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(run_rows).to_csv(outputs["runs"], index=False)
    pd.concat(history_parts, ignore_index=True).to_csv(
        outputs["histories"],
        index=False,
    )
    pd.DataFrame(probe_rows).to_csv(outputs["probe_runs"], index=False)
    pd.DataFrame(ablation_rows).to_csv(
        outputs["ablation_runs"],
        index=False,
    )
    pd.DataFrame(history_gain_rows).to_csv(
        outputs["history_gain_runs"],
        index=False,
    )
    pd.DataFrame(gradient_rows).to_csv(
        outputs["gradient_diagnostics"],
        index=False,
    )
    pd.DataFrame(baseline_rows).to_csv(
        outputs["baseline_runs"],
        index=False,
    )
    pd.DataFrame(local_rows).to_csv(
        outputs["local_reference"],
        index=False,
    )

    _save_json(
        outputs["manifest"],
        {
            "experiment_id": EXPERIMENT_ID,
            "protocol_version": PROTOCOL_VERSION,
            "expected_runs": len(run_specs()),
            "fixed_when_architecture": WHEN_CONDITION,
            "objectives": [asdict(objective) for objective in OBJECTIVES],
            "seeds": list(SEEDS),
            "source_when_experiment": parent.EXPERIMENT_ID,
            "source_when_protocol": parent.PROTOCOL_VERSION,
            "source_local_experiment": exp52.EXPERIMENT_ID,
            "source_local_protocol": exp52.PROTOCOL_VERSION,
            "primary_cross_objective_readout": (
                "common post-hoc Linear/Ridge probes on frozen U_t"
            ),
            "checkpoint_selection": (
                "min validation weighted objective loss; tie lower progress MAE; "
                "tie higher phase BA"
            ),
            "primary_metrics": [
                "U phase balanced accuracy",
                "U progress sample-balanced MAE",
                "per-gesture progress Spearman",
                "monotonic violation rate",
                "H_reset and H_shuffle for phase/progress",
                "elapsed-time baseline delta",
            ],
            "gradient_diagnostic": (
                "raw and objective-weighted gradient norms on shared RSNN "
                "input_projection + recurrent weights, one fixed train batch per epoch"
            ),
            "causal_ablation": ["state_reset", "temporal_shuffle"],
            "shuffle_replicates": parent.SHUFFLE_REPLICATES,
            "spike_readouts": [
                "membrane",
                "synaptic",
                "instantaneous spike",
                "trailing 250ms spike count",
                "trailing 500ms spike count",
            ],
            "baselines": ["what_only", "elapsed_time_only"],
            "aggregation_policy": (
                "finalizer only aggregates completed run/baseline artifacts; "
                "notebook is analysis-only"
            ),
            "files": {
                name: path.name
                for name, path in outputs.items()
                if name != "manifest"
            },
        },
    )
    return outputs


def _config_from_args(args: argparse.Namespace) -> Config:
    repo_root = (
        Path(args.repo_root).resolve()
        if args.repo_root
        else find_repo_root()
    )
    return Config(
        repo_root=repo_root,
        results_dir=results_dir(repo_root),
        device=args.device,
        epochs=args.epochs,
        batch_size=args.batch_size,
        threads=args.threads,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Experiment 5.3.2.1 RSNN WHEN objective upper-bound sweep"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    def common(subparser: argparse.ArgumentParser) -> None:
        subparser.add_argument("--repo-root", type=str, default=None)
        subparser.add_argument("--device", type=str, default="cpu")
        subparser.add_argument("--threads", type=int, default=1)
        subparser.add_argument(
            "--batch-size",
            type=int,
            default=parent.BATCH_SIZE,
        )
        subparser.add_argument(
            "--epochs",
            type=int,
            default=parent.EPOCHS,
        )
        subparser.add_argument("--force", action="store_true")

    prepare = subparsers.add_parser("prepare-local")
    common(prepare)
    prepare.add_argument("--array-task-id", type=int, required=True)

    run = subparsers.add_parser("run-one")
    common(run)
    run.add_argument("--array-task-id", type=int, required=True)

    finalize = subparsers.add_parser("finalize")
    finalize.add_argument("--repo-root", type=str, default=None)
    return parser


def main() -> None:
    args = _parser().parse_args()
    if args.command == "finalize":
        repo_root = (
            Path(args.repo_root).resolve()
            if args.repo_root
            else find_repo_root()
        )
        outputs = finalize_experiment(repo_root)
        for name, path in outputs.items():
            print(f"{name}: {path}")
        return

    config = _config_from_args(args)
    data = base.prepare_data(config.repo_root)

    if args.command == "prepare-local":
        task_id = int(args.array_task_id)
        if not 0 <= task_id < len(SEEDS):
            raise IndexError(
                f"prepare-local task {task_id} outside 0..{len(SEEDS)-1}"
            )
        print(
            prepare_local_seed(
                SEEDS[task_id],
                data,
                config,
                force=args.force,
            )
        )
        return

    if args.command == "run-one":
        specs = run_specs()
        task_id = int(args.array_task_id)
        if not 0 <= task_id < len(specs):
            raise IndexError(
                f"run-one task {task_id} outside 0..{len(specs)-1}"
            )
        payload = run_one(
            specs[task_id],
            data,
            config,
            force=args.force,
        )
        print(json.dumps(payload, indent=2, sort_keys=True))
        return

    raise RuntimeError(f"Unhandled command: {args.command}")


if __name__ == "__main__":
    main()
