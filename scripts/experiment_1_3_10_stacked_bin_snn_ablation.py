from __future__ import annotations

import argparse
import copy
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import random
import time

import numpy as np
import pandas as pd
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import snntorch as snn
from snntorch import surrogate
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score

from snn.accel_reconstruction_eval.datasets import load_acceleration_data


EXPERIMENT_ID = "experiment_1_3_10_stacked_bin_snn_ablation"
PROTOCOL_VERSION = "stacked250_v1"

SPLIT_SEEDS = (11, 23, 101)
N_TRAIN_USERS, N_VAL_USERS, N_TEST_USERS = 12, 4, 4
ARCHITECTURES: dict[str, tuple[int, ...]] = {
    "1h128": (128,),
    "1h256": (256,),
    "2h128": (128, 128),
}
OBJECTIVES = ("whole_count_ce", "timestep_ce")
TRAIN_REGIMES = ("weight_only", "trainable_dynamics")
EXPECTED_RUNS = len(SPLIT_SEEDS) * len(ARCHITECTURES) * len(OBJECTIVES) * len(TRAIN_REGIMES)

DEFAULT_LABELS = ("A", "B", "C", "D", "E", "X", "G", "H", "I", "J", "K", "L")
EVENT_CHANNEL_COUNT = 30
TOTAL_CHANNEL_COUNT = 36
EXPECTED_SAMPLING_RATE_HZ = 64.0
EXPECTED_EVENT_REPRESENTATION = "unsigned"
EXPECTED_EVENT_FEATURE_SCHEMA = "custom_wavelet_polarity_split_abs_events_v1"
GLOBAL_PADDED_LENGTH = 256
BIN_MS = 250.0
HIDDEN_SHIFTS = (2, 3, 4)
TAU_MEM_MS = 22.0
THRESHOLD = 0.5
OUTPUT_TAU_SYN_MS = 77.47
SURROGATE_SLOPE = 25.0
RESET_MECHANISM = "subtract"

BATCH_SIZE = 128
NUM_EPOCHS = 200
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 0.0
GRAD_CLIP_NORM: float | None = None


@dataclass(frozen=True)
class RunSpec:
    split_seed: int
    architecture: str
    objective: str
    train_regime: str

    @property
    def key(self) -> str:
        return (
            f"split{self.split_seed}__{self.architecture}__"
            f"{self.objective}__{self.train_regime}"
        )


@dataclass(frozen=True)
class Config:
    repo_root: Path
    labels: tuple[str, ...]
    device: str = "cpu"
    batch_size: int = BATCH_SIZE
    num_epochs: int = NUM_EPOCHS
    learning_rate: float = LEARNING_RATE
    weight_decay: float = WEIGHT_DECAY
    grad_clip_norm: float | None = GRAD_CLIP_NORM
    threads: int = 1
    overwrite: bool = False


@dataclass
class Cohort:
    manifest: pd.DataFrame
    packages: object
    fs: float
    labels: tuple[str, ...]


def find_repo_root(start: Path | None = None) -> Path:
    start = (start or Path.cwd()).resolve()
    for candidate in (start, *start.parents):
        if (candidate / "snn").is_dir() and (candidate / "notebooks").is_dir():
            return candidate
    raise FileNotFoundError("Could not locate writingRing repository root")


def derive_seed(seed: int, *parts: object) -> int:
    text = "|".join(map(str, (seed, *parts)))
    return int.from_bytes(hashlib.sha256(text.encode("utf-8")).digest()[:4], "little")


def seed_everything(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except TypeError:
        torch.use_deterministic_algorithms(True)


def parse_labels(text: str) -> tuple[str, ...]:
    labels = tuple(item.strip() for item in text.split(",") if item.strip())
    if len(labels) < 2:
        raise ValueError("At least two labels are required")
    if len(set(labels)) != len(labels):
        raise ValueError(f"Duplicate labels are not allowed: {labels}")
    unknown = sorted(set(labels) - set(DEFAULT_LABELS))
    if unknown:
        raise ValueError(f"Unknown labels: {unknown}; available labels: {DEFAULT_LABELS}")
    return tuple(sorted(labels))


def label_tag(labels: tuple[str, ...]) -> str:
    return "labels_" + "-".join(labels)


def results_dir(repo_root: Path, labels: tuple[str, ...]) -> Path:
    return (
        repo_root
        / "notebooks"
        / "artifacts"
        / EXPERIMENT_ID
        / PROTOCOL_VERSION
        / label_tag(labels)
    )


def run_specs() -> list[RunSpec]:
    return [
        RunSpec(split_seed, architecture, objective, train_regime)
        for split_seed in SPLIT_SEEDS
        for architecture in ARCHITECTURES
        for objective in OBJECTIVES
        for train_regime in TRAIN_REGIMES
    ]


def _dataset_roots(repo_root: Path) -> list[Path]:
    return [
        repo_root
        / "outputs/action0_wavelets_0e5_1_2_4_8_sr_64/low-pass/aligned-board-events/segmentation_padded",
        repo_root
        / "outputs/action1_wavelets_0e5_1_2_4_8_sr_64/low-pass/aligned-board-events/segmentation_padded",
    ]


def load_cohort(repo_root: Path, selected_labels: tuple[str, ...]) -> Cohort:
    data = load_acceleration_data(
        _dataset_roots(repo_root),
        repository_root=repo_root,
        require_reconstruction=False,
    )
    fs_values = {float(metadata.sampling_rate_hz) for metadata in data.producer_metadatas}
    if len(fs_values) != 1:
        raise ValueError(f"Expected one shared sampling rate, got {fs_values}")
    fs = fs_values.pop()
    if not np.isclose(fs, EXPECTED_SAMPLING_RATE_HZ):
        raise ValueError(f"Expected {EXPECTED_SAMPLING_RATE_HZ} Hz, got {fs}")

    for root, metadata in zip(data.padded_roots, data.producer_metadatas, strict=True):
        raw = metadata.raw
        if raw.get("event_representation") != EXPECTED_EVENT_REPRESENTATION:
            raise ValueError(f"{root}: unexpected event representation")
        if raw.get("event_feature_schema") != EXPECTED_EVENT_FEATURE_SCHEMA:
            raise ValueError(f"{root}: unexpected event feature schema")
        if raw.get("event_channel_count") != EVENT_CHANNEL_COUNT:
            raise ValueError(f"{root}: expected {EVENT_CHANNEL_COUNT} event channels")
        if metadata.channel_count != TOTAL_CHANNEL_COUNT:
            raise ValueError(f"{root}: expected {TOTAL_CHANNEL_COUNT} total channels")

    rows: list[dict[str, object]] = []
    all_labels = set(DEFAULT_LABELS)
    for package_index, package in enumerate(data.packages):
        for segment_index, label in enumerate(package.labels.astype(str)):
            if label not in all_labels:
                continue
            rows.append(
                {
                    "package_index": package_index,
                    "segment_index": segment_index,
                    "user": str(package.user),
                    "action": str(package.action),
                    "label": str(label),
                    "valid_length": int(package.valid_lengths[segment_index]),
                    "package_padded_length": int(package.padded_spike_imu.shape[1]),
                    "sample_id": f"{package.user}/action_{package.action}/{segment_index}",
                }
            )
    full_manifest = pd.DataFrame(rows)
    if len(full_manifest) != 853 or full_manifest.user.nunique() != 20:
        raise ValueError(
            f"Unexpected full cohort: samples={len(full_manifest)} users={full_manifest.user.nunique()}"
        )
    if int(full_manifest.package_padded_length.max()) != GLOBAL_PADDED_LENGTH:
        raise ValueError(
            f"Expected global padded length {GLOBAL_PADDED_LENGTH}, "
            f"got {full_manifest.package_padded_length.max()}"
        )

    manifest = full_manifest[full_manifest.label.isin(selected_labels)].reset_index(drop=True)
    present_labels = tuple(sorted(manifest.label.unique().tolist()))
    if present_labels != tuple(sorted(selected_labels)):
        raise ValueError(f"Requested labels {selected_labels}, found {present_labels}")
    if manifest.user.nunique() != 20:
        raise ValueError(
            "Selected-label cohort must still contain all 20 users for the documented user split protocol; "
            f"found {manifest.user.nunique()}"
        )

    class_to_idx = {label: idx for idx, label in enumerate(present_labels)}
    manifest["label_idx"] = manifest.label.map(class_to_idx).astype(int)
    return Cohort(manifest=manifest, packages=data.packages, fs=fs, labels=present_labels)


def make_user_split(manifest: pd.DataFrame, split_seed: int) -> dict[str, pd.DataFrame]:
    if split_seed not in SPLIT_SEEDS:
        raise ValueError(f"Unknown split seed: {split_seed}")
    users = sorted(manifest.user.unique().tolist())
    if len(users) != N_TRAIN_USERS + N_VAL_USERS + N_TEST_USERS:
        raise ValueError(f"Expected 20 users, got {len(users)}")
    rng = np.random.default_rng(derive_seed(split_seed, "user_split"))
    perm = np.asarray(users, dtype=object)
    rng.shuffle(perm)
    train_users = set(perm[:N_TRAIN_USERS].tolist())
    val_users = set(perm[N_TRAIN_USERS : N_TRAIN_USERS + N_VAL_USERS].tolist())
    test_users = set(perm[-N_TEST_USERS:].tolist())
    if train_users & val_users or train_users & test_users or val_users & test_users:
        raise RuntimeError("User-disjoint split construction failed")
    return {
        "train": manifest[manifest.user.isin(train_users)].reset_index(drop=True),
        "val": manifest[manifest.user.isin(val_users)].reset_index(drop=True),
        "test": manifest[manifest.user.isin(test_users)].reset_index(drop=True),
    }


def split_users(parts: dict[str, pd.DataFrame]) -> dict[str, tuple[str, ...]]:
    return {
        f"{name}_users": tuple(sorted(frame.user.unique().tolist()))
        for name, frame in parts.items()
    }


def sample_hash(parts: dict[str, pd.DataFrame]) -> str:
    payload: list[str] = []
    for split_name in ("train", "val", "test"):
        payload.extend(
            f"{split_name}:{sample_id}" for sample_id in parts[split_name].sample_id.tolist()
        )
    return hashlib.sha256("\n".join(payload).encode("utf-8")).hexdigest()


def build_global_padded_events(cohort: Cohort, frame: pd.DataFrame) -> np.ndarray:
    out = np.zeros((len(frame), GLOBAL_PADDED_LENGTH, EVENT_CHANNEL_COUNT), dtype=np.float32)
    for index, row in enumerate(frame.itertuples(index=False)):
        package = cohort.packages[int(row.package_index)]
        x = np.asarray(
            package.padded_spike_imu[int(row.segment_index), :, :EVENT_CHANNEL_COUNT],
            dtype=np.float32,
        )
        valid = min(int(row.valid_length), len(x), GLOBAL_PADDED_LENGTH)
        out[index, :valid] = x[:valid]
    return out


def build_stacked_input(cohort: Cohort, frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, dict[str, int | float]]:
    bin_steps = max(1, int(np.rint(BIN_MS * cohort.fs / 1000.0)))
    n_bins = int(math.ceil(GLOBAL_PADDED_LENGTH / bin_steps))
    padded_steps = n_bins * bin_steps
    events = build_global_padded_events(cohort, frame)
    if padded_steps > GLOBAL_PADDED_LENGTH:
        events = np.pad(events, ((0, 0), (0, padded_steps - GLOBAL_PADDED_LENGTH), (0, 0)))
    n, _, channels = events.shape
    bins = events.reshape(n, n_bins, bin_steps, channels)
    stacked = (
        bins.transpose(0, 2, 1, 3)
        .reshape(n, bin_steps, n_bins * channels)
        .astype(np.float32, copy=False)
    )
    labels = frame.label_idx.to_numpy(dtype=np.int64, copy=True)
    meta = {
        "bin_ms_requested": float(BIN_MS),
        "bin_steps": int(bin_steps),
        "bin_ms_actual": float(1000.0 * bin_steps / cohort.fs),
        "n_bins": int(n_bins),
        "input_dim": int(n_bins * channels),
        "snn_timesteps": int(bin_steps),
    }
    return stacked, labels, meta


def shift_to_alpha(shift: int) -> float:
    return float(1.0 - 2.0 ** (-int(shift)))


def tau_ms_to_decay(tau_ms: float, sampling_rate_hz: float) -> float:
    dt_ms = 1000.0 / float(sampling_rate_hz)
    return float(math.exp(-dt_ms / float(tau_ms)))


def decay_to_tau_ms(decay: np.ndarray | torch.Tensor | float, sampling_rate_hz: float) -> np.ndarray:
    values = np.asarray(decay, dtype=np.float64)
    values = np.clip(values, 1e-6, 1.0 - 1e-6)
    dt_ms = 1000.0 / float(sampling_rate_hz)
    return -dt_ms / np.log(values)


def allocate_neurons(shifts: tuple[int, ...], width: int) -> tuple[int, ...]:
    base, remainder = divmod(width, len(shifts))
    counts = [base] * len(shifts)
    if remainder == 1:
        counts[0] += 1
    elif remainder == 2:
        counts[0] += 1
        counts[-1] += 1
    else:
        for index in range(remainder):
            counts[index] += 1
    return tuple(counts)


def build_alpha_vector(width: int) -> torch.Tensor:
    counts = allocate_neurons(HIDDEN_SHIFTS, width)
    values = [
        shift_to_alpha(shift)
        for shift, count in zip(HIDDEN_SHIFTS, counts, strict=True)
        for _ in range(count)
    ]
    if len(values) != width:
        raise RuntimeError("Hidden alpha allocation produced the wrong width")
    return torch.tensor(values, dtype=torch.float32)


class StackedBinSNN(nn.Module):
    """Feed-forward stacked-bin Synaptic-LIF SNN with configurable hidden depth/width."""

    def __init__(
        self,
        input_dim: int,
        hidden_sizes: tuple[int, ...],
        num_classes: int,
        sampling_rate_hz: float,
        trainable_dynamics: bool,
    ) -> None:
        super().__init__()
        if not hidden_sizes:
            raise ValueError("At least one hidden layer is required")
        self.hidden_sizes = tuple(int(width) for width in hidden_sizes)
        self.num_classes = int(num_classes)
        self.sampling_rate_hz = float(sampling_rate_hz)
        self.trainable_dynamics = bool(trainable_dynamics)

        spike_grad = surrogate.fast_sigmoid(slope=SURROGATE_SLOPE)
        beta0 = tau_ms_to_decay(TAU_MEM_MS, self.sampling_rate_hz)

        self.hidden_fcs = nn.ModuleList()
        self.hidden_lifs = nn.ModuleList()
        previous = int(input_dim)
        for width in self.hidden_sizes:
            self.hidden_fcs.append(nn.Linear(previous, width, bias=False))
            self.hidden_lifs.append(
                snn.Synaptic(
                    alpha=build_alpha_vector(width),
                    beta=torch.full((width,), beta0, dtype=torch.float32),
                    threshold=torch.full((width,), THRESHOLD, dtype=torch.float32),
                    spike_grad=spike_grad,
                    reset_mechanism=RESET_MECHANISM,
                    learn_alpha=self.trainable_dynamics,
                    learn_beta=self.trainable_dynamics,
                    learn_threshold=self.trainable_dynamics,
                )
            )
            previous = width

        self.fc_out = nn.Linear(previous, self.num_classes, bias=False)
        output_alpha0 = tau_ms_to_decay(OUTPUT_TAU_SYN_MS, self.sampling_rate_hz)
        self.lif_out = snn.Synaptic(
            alpha=output_alpha0,
            beta=beta0,
            threshold=THRESHOLD,
            spike_grad=spike_grad,
            reset_mechanism=RESET_MECHANISM,
            learn_alpha=self.trainable_dynamics,
            learn_beta=self.trainable_dynamics,
            learn_threshold=self.trainable_dynamics,
        )

    def forward(self, x: torch.Tensor, return_hidden: bool = False):
        if x.ndim != 3:
            raise ValueError(f"Expected [B,T,C] input, got {tuple(x.shape)}")
        batch_size = x.shape[0]
        device = x.device
        dtype = x.dtype

        syn_states = [torch.zeros(batch_size, width, device=device, dtype=dtype) for width in self.hidden_sizes]
        mem_states = [torch.zeros_like(state) for state in syn_states]
        syn_out = torch.zeros(batch_size, self.num_classes, device=device, dtype=dtype)
        mem_out = torch.zeros_like(syn_out)

        output_spikes: list[torch.Tensor] = []
        hidden_records: list[list[torch.Tensor]] | None = (
            [[] for _ in self.hidden_sizes] if return_hidden else None
        )

        for timestep in range(x.shape[1]):
            current = x[:, timestep]
            for layer_index, (fc, lif) in enumerate(zip(self.hidden_fcs, self.hidden_lifs, strict=True)):
                current = fc(current)
                spike, syn_states[layer_index], mem_states[layer_index] = lif(
                    current,
                    syn_states[layer_index],
                    mem_states[layer_index],
                )
                current = spike
                if hidden_records is not None:
                    hidden_records[layer_index].append(spike)
            current_out = self.fc_out(current)
            spike_out, syn_out, mem_out = self.lif_out(current_out, syn_out, mem_out)
            output_spikes.append(spike_out)

        out = torch.stack(output_spikes, dim=1)
        if hidden_records is None:
            return out
        hidden = [torch.stack(records, dim=1) for records in hidden_records]
        return out, hidden


def objective_loss(out_spikes: torch.Tensor, labels: torch.Tensor, objective: str) -> torch.Tensor:
    if objective == "whole_count_ce":
        return F.cross_entropy(out_spikes.sum(dim=1), labels)
    if objective == "timestep_ce":
        batch, timesteps, classes = out_spikes.shape
        targets = labels[:, None].expand(batch, timesteps).reshape(-1)
        return F.cross_entropy(out_spikes.reshape(batch * timesteps, classes), targets)
    raise ValueError(f"Unknown objective: {objective}")


def whole_count_logits(out_spikes: torch.Tensor) -> torch.Tensor:
    return out_spikes.sum(dim=1)


def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
    }


def make_loader(
    x: np.ndarray,
    y: np.ndarray,
    batch_size: int,
    shuffle: bool,
    seed: int,
) -> DataLoader:
    dataset = TensorDataset(torch.from_numpy(x.copy()), torch.from_numpy(y.copy()))
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=False,
        generator=generator if shuffle else None,
        drop_last=False,
    )


def evaluate(
    model: StackedBinSNN,
    loader: DataLoader,
    objective: str,
    device: torch.device,
) -> dict[str, object]:
    model.eval()
    losses: list[float] = []
    y_true: list[np.ndarray] = []
    y_pred: list[np.ndarray] = []
    hidden_spike_sums: list[float] | None = None
    hidden_spike_counts: list[int] | None = None
    output_spike_sum = 0.0
    output_spike_count = 0

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            out, hidden = model(x, return_hidden=True)
            loss = objective_loss(out, y, objective)
            losses.append(float(loss.item()) * len(y))
            logits = whole_count_logits(out)
            y_true.append(y.cpu().numpy())
            y_pred.append(logits.argmax(dim=1).cpu().numpy())

            if hidden_spike_sums is None:
                hidden_spike_sums = [0.0] * len(hidden)
                hidden_spike_counts = [0] * len(hidden)
            assert hidden_spike_counts is not None
            for index, spikes in enumerate(hidden):
                hidden_spike_sums[index] += float(spikes.sum().item())
                hidden_spike_counts[index] += int(spikes.numel())
            output_spike_sum += float(out.sum().item())
            output_spike_count += int(out.numel())

    truth = np.concatenate(y_true)
    pred = np.concatenate(y_pred)
    metrics: dict[str, object] = {
        "loss": float(sum(losses) / len(truth)),
        **classification_metrics(truth, pred),
        "output_firing_rate": float(output_spike_sum / max(1, output_spike_count)),
    }
    if hidden_spike_sums is not None and hidden_spike_counts is not None:
        metrics["hidden_firing_rates"] = [
            float(total / max(1, count))
            for total, count in zip(hidden_spike_sums, hidden_spike_counts, strict=True)
        ]
    else:
        metrics["hidden_firing_rates"] = []
    return metrics


def _tensor_numpy(value: torch.Tensor | float) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy().astype(np.float64, copy=False)
    return np.asarray(value, dtype=np.float64)


def _summary(values: np.ndarray) -> dict[str, float]:
    flat = np.asarray(values, dtype=np.float64).reshape(-1)
    return {
        "mean": float(np.mean(flat)),
        "std": float(np.std(flat)),
        "min": float(np.min(flat)),
        "q25": float(np.quantile(flat, 0.25)),
        "median": float(np.median(flat)),
        "q75": float(np.quantile(flat, 0.75)),
        "max": float(np.max(flat)),
    }


def dynamics_summary(model: StackedBinSNN) -> dict[str, object]:
    layers: list[dict[str, object]] = []
    for index, lif in enumerate(model.hidden_lifs, start=1):
        alpha = _tensor_numpy(lif.alpha)
        beta = _tensor_numpy(lif.beta)
        threshold = _tensor_numpy(lif.threshold)
        layers.append(
            {
                "layer": f"hidden_{index}",
                "tau_syn_ms": _summary(decay_to_tau_ms(alpha, model.sampling_rate_hz)),
                "tau_mem_ms": _summary(decay_to_tau_ms(beta, model.sampling_rate_hz)),
                "threshold": _summary(threshold),
            }
        )
    output_alpha = _tensor_numpy(model.lif_out.alpha)
    output_beta = _tensor_numpy(model.lif_out.beta)
    output_threshold = _tensor_numpy(model.lif_out.threshold)
    return {
        "hidden": layers,
        "output": {
            "tau_syn_ms": _summary(decay_to_tau_ms(output_alpha, model.sampling_rate_hz)),
            "tau_mem_ms": _summary(decay_to_tau_ms(output_beta, model.sampling_rate_hz)),
            "threshold": _summary(output_threshold),
        },
    }


def parameter_counts(model: nn.Module) -> dict[str, int]:
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    return {"total": int(total), "trainable": int(trainable)}


def atomic_json_dump(payload: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
    os.replace(tmp, path)


def atomic_csv_dump(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    frame.to_csv(tmp, index=False)
    os.replace(tmp, path)


def run_path(config: Config, spec: RunSpec) -> Path:
    return results_dir(config.repo_root, config.labels) / "runs" / f"run_{spec.key}.json"


def history_path(config: Config, spec: RunSpec) -> Path:
    return results_dir(config.repo_root, config.labels) / "histories" / f"history_{spec.key}.csv"


def train_one_run(config: Config, spec: RunSpec) -> dict[str, object]:
    output_path = run_path(config, spec)
    if output_path.exists() and not config.overwrite:
        with output_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        print(f"Run exists; skipping: {output_path}")
        return payload

    torch.set_num_threads(max(1, config.threads))
    try:
        torch.set_num_interop_threads(max(1, min(config.threads, 2)))
    except RuntimeError:
        pass

    cohort = load_cohort(config.repo_root, config.labels)
    parts = make_user_split(cohort.manifest, spec.split_seed)
    train_x, train_y, representation_meta = build_stacked_input(cohort, parts["train"])
    val_x, val_y, _ = build_stacked_input(cohort, parts["val"])
    test_x, test_y, _ = build_stacked_input(cohort, parts["test"])

    hidden_sizes = ARCHITECTURES[spec.architecture]
    model_seed = derive_seed(spec.split_seed, "model_init", spec.architecture)
    loader_seed = derive_seed(spec.split_seed, "loader", spec.architecture)
    seed_everything(model_seed)

    device = torch.device(config.device)
    model = StackedBinSNN(
        input_dim=representation_meta["input_dim"],
        hidden_sizes=hidden_sizes,
        num_classes=len(cohort.labels),
        sampling_rate_hz=cohort.fs,
        trainable_dynamics=(spec.train_regime == "trainable_dynamics"),
    ).to(device)
    initial_dynamics = dynamics_summary(model)
    counts = parameter_counts(model)

    train_loader = make_loader(train_x, train_y, config.batch_size, True, loader_seed)
    train_eval_loader = make_loader(train_x, train_y, config.batch_size, False, loader_seed)
    val_loader = make_loader(val_x, val_y, config.batch_size, False, loader_seed)
    test_loader = make_loader(test_x, test_y, config.batch_size, False, loader_seed)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    best_state: dict[str, torch.Tensor] | None = None
    best_epoch = -1
    best_val_ba = -np.inf
    best_val_loss = np.inf
    history: list[dict[str, object]] = []

    print("=" * 110)
    print(
        f"run={spec.key} | labels={cohort.labels} | hidden={hidden_sizes} | "
        f"device={device} | trainable_params={counts['trainable']}"
    )
    print(
        f"stacked shape: T={representation_meta['snn_timesteps']} x C={representation_meta['input_dim']} | "
        f"split sizes={len(train_y)}/{len(val_y)}/{len(test_y)}"
    )

    for epoch in range(1, config.num_epochs + 1):
        model.train()
        batch_loss_sum = 0.0
        sample_count = 0
        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad(set_to_none=True)
            out = model(x)
            loss = objective_loss(out, y, spec.objective)
            loss.backward()
            if config.grad_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip_norm)
            optimizer.step()
            batch_loss_sum += float(loss.item()) * len(y)
            sample_count += len(y)

        train_metrics = evaluate(model, train_eval_loader, spec.objective, device)
        val_metrics = evaluate(model, val_loader, spec.objective, device)
        train_objective_loss = float(batch_loss_sum / max(1, sample_count))
        row: dict[str, object] = {
            "epoch": epoch,
            "train_objective_loss": train_objective_loss,
            "train_loss": train_metrics["loss"],
            "train_balanced_accuracy": train_metrics["balanced_accuracy"],
            "train_accuracy": train_metrics["accuracy"],
            "train_macro_f1": train_metrics["macro_f1"],
            "val_loss": val_metrics["loss"],
            "val_balanced_accuracy": val_metrics["balanced_accuracy"],
            "val_accuracy": val_metrics["accuracy"],
            "val_macro_f1": val_metrics["macro_f1"],
            "train_output_firing_rate": train_metrics["output_firing_rate"],
            "val_output_firing_rate": val_metrics["output_firing_rate"],
        }
        for layer_index, firing_rate in enumerate(train_metrics["hidden_firing_rates"], start=1):
            row[f"train_hidden{layer_index}_firing_rate"] = firing_rate
        for layer_index, firing_rate in enumerate(val_metrics["hidden_firing_rates"], start=1):
            row[f"val_hidden{layer_index}_firing_rate"] = firing_rate
        history.append(row)

        val_ba = float(val_metrics["balanced_accuracy"])
        val_loss = float(val_metrics["loss"])
        improved = val_ba > best_val_ba or (
            np.isclose(val_ba, best_val_ba) and val_loss < best_val_loss
        )
        if improved:
            best_val_ba = val_ba
            best_val_loss = val_loss
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())

        if epoch == 1 or epoch % 10 == 0 or epoch == config.num_epochs:
            hidden_fr = "/".join(f"{value:.4f}" for value in val_metrics["hidden_firing_rates"])
            print(
                f"epoch={epoch:3d} | train BA={float(train_metrics['balanced_accuracy']):.4f} "
                f"| val BA={val_ba:.4f} | val loss={val_loss:.4f} "
                f"| val hidden/out FR={hidden_fr}/{float(val_metrics['output_firing_rate']):.4f}"
            )

    if best_state is None:
        raise RuntimeError("Training produced no checkpoint")
    model.load_state_dict(best_state)
    train_best = evaluate(model, train_eval_loader, spec.objective, device)
    val_best = evaluate(model, val_loader, spec.objective, device)
    test_best = evaluate(model, test_loader, spec.objective, device)
    final_dynamics = dynamics_summary(model)

    history_frame = pd.DataFrame(history)
    atomic_csv_dump(history_frame, history_path(config, spec))

    payload: dict[str, object] = {
        "experiment_id": EXPERIMENT_ID,
        "protocol_version": PROTOCOL_VERSION,
        "run_key": spec.key,
        "split_seed": int(spec.split_seed),
        "architecture": spec.architecture,
        "hidden_sizes": list(hidden_sizes),
        "objective": spec.objective,
        "train_regime": spec.train_regime,
        "labels": list(cohort.labels),
        "num_classes": len(cohort.labels),
        "sampling_rate_hz": cohort.fs,
        "representation": representation_meta,
        "hidden_shifts": list(HIDDEN_SHIFTS),
        "hidden_shift_allocations": [
            list(allocate_neurons(HIDDEN_SHIFTS, width)) for width in hidden_sizes
        ],
        "tau_mem_init_ms": TAU_MEM_MS,
        "threshold_init": THRESHOLD,
        "output_tau_syn_init_ms": OUTPUT_TAU_SYN_MS,
        "surrogate_slope": SURROGATE_SLOPE,
        "reset_mechanism": RESET_MECHANISM,
        "model_seed": int(model_seed),
        "loader_seed": int(loader_seed),
        "sample_hash": sample_hash(parts),
        **split_users(parts),
        "split_sizes": {name: int(len(frame)) for name, frame in parts.items()},
        "parameter_counts": counts,
        "training": {
            "batch_size": config.batch_size,
            "num_epochs": config.num_epochs,
            "learning_rate": config.learning_rate,
            "weight_decay": config.weight_decay,
            "grad_clip_norm": config.grad_clip_norm,
            "device": str(device),
            "threads": config.threads,
        },
        "best_epoch": int(best_epoch),
        "metrics": {
            "train": train_best,
            "val": val_best,
            "test": test_best,
        },
        "initial_dynamics": initial_dynamics,
        "best_dynamics": final_dynamics,
        "history_file": str(history_path(config, spec).relative_to(config.repo_root)),
    }
    atomic_json_dump(payload, output_path)
    print(
        f"BEST epoch={best_epoch} | val BA={float(val_best['balanced_accuracy']):.4f} "
        f"| test BA={float(test_best['balanced_accuracy']):.4f} "
        f"| test macro-F1={float(test_best['macro_f1']):.4f}"
    )
    print(f"Saved: {output_path}")
    return payload


def flatten_run(payload: dict[str, object]) -> dict[str, object]:
    metrics = payload["metrics"]
    assert isinstance(metrics, dict)
    row: dict[str, object] = {
        "run_key": payload["run_key"],
        "split_seed": payload["split_seed"],
        "architecture": payload["architecture"],
        "objective": payload["objective"],
        "train_regime": payload["train_regime"],
        "num_classes": payload["num_classes"],
        "best_epoch": payload["best_epoch"],
        "trainable_parameters": payload["parameter_counts"]["trainable"],
        "total_parameters": payload["parameter_counts"]["total"],
        "sample_hash": payload["sample_hash"],
    }
    for split_name in ("train", "val", "test"):
        split_metrics = metrics[split_name]
        for metric in ("loss", "balanced_accuracy", "accuracy", "macro_f1", "output_firing_rate"):
            row[f"{split_name}_{metric}"] = split_metrics[metric]
        for layer_index, firing_rate in enumerate(split_metrics["hidden_firing_rates"], start=1):
            row[f"{split_name}_hidden{layer_index}_firing_rate"] = firing_rate

    best_dynamics = payload["best_dynamics"]
    for hidden in best_dynamics["hidden"]:
        layer_name = hidden["layer"]
        for quantity in ("tau_syn_ms", "tau_mem_ms", "threshold"):
            row[f"{layer_name}_{quantity}_mean"] = hidden[quantity]["mean"]
            row[f"{layer_name}_{quantity}_std"] = hidden[quantity]["std"]
    output = best_dynamics["output"]
    for quantity in ("tau_syn_ms", "tau_mem_ms", "threshold"):
        row[f"output_{quantity}_mean"] = output[quantity]["mean"]
        row[f"output_{quantity}_std"] = output[quantity]["std"]
    return row


def finalize_if_ready(config: Config, require_complete: bool = False) -> bool:
    root = results_dir(config.repo_root, config.labels)
    payloads: list[dict[str, object]] = []
    missing: list[str] = []
    for spec in run_specs():
        path = run_path(config, spec)
        if not path.exists():
            missing.append(spec.key)
            continue
        with path.open("r", encoding="utf-8") as handle:
            payloads.append(json.load(handle))
    if missing:
        print(f"Finalization deferred: {len(payloads)}/{EXPECTED_RUNS} runs complete")
        if require_complete:
            raise RuntimeError(f"Missing {len(missing)} runs; first missing: {missing[:5]}")
        return False

    frame = pd.DataFrame(flatten_run(payload) for payload in payloads)
    frame = frame.sort_values(["architecture", "train_regime", "objective", "split_seed"]).reset_index(drop=True)
    atomic_csv_dump(frame, root / "all_runs.csv")

    summary = (
        frame.groupby(["architecture", "train_regime", "objective"], as_index=False)
        .agg(
            n_splits=("split_seed", "nunique"),
            mean_val_ba=("val_balanced_accuracy", "mean"),
            sd_val_ba=("val_balanced_accuracy", "std"),
            mean_test_ba=("test_balanced_accuracy", "mean"),
            sd_test_ba=("test_balanced_accuracy", "std"),
            mean_test_accuracy=("test_accuracy", "mean"),
            sd_test_accuracy=("test_accuracy", "std"),
            mean_test_macro_f1=("test_macro_f1", "mean"),
            sd_test_macro_f1=("test_macro_f1", "std"),
            mean_best_epoch=("best_epoch", "mean"),
        )
        .sort_values(["architecture", "train_regime", "objective"])
        .reset_index(drop=True)
    )
    atomic_csv_dump(summary, root / "summary.csv")

    manifest = {
        "experiment_id": EXPERIMENT_ID,
        "protocol_version": PROTOCOL_VERSION,
        "labels": list(config.labels),
        "expected_runs": EXPECTED_RUNS,
        "split_seeds": list(SPLIT_SEEDS),
        "architectures": {name: list(widths) for name, widths in ARCHITECTURES.items()},
        "objectives": list(OBJECTIVES),
        "train_regimes": list(TRAIN_REGIMES),
        "readout": "whole_output_spike_count",
        "representation": "250ms bins stacked across absolute-bin identity; 16 within-bin timesteps x 480 channels at 64 Hz",
    }
    atomic_json_dump(manifest, root / "experiment_manifest.json")
    print(f"Finalized {len(frame)} runs under {root}")
    return True


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-index", type=int, default=None, help=f"Run index in [0, {EXPECTED_RUNS - 1}]")
    parser.add_argument(
        "--labels",
        type=str,
        default=",".join(DEFAULT_LABELS),
        help="Comma-separated label subset; output-neuron count follows this selection",
    )
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--threads", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", "1")))
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--num-epochs", type=int, default=NUM_EPOCHS)
    parser.add_argument("--learning-rate", type=float, default=LEARNING_RATE)
    parser.add_argument("--weight-decay", type=float, default=WEIGHT_DECAY)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--settle-seconds", type=float, default=0.0)
    parser.add_argument("--finalize-if-ready", action="store_true")
    parser.add_argument("--finalize-only", action="store_true")
    parser.add_argument("--require-complete", action="store_true")
    parser.add_argument("--list-runs", action="store_true")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    labels = parse_labels(args.labels)
    repo_root = find_repo_root()
    config = Config(
        repo_root=repo_root,
        labels=labels,
        device=args.device,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        threads=max(1, args.threads),
        overwrite=args.overwrite,
    )

    specs = run_specs()
    if args.list_runs:
        for index, spec in enumerate(specs):
            print(index, spec.key)
        return

    if args.finalize_only:
        finalize_if_ready(config, require_complete=args.require_complete)
        return

    if args.run_index is None:
        raise SystemExit("Provide --run-index, --finalize-only, or --list-runs")
    if not 0 <= args.run_index < len(specs):
        raise SystemExit(f"--run-index must be in [0, {len(specs) - 1}]")

    spec = specs[args.run_index]
    train_one_run(config, spec)
    if args.finalize_if_ready:
        if args.settle_seconds > 0:
            time.sleep(args.settle_seconds)
        finalize_if_ready(config, require_complete=False)


if __name__ == "__main__":
    main()
