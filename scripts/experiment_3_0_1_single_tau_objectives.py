from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import copy
import hashlib
import math
import os
import random

import numpy as np
import pandas as pd
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import snntorch as snn
from snntorch import surrogate
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler

from snn.accel_reconstruction_eval.datasets import load_acceleration_data


EXPERIMENT_ID = "experiment_3_0_1_single_tau_objective_comparison"
PROTOCOL_VERSION = "paired_split_v2"

SHIFTS = (2, 3, 4, 5, 6, 7)
OBJECTIVES = (
    "timestep_ce",
    "relative10_sequence_ce",
    "fixed250_sequence_ce",
)
SEEDS = (11, 23, 101)
EXPECTED_RUNS = len(SHIFTS) * len(OBJECTIVES) * len(SEEDS)

SPLIT_SEED = 12345
TRAIN_FRACTION = 0.70
VAL_FRACTION = 0.15
WIDTHS = (128, 128, 64)
TAU_MEM_MS = 22.54
THRESHOLD = 0.5
SURROGATE_SLOPE = 25.0
RESET = "subtract"

BATCH_SIZE = 128
EPOCHS = 100
LR = 1e-3
N_REL = 10
FIXED_MS = 250.0
NWORKERS = 0
EVENT_CHANNELS = 30
TOTAL_CHANNELS = 36
EXPECTED_FS = 64.0
LABELS = ("A", "B", "C", "D", "E", "X", "G", "H", "I", "J", "K", "L")


def protocol_results_dir(repo_root: Path) -> Path:
    return (
        repo_root
        / "notebooks"
        / "artifacts"
        / EXPERIMENT_ID
        / PROTOCOL_VERSION
    )


def dseed(seed: int, *parts: object) -> int:
    text = "|".join(map(str, (seed, *parts)))
    return int.from_bytes(hashlib.sha256(text.encode()).digest()[:4], "little")


def seed_all(seed: int) -> None:
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


def alpha(shift: int) -> float:
    return float(1.0 - 2.0 ** (-int(shift)))


def tau_ms(shift: int, fs: float) -> float:
    return float(-(1000.0 / fs) / math.log(alpha(shift)))


def tau_table(fs: float) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"shift": shift, "alpha": alpha(shift), "tau_syn_ms": tau_ms(shift, fs)}
            for shift in SHIFTS
        ]
    )


@dataclass
class Data:
    Xtr: np.ndarray
    ytr: np.ndarray
    ltr: np.ndarray
    Xva: np.ndarray
    yva: np.ndarray
    lva: np.ndarray
    Xte: np.ndarray
    yte: np.ndarray
    lte: np.ndarray
    labels: tuple[str, ...]
    fs: float
    T: int
    bin_steps: int
    n_bins: int
    split: dict[str, tuple[str, ...]]


@dataclass(frozen=True)
class Config:
    repo_root: Path
    results_dir: Path
    device: str = "cpu"
    epochs: int = EPOCHS
    batch_size: int = BATCH_SIZE
    resume: bool = True
    threads: int = 1


def prepare_data(repo_root: Path) -> Data:
    roots = [
        repo_root
        / "outputs/action0_wavelets_0e5_1_2_4_8_sr_64/low-pass/aligned-board-events/segmentation_padded",
        repo_root
        / "outputs/action1_wavelets_0e5_1_2_4_8_sr_64/low-pass/aligned-board-events/segmentation_padded",
    ]
    data = load_acceleration_data(
        roots,
        repository_root=repo_root,
        require_reconstruction=False,
    )

    fs_values = {float(metadata.sampling_rate_hz) for metadata in data.producer_metadatas}
    if len(fs_values) != 1:
        raise ValueError(f"Expected one sampling rate, got {fs_values}")
    fs = fs_values.pop()
    if not np.isclose(fs, EXPECTED_FS):
        raise ValueError(f"Expected {EXPECTED_FS} Hz, got {fs}")

    for root, metadata in zip(data.padded_roots, data.producer_metadatas, strict=True):
        raw = metadata.raw
        if raw.get("event_representation") != "unsigned":
            raise ValueError(f"{root}: expected unsigned events")
        if raw.get("event_feature_schema") != "custom_wavelet_polarity_split_abs_events_v1":
            raise ValueError(f"{root}: unexpected event feature schema")
        if raw.get("event_channel_count") != EVENT_CHANNELS:
            raise ValueError(f"{root}: expected {EVENT_CHANNELS} event channels")
        if metadata.channel_count != TOTAL_CHANNELS:
            raise ValueError(f"{root}: expected {TOTAL_CHANNELS} total channels")

    rows: list[tuple[int, int, str, str, int, int]] = []
    keep = set(LABELS)
    for package_index, package in enumerate(data.packages):
        for segment_index, label in enumerate(package.labels.astype(str)):
            if label not in keep:
                continue
            rows.append(
                (
                    package_index,
                    segment_index,
                    str(package.user),
                    str(label),
                    int(package.valid_lengths[segment_index]),
                    int(package.padded_spike_imu.shape[1]),
                )
            )

    manifest = pd.DataFrame(
        rows,
        columns=("pi", "si", "user", "label", "valid", "pad"),
    )
    labels = tuple(sorted(manifest.label.unique().tolist()))
    class_to_idx = {label: idx for idx, label in enumerate(labels)}
    manifest["y"] = manifest.label.map(class_to_idx).astype(int)

    # Match the existing Phase-B / Experiment-3.0 split protocol exactly:
    # seed SPLIT_SEED directly, shuffle sorted users, then floor the fractions.
    users = np.asarray(sorted(manifest.user.unique().tolist()), dtype=object)
    rng = np.random.default_rng(SPLIT_SEED)
    rng.shuffle(users)
    n_users = len(users)
    n_train = max(1, int(np.floor(TRAIN_FRACTION * n_users)))
    n_val = max(1, int(np.floor(VAL_FRACTION * n_users)))
    if n_train + n_val >= n_users:
        n_train, n_val = n_users - 2, 1

    train_users = set(users[:n_train].tolist())
    val_users = set(users[n_train : n_train + n_val].tolist())
    test_users = set(users[n_train + n_val :].tolist())
    parts = [
        manifest[manifest.user.isin(user_set)].reset_index(drop=True)
        for user_set in (train_users, val_users, test_users)
    ]

    original_T = int(manifest.pad.max())
    bin_steps = int(np.rint(FIXED_MS * fs / 1000.0))
    n_bins = int(math.ceil(original_T / bin_steps))
    T = n_bins * bin_steps
    if bin_steps != 16:
        raise ValueError(f"Expected 250 ms = 16 samples at 64 Hz, got {bin_steps}")

    def build_events(frame: pd.DataFrame) -> np.ndarray:
        out = np.zeros((len(frame), T, EVENT_CHANNELS), dtype=np.float32)
        for row_index, row in enumerate(frame.itertuples(index=False)):
            x = np.asarray(
                data.packages[int(row.pi)].padded_spike_imu[
                    int(row.si), :, :EVENT_CHANNELS
                ],
                dtype=np.float32,
            )
            n = min(len(x), T, int(row.valid))
            out[row_index, :n] = x[:n]
        return out

    arrays: list[np.ndarray] = []
    for frame in parts:
        arrays.extend(
            [
                build_events(frame),
                frame.y.to_numpy(dtype=np.int64, copy=True),
                np.minimum(
                    frame.valid.to_numpy(dtype=np.int64, copy=True),
                    T,
                ).copy(),
            ]
        )

    split = {
        "train_users": tuple(sorted(train_users)),
        "val_users": tuple(sorted(val_users)),
        "test_users": tuple(sorted(test_users)),
    }
    return Data(*arrays, labels, fs, T, bin_steps, n_bins, split)


def loader(
    X: np.ndarray,
    y: np.ndarray,
    lengths: np.ndarray,
    batch_size: int,
    shuffle: bool,
    seed: int,
) -> DataLoader:
    # torch.tensor copies the arrays, avoiding undefined behavior from read-only
    # NumPy views produced by some pandas/numpy code paths.
    dataset = TensorDataset(
        torch.tensor(X, dtype=torch.float32),
        torch.tensor(y, dtype=torch.long),
        torch.tensor(lengths, dtype=torch.long),
    )
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=NWORKERS,
        generator=generator,
    )


def mask(lengths: torch.Tensor, T: int) -> torch.Tensor:
    return torch.arange(T, device=lengths.device)[None, :] < lengths[:, None]


def fixed_counts(
    spikes: torch.Tensor,
    lengths: torch.Tensor,
    steps: int,
) -> torch.Tensor:
    batch, T, dim = spikes.shape
    valid = mask(lengths, T).to(spikes.dtype).unsqueeze(-1)
    masked = spikes * valid
    n_bins = int(math.ceil(T / steps))
    masked = F.pad(masked, (0, 0, 0, n_bins * steps - T))
    return masked.reshape(batch, n_bins, steps, dim).sum(dim=2)


def relative_counts(
    spikes: torch.Tensor,
    lengths: torch.Tensor,
    n_bins: int = N_REL,
) -> torch.Tensor:
    batch, T, dim = spikes.shape
    positions = torch.arange(T, device=spikes.device)[None, :].expand(batch, T)
    denominators = lengths.clamp_min(1)[:, None]
    valid = positions < denominators
    bin_index = torch.div(
        positions * n_bins,
        denominators,
        rounding_mode="floor",
    ).clamp(max=n_bins - 1)
    out = torch.zeros(
        batch,
        n_bins,
        dim,
        device=spikes.device,
        dtype=spikes.dtype,
    )
    out.scatter_add_(
        1,
        bin_index.unsqueeze(-1).expand(-1, -1, dim),
        spikes * valid.to(spikes.dtype).unsqueeze(-1),
    )
    return out


class Net(nn.Module):
    def __init__(
        self,
        shift: int,
        objective: str,
        n_classes: int,
        T: int,
        fs: float,
        bin_steps: int,
    ) -> None:
        super().__init__()
        h1, h2, dim = WIDTHS
        alpha_value = alpha(shift)
        beta = math.exp(-(1000.0 / fs) / TAU_MEM_MS)
        spike_grad = surrogate.fast_sigmoid(slope=SURROGATE_SLOPE)

        def make_lif(width: int) -> snn.Synaptic:
            return snn.Synaptic(
                alpha=torch.full((width,), alpha_value),
                beta=beta,
                threshold=THRESHOLD,
                spike_grad=spike_grad,
                reset_mechanism=RESET,
            )

        self.f1 = nn.Linear(EVENT_CHANNELS, h1, bias=False)
        self.l1 = make_lif(h1)
        self.f2 = nn.Linear(h1, h2, bias=False)
        self.l2 = make_lif(h2)
        self.f3 = nn.Linear(h2, dim, bias=False)
        self.l3 = make_lif(dim)
        self.objective = objective
        self.T = T
        self.bin_steps = bin_steps

        if objective == "timestep_ce":
            head_dim = dim
        elif objective == "relative10_sequence_ce":
            head_dim = dim * N_REL
        elif objective == "fixed250_sequence_ce":
            head_dim = dim * int(math.ceil(T / bin_steps))
        else:
            raise ValueError(f"Unknown objective: {objective}")
        self.head = nn.Linear(head_dim, n_classes, bias=True)

    def features(
        self,
        x: torch.Tensor,
        lengths: torch.Tensor,
        stats: bool = False,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor] | None]:
        batch, T, _ = x.shape
        h1, h2, dim = WIDTHS
        syn1 = torch.zeros(batch, h1, device=x.device, dtype=x.dtype)
        mem1 = torch.zeros_like(syn1)
        syn2 = torch.zeros(batch, h2, device=x.device, dtype=x.dtype)
        mem2 = torch.zeros_like(syn2)
        syn3 = torch.zeros(batch, dim, device=x.device, dtype=x.dtype)
        mem3 = torch.zeros_like(syn3)
        sequence: list[torch.Tensor] = []
        spike_sums = torch.zeros(3, device=x.device, dtype=x.dtype)

        for timestep in range(T):
            s1, syn1, mem1 = self.l1(self.f1(x[:, timestep]), syn1, mem1)
            s2, syn2, mem2 = self.l2(self.f2(s1), syn2, mem2)
            s3, syn3, mem3 = self.l3(self.f3(s2), syn3, mem3)
            sequence.append(s3)
            if stats:
                valid_t = (timestep < lengths).to(x.dtype).unsqueeze(-1)
                spike_sums += torch.stack(
                    [
                        (s1 * valid_t).sum(),
                        (s2 * valid_t).sum(),
                        (s3 * valid_t).sum(),
                    ]
                )

        sequence_tensor = torch.stack(sequence, dim=1)
        if not stats:
            return sequence_tensor, None
        return sequence_tensor, {
            "spike_sums": spike_sums,
            "valid_steps": lengths.to(x.dtype).sum(),
        }

    def loss_logits(
        self,
        spikes: torch.Tensor,
        lengths: torch.Tensor,
        y: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch, T, _ = spikes.shape
        if self.objective == "timestep_ce":
            logits_t = self.head(spikes)
            valid = mask(lengths, T)
            targets = y[:, None].expand(batch, T)
            loss = F.cross_entropy(logits_t[valid], targets[valid])
            weights = valid.to(logits_t.dtype).unsqueeze(-1)
            segment_logits = (logits_t * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1)
            return loss, segment_logits

        if self.objective == "relative10_sequence_ce":
            features = relative_counts(spikes, lengths)
        else:
            features = fixed_counts(spikes, lengths, self.bin_steps)
        logits = self.head(features.flatten(start_dim=1))
        return F.cross_entropy(logits, y), logits

    def representation(
        self,
        spikes: torch.Tensor,
        lengths: torch.Tensor,
    ) -> torch.Tensor:
        return fixed_counts(spikes, lengths, self.bin_steps).flatten(start_dim=1)


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
    }


def evaluate(
    model: Net,
    data_loader: DataLoader,
    device: torch.device,
    *,
    stats: bool = False,
    representation: bool = False,
) -> dict[str, object]:
    model.eval()
    y_true_parts: list[np.ndarray] = []
    y_pred_parts: list[np.ndarray] = []
    losses: list[float] = []
    representation_parts: list[np.ndarray] = []
    total_spikes = np.zeros(3, dtype=np.float64)
    total_valid_steps = 0.0

    with torch.no_grad():
        for X, y, lengths in data_loader:
            X = X.to(device)
            y = y.to(device)
            lengths = lengths.to(device)
            spikes, stat = model.features(X, lengths, stats=stats)
            loss, logits = model.loss_logits(spikes, lengths, y)
            y_true_parts.append(y.cpu().numpy())
            y_pred_parts.append(logits.argmax(dim=1).cpu().numpy())
            losses.append(float(loss.item()) * len(y))

            if representation:
                representation_parts.append(
                    model.representation(spikes, lengths).cpu().numpy()
                )
            if stat is not None:
                total_spikes += stat["spike_sums"].cpu().numpy().astype(np.float64)
                total_valid_steps += float(stat["valid_steps"].item())

    y_true = np.concatenate(y_true_parts)
    out: dict[str, object] = metrics(y_true, np.concatenate(y_pred_parts))
    out["loss"] = float(sum(losses) / len(y_true))

    if stats:
        denominator = total_valid_steps * np.asarray(WIDTHS, dtype=np.float64)
        out["rates"] = total_spikes / np.maximum(denominator, 1.0)
    if representation:
        out["rep"] = np.concatenate(representation_parts)
        out["y"] = y_true
    return out


def zero_input_firing_rates(model: Net, T: int, device: torch.device) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        X = torch.zeros(1, T, EVENT_CHANNELS, device=device)
        lengths = torch.tensor([T], dtype=torch.long, device=device)
        _, stat = model.features(X, lengths, stats=True)
    if stat is None:
        raise RuntimeError("Missing zero-input firing statistics")
    spikes = stat["spike_sums"].cpu().numpy().astype(np.float64)
    valid_steps = float(stat["valid_steps"].item())
    return spikes / np.maximum(valid_steps * np.asarray(WIDTHS, dtype=np.float64), 1.0)


def probe(
    train: dict[str, object],
    val: dict[str, object],
    test: dict[str, object],
    seed: int,
) -> dict[str, float]:
    scaler = StandardScaler().fit(train["rep"])
    train_x = scaler.transform(train["rep"])
    val_x = scaler.transform(val["rep"])
    test_x = scaler.transform(test["rep"])

    best: tuple[float, float, LogisticRegression] | None = None
    for C in (1e-3, 1e-2, 1e-1, 1.0, 10.0):
        classifier = LogisticRegression(
            C=C,
            max_iter=5000,
            solver="lbfgs",
            random_state=seed,
        ).fit(train_x, train["y"])
        val_ba = balanced_accuracy_score(val["y"], classifier.predict(val_x))
        if best is None or val_ba > best[0]:
            best = (float(val_ba), float(C), classifier)

    if best is None:
        raise RuntimeError("Linear-probe selection produced no candidate")
    val_ba, C, classifier = best
    test_metrics = metrics(test["y"], classifier.predict(test_x))
    return {
        "probe_C": C,
        "probe_val_balanced_accuracy": val_ba,
        "probe_test_balanced_accuracy": test_metrics["balanced_accuracy"],
        "probe_test_macro_f1": test_metrics["macro_f1"],
        "probe_test_accuracy": test_metrics["accuracy"],
    }


def checkpoint_provenance(
    shift: int,
    objective: str,
    seed: int,
    data: Data,
    config: Config,
) -> dict[str, object]:
    return {
        "experiment_id": EXPERIMENT_ID,
        "protocol_version": PROTOCOL_VERSION,
        "shift": int(shift),
        "objective": str(objective),
        "seed": int(seed),
        "split_seed": SPLIT_SEED,
        "train_users": data.split["train_users"],
        "val_users": data.split["val_users"],
        "test_users": data.split["test_users"],
        "labels": data.labels,
        "sampling_rate_hz": float(data.fs),
        "padded_length": int(data.T),
        "fixed_bin_steps": int(data.bin_steps),
        "widths": WIDTHS,
        "tau_mem_ms": TAU_MEM_MS,
        "threshold": THRESHOLD,
        "surrogate_slope": SURROGATE_SLOPE,
        "reset": RESET,
        "epochs": int(config.epochs),
        "batch_size": int(config.batch_size),
        "learning_rate": LR,
        "relative_bins": N_REL,
        "fixed_bin_ms": FIXED_MS,
        "event_channels": EVENT_CHANNELS,
    }


def validate_checkpoint(
    payload: dict[str, object],
    expected: dict[str, object],
    path: Path,
) -> None:
    actual = payload.get("provenance")
    if actual != expected:
        raise ValueError(
            f"Checkpoint provenance mismatch for {path}. "
            "Use a new protocol results directory or --force-retrain."
        )
    if not isinstance(payload.get("result"), dict):
        raise ValueError(f"Checkpoint has no result dict: {path}")


def run_one(
    shift: int,
    objective: str,
    seed: int,
    data: Data,
    config: Config,
) -> dict[str, object]:
    checkpoint = (
        config.results_dir
        / "checkpoints"
        / f"shift{shift}__{objective}__seed{seed}.pt"
    )
    provenance = checkpoint_provenance(shift, objective, seed, data, config)
    if config.resume and checkpoint.exists():
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        validate_checkpoint(payload, provenance, checkpoint)
        return payload["result"]

    torch.set_num_threads(config.threads)
    device = torch.device(config.device)

    # Paired design: for a given master seed, all shifts and all objectives
    # receive exactly the same initial f1/f2/f3 backbone weights.
    seed_all(dseed(seed, "shared_backbone_init"))
    model = Net(
        shift,
        objective,
        len(data.labels),
        data.T,
        data.fs,
        data.bin_steps,
    ).to(device)
    # Keep the objective-specific head deterministic without changing the
    # already-created paired backbone.
    seed_all(dseed(seed, objective, "head_init"))
    model.head.reset_parameters()

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    partitions = [
        (data.Xtr, data.ytr, data.ltr),
        (data.Xva, data.yva, data.lva),
        (data.Xte, data.yte, data.lte),
    ]
    train_loader = loader(
        *partitions[0],
        config.batch_size,
        True,
        dseed(seed, "train", "loader"),
    )
    eval_loaders = [
        loader(*partition, config.batch_size, False, dseed(seed, name, "loader"))
        for partition, name in zip(partitions, ("train", "val", "test"), strict=True)
    ]

    best_val_ba = -np.inf
    best_val_loss = np.inf
    best_state: dict[str, torch.Tensor] | None = None
    best_epoch = -1
    history: list[dict[str, float | int]] = []

    for epoch in range(1, config.epochs + 1):
        model.train()
        train_true: list[np.ndarray] = []
        train_pred: list[np.ndarray] = []
        train_loss_sum = 0.0

        for X, y, lengths in train_loader:
            X = X.to(device)
            y = y.to(device)
            lengths = lengths.to(device)
            optimizer.zero_grad(set_to_none=True)
            spikes, _ = model.features(X, lengths)
            loss, logits = model.loss_logits(spikes, lengths, y)
            loss.backward()
            optimizer.step()
            train_loss_sum += float(loss.item()) * len(y)
            train_true.append(y.cpu().numpy())
            train_pred.append(logits.detach().argmax(dim=1).cpu().numpy())

        train_metrics = metrics(np.concatenate(train_true), np.concatenate(train_pred))
        val_metrics = evaluate(model, eval_loaders[1], device)
        val_ba = float(val_metrics["balanced_accuracy"])
        val_loss = float(val_metrics["loss"])
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss_sum / len(data.ytr),
                "train_balanced_accuracy": train_metrics["balanced_accuracy"],
                "val_loss": val_loss,
                "val_balanced_accuracy": val_ba,
            }
        )

        improved = (
            val_ba > best_val_ba + 1e-12
            or (
                abs(val_ba - best_val_ba) <= 1e-12
                and val_loss < best_val_loss - 1e-12
            )
        )
        if improved:
            best_val_ba = val_ba
            best_val_loss = val_loss
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())

    if best_state is None:
        raise RuntimeError("No best checkpoint was selected")
    model.load_state_dict(best_state)

    final = [
        evaluate(model, eval_loader, device, stats=True, representation=True)
        for eval_loader in eval_loaders
    ]
    probe_metrics = probe(
        final[0],
        final[1],
        final[2],
        dseed(seed, "probe", shift, objective),
    )
    zero_rates = zero_input_firing_rates(model, data.T, device)

    result: dict[str, object] = {
        "protocol_version": PROTOCOL_VERSION,
        "shift": shift,
        "tau_syn_ms": tau_ms(shift, data.fs),
        "objective": objective,
        "seed": seed,
        "best_epoch": best_epoch,
        "history": history,
        **probe_metrics,
    }
    for name, partition_metrics in zip(("train", "val", "test"), final, strict=True):
        for key in ("loss", "accuracy", "balanced_accuracy", "macro_f1"):
            result[f"{name}_{key}"] = float(partition_metrics[key])

    result["train_test_ba_gap"] = (
        result["train_balanced_accuracy"] - result["test_balanced_accuracy"]
    )
    for layer_index in range(3):
        result[f"test_l{layer_index + 1}_firing_rate"] = float(
            final[2]["rates"][layer_index]
        )
        result[f"zero_l{layer_index + 1}_firing_rate"] = float(zero_rates[layer_index])

    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "experiment_id": EXPERIMENT_ID,
            "protocol_version": PROTOCOL_VERSION,
            "provenance": provenance,
            "result": result,
            "state_dict": best_state,
        },
        checkpoint,
    )
    return result


def run_sweep(data: Data, config: Config, n_jobs: int = 1) -> pd.DataFrame:
    specs = [
        (shift, objective, seed)
        for shift in SHIFTS
        for objective in OBJECTIVES
        for seed in SEEDS
    ]
    if n_jobs > 1 and config.device == "cpu":
        from joblib import Parallel, delayed, parallel_config

        with parallel_config(
            backend="loky",
            n_jobs=n_jobs,
            inner_max_num_threads=config.threads,
        ):
            results = Parallel()(
                delayed(run_one)(shift, objective, seed, data, config)
                for shift, objective, seed in specs
            )
    else:
        results = [
            run_one(shift, objective, seed, data, config)
            for shift, objective, seed in specs
        ]

    rows = [{key: value for key, value in result.items() if key != "history"} for result in results]
    frame = pd.DataFrame(rows).sort_values(["objective", "shift", "seed"])
    config.results_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(config.results_dir / "experiment_3_0_1_results.csv", index=False)
    pd.DataFrame(
        [
            {
                "shift": result["shift"],
                "objective": result["objective"],
                "seed": result["seed"],
                **epoch_row,
            }
            for result in results
            for epoch_row in result["history"]
        ]
    ).to_csv(config.results_dir / "experiment_3_0_1_history.csv", index=False)
    return frame


def aggregate(frame: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "test_balanced_accuracy",
        "test_macro_f1",
        "test_accuracy",
        "probe_test_balanced_accuracy",
        "probe_test_macro_f1",
        "val_balanced_accuracy",
        "train_test_ba_gap",
        "best_epoch",
        "test_l1_firing_rate",
        "test_l2_firing_rate",
        "test_l3_firing_rate",
    ]
    grouped = frame.groupby(
        ["objective", "shift", "tau_syn_ms"],
        as_index=False,
    )[columns]
    mean = grouped.mean().rename(columns={column: f"mean_{column}" for column in columns})
    std = grouped.std(ddof=1).rename(columns={column: f"sd_{column}" for column in columns})
    return mean.merge(std, on=["objective", "shift", "tau_syn_ms"])
