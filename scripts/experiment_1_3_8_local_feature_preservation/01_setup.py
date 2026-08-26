from __future__ import annotations
from pathlib import Path
import copy, hashlib, math, os, random, sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
import torch.nn.functional as F
import snntorch as snn
from snntorch import surrogate
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
from IPython.display import display

def find_repo_root(start: Path | None = None) -> Path:
    start = (start or Path.cwd()).resolve()
    for candidate in (start, *start.parents):
        if (candidate / "snn").is_dir() and (candidate / "notebooks").is_dir():
            return candidate
    raise FileNotFoundError("Could not locate writingRing repository root")

REPO_ROOT = find_repo_root()
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
from snn.accel_reconstruction_eval.datasets import load_acceleration_data

ACTION0_DIR = REPO_ROOT / "outputs/action0_wavelets_0e5_1_2_4_8_sr_64"
ACTION1_DIR = REPO_ROOT / "outputs/action1_wavelets_0e5_1_2_4_8_sr_64"
DATASET_ROOTS = [
    ACTION0_DIR / "low-pass/aligned-board-events/segmentation_padded",
    ACTION1_DIR / "low-pass/aligned-board-events/segmentation_padded",
]

EXPECTED_EVENT_REPRESENTATION = "unsigned"
EXPECTED_EVENT_FEATURE_SCHEMA = "custom_wavelet_polarity_split_abs_events_v1"
EVENT_CHANNEL_COUNT = 30
TOTAL_CHANNEL_COUNT = 36
EXPECTED_SAMPLING_RATE_HZ = 64.0
INCLUDED_LABELS = ("A","B","C","D","E","X","G","H","I","J","K","L")

SPLIT_SEED = 12345
TRAIN_FRACTION = 0.70
VAL_FRACTION = 0.15
SEEDS = (11, 23, 101)
DEV_SEED = 11

FIXED_BIN_MS = 250.0
HALF_BIN_MS = 125.0
SNN_LAYER_WIDTHS = (128, 128, 64)
SNN_LAYER_SHIFTS = ((2, 3), (2, 3), (2,))
TAU_MEM_MS = 22.54
THRESHOLD = 0.5
SURROGATE_SLOPE = 25.0
RESET_MECHANISM = "subtract"

BATCH_SIZE = 64
NUM_WORKERS = 0
NUM_EPOCHS = 100
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 0.0
GRAD_CLIP_NORM = None

LAMBDA_COUNT_GRID = (0.05, 0.10, 0.30)
LAMBDA_TEMP_GRID = (0.05, 0.10, 0.30)
LOGREG_C_GRID = (1e-3, 1e-2, 1e-1, 1.0, 10.0)
LOGREG_MAX_ITER = 5000

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EXPERIMENT_ID = "experiment_1_3_8_snn_local_feature_preservation"
RESULTS_DIR = REPO_ROOT / "notebooks/artifacts" / EXPERIMENT_ID
DEV_DIR = RESULTS_DIR / "development"
FINAL_DIR = RESULTS_DIR / "final"
for p in (RESULTS_DIR, DEV_DIR, FINAL_DIR):
    p.mkdir(parents=True, exist_ok=True)

RESUME_EXISTING = True
SAVE_CHECKPOINTS = True
EPS = 1e-6

print("Repository root:", REPO_ROOT)
print("Device:", DEVICE)
print("Experiment:", EXPERIMENT_ID)

def derive_seed(master_seed: int, *parts: object) -> int:
    text = "|".join([str(master_seed), *(str(p) for p in parts)])
    return int.from_bytes(hashlib.sha256(text.encode()).digest()[:4], "little")

def seed_everything(seed: int):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except TypeError:
        torch.use_deterministic_algorithms(True)

def worker_init_fn(worker_id: int):
    s = torch.initial_seed() % (2**32)
    random.seed(s); np.random.seed(s)

def shift_to_alpha(shift: int) -> float:
    return float(1.0 - 2.0 ** (-int(shift)))

def allocate_neurons(width: int, shifts: tuple[int, ...]) -> tuple[int, ...]:
    base, remainder = divmod(width, len(shifts))
    counts = [base] * len(shifts)
    order, left, right = [], 0, len(shifts) - 1
    while left <= right:
        order.append(left)
        if left != right:
            order.append(right)
        left += 1; right -= 1
    for i in range(remainder):
        counts[order[i]] += 1
    return tuple(counts)

def alpha_vector(width: int, shifts: tuple[int, ...]) -> torch.Tensor:
    vals = []
    for shift, n in zip(shifts, allocate_neurons(width, shifts), strict=True):
        vals.extend([shift_to_alpha(shift)] * n)
    return torch.tensor(vals, dtype=torch.float32)

seed_everything(2026)

data = load_acceleration_data(DATASET_ROOTS, repository_root=REPO_ROOT, require_reconstruction=False)

fs_values = {float(m.sampling_rate_hz) for m in data.producer_metadatas}
if len(fs_values) != 1:
    raise ValueError(f"Expected one shared sampling rate, got {fs_values}")
SAMPLING_RATE_HZ = fs_values.pop()
if not np.isclose(SAMPLING_RATE_HZ, EXPECTED_SAMPLING_RATE_HZ):
    raise ValueError((SAMPLING_RATE_HZ, EXPECTED_SAMPLING_RATE_HZ))

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

rows = []
keep = set(INCLUDED_LABELS)
for package_index, package in enumerate(data.packages):
    for segment_index, label in enumerate(package.labels.astype(str)):
        if label in keep:
            rows.append({
                "package_index": package_index,
                "segment_index": segment_index,
                "user": str(package.user),
                "action": str(package.action),
                "label": str(label),
                "valid_length": int(package.valid_lengths[segment_index]),
                "package_padded_length": int(package.padded_spike_imu.shape[1]),
                "sample_id": f"{package.user}/action_{package.action}/{segment_index}",
            })

manifest = pd.DataFrame(rows)
labels_sorted = sorted(manifest.label.unique().tolist())
CLASS_TO_IDX = {lab: i for i, lab in enumerate(labels_sorted)}
manifest["label_idx"] = manifest.label.map(CLASS_TO_IDX).astype(int)
N_CLASSES = len(labels_sorted)

GLOBAL_PADDED_LENGTH = int(manifest.package_padded_length.max())
BIN_SAMPLES = int(np.rint(FIXED_BIN_MS * SAMPLING_RATE_HZ / 1000.0))
HALF_BIN_SAMPLES = int(np.rint(HALF_BIN_MS * SAMPLING_RATE_HZ / 1000.0))
N_BINS = int(math.ceil(GLOBAL_PADDED_LENGTH / BIN_SAMPLES))
PADDED_LENGTH = N_BINS * BIN_SAMPLES
if BIN_SAMPLES != 16 or HALF_BIN_SAMPLES != 8 or PADDED_LENGTH != 256:
    raise ValueError((BIN_SAMPLES, HALF_BIN_SAMPLES, PADDED_LENGTH))

def make_user_split(split_seed: int):
    users = np.asarray(sorted(manifest.user.unique()), dtype=object)
    rng = np.random.default_rng(derive_seed(split_seed, "user_split"))
    rng.shuffle(users)
    n_train = int(round(TRAIN_FRACTION * len(users)))
    n_val = int(round(VAL_FRACTION * len(users)))
    train_users = set(users[:n_train])
    val_users = set(users[n_train:n_train+n_val])
    test_users = set(users[n_train+n_val:])
    part = lambda us: manifest[manifest.user.isin(us)].reset_index(drop=True)
    return part(train_users), part(val_users), part(test_users), {
        "train_users": tuple(sorted(train_users)),
        "val_users": tuple(sorted(val_users)),
        "test_users": tuple(sorted(test_users)),
    }

def build_events(df: pd.DataFrame) -> np.ndarray:
    out = np.zeros((len(df), PADDED_LENGTH, EVENT_CHANNEL_COUNT), dtype=np.float32)
    for i, row in enumerate(df.itertuples(index=False)):
        package = data.packages[int(row.package_index)]
        x = np.asarray(
            package.padded_spike_imu[int(row.segment_index), :, :EVENT_CHANNEL_COUNT],
            dtype=np.float32,
        )
        n = min(len(x), PADDED_LENGTH)
        out[i, :n] = x[:n]
    return out

train_df, val_df, test_df, split_info = make_user_split(SPLIT_SEED)
X_train, X_val, X_test = map(build_events, (train_df, val_df, test_df))
y_train = train_df.label_idx.to_numpy(np.int64)
y_val = val_df.label_idx.to_numpy(np.int64)
y_test = test_df.label_idx.to_numpy(np.int64)
valid_train = np.minimum(train_df.valid_length.to_numpy(np.int64), PADDED_LENGTH)
valid_val = np.minimum(val_df.valid_length.to_numpy(np.int64), PADDED_LENGTH)
valid_test = np.minimum(test_df.valid_length.to_numpy(np.int64), PADDED_LENGTH)

print(f"samples={len(manifest)}, users={manifest.user.nunique()}, classes={N_CLASSES}")
print("labels:", labels_sorted)
print("train/val/test:", X_train.shape, X_val.shape, X_test.shape)
print("readout:", BIN_SAMPLES, "samples =", 1000.0 * BIN_SAMPLES / SAMPLING_RATE_HZ, "ms")
print(split_info)

def build_aux_targets(X: np.ndarray, valid_lengths: np.ndarray):
    N, T, C = X.shape
    windows = X.reshape(N, N_BINS, BIN_SAMPLES, C)
    raw_count = windows.sum(axis=2).astype(np.float32)
    early = windows[:, :, :HALF_BIN_SAMPLES].sum(axis=2)
    late = windows[:, :, HALF_BIN_SAMPLES:].sum(axis=2)
    total = early + late
    temporal = ((early - late) / (total + EPS)).astype(np.float32)

    starts = np.arange(N_BINS, dtype=np.int64) * BIN_SAMPLES
    ends = starts + BIN_SAMPLES
    valid_fraction = np.clip(
        (valid_lengths[:, None] - starts[None, :]) / float(BIN_SAMPLES), 0.0, 1.0
    ).astype(np.float32)
    full_bin = valid_lengths[:, None] >= ends[None, :]
    active_channel = total > 0
    temporal_mask = (full_bin[:, :, None] & active_channel).astype(np.float32)

    return {
        "raw_count": raw_count,
        "count_log": np.log1p(raw_count).astype(np.float32),
        "count_weight": valid_fraction,
        "temporal": temporal,
        "temporal_mask": temporal_mask,
    }

targets_train = build_aux_targets(X_train, valid_train)
targets_val = build_aux_targets(X_val, valid_val)
targets_test = build_aux_targets(X_test, valid_test)

def weighted_channel_stats(values: np.ndarray, bin_weights: np.ndarray):
    flat = values.reshape(-1, values.shape[-1]).astype(np.float64)
    w = bin_weights.reshape(-1).astype(np.float64)
    denom = max(float(w.sum()), EPS)
    mu = (flat * w[:, None]).sum(axis=0) / denom
    var = (((flat - mu[None, :]) ** 2) * w[:, None]).sum(axis=0) / denom
    sigma = np.sqrt(np.maximum(var, 1e-8))
    return mu.astype(np.float32), sigma.astype(np.float32)

COUNT_LOG_MEAN, COUNT_LOG_STD = weighted_channel_stats(
    targets_train["count_log"], targets_train["count_weight"]
)

def normalize_count_target(target_dict):
    return (
        (target_dict["count_log"] - COUNT_LOG_MEAN[None, None, :])
        / (COUNT_LOG_STD[None, None, :] + EPS)
    ).astype(np.float32)

count_norm_train = normalize_count_target(targets_train)
count_norm_val = normalize_count_target(targets_val)
count_norm_test = normalize_count_target(targets_test)

print("semantic target mean range:", float(COUNT_LOG_MEAN.min()), float(COUNT_LOG_MEAN.max()))
print("semantic target std range:", float(COUNT_LOG_STD.min()), float(COUNT_LOG_STD.max()))
print("train temporal active elements:", int(targets_train["temporal_mask"].sum()))
