from __future__ import annotations

import csv
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import sys
from typing import Any, Mapping, Sequence

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from writingring.recording_features import (  # noqa: E402
    POLARITY_SPLIT_WAVELET_SPIKE_IMU_FEATURE_SCHEMA,
    SPIKE_IMU_TRANSIENT_CHANNEL_NAMES,
)
from writingring.spike_encoding.encoders.custom_wavelet import (  # noqa: E402
    CustomWaveletEncoder,
    CustomWaveletSettings,
)

RATE_HZ = 64.0
FREQUENCIES_HZ = (0.5, 1.0, 2.0, 4.0, 8.0)
MAX_FILTER_TIME_S = 0.3
SOURCE_CHANNELS = 36
SOURCE_EVENTS = 30
OUTPUT_EVENTS = 60
OUTPUT_CHANNELS = 66
OUTPUT_SCHEMA = "linear_accel_angular_accel_polarity_split_wavelet_events_plus_imu_v1"
OUTPUT_EVENT_SCHEMA = "linear_accel_angular_accel_polarity_split_abs_events_v1"
OUTPUT_ENCODER_SCHEMA = "linear_angular_accel_wavelet_encoder_spec_v1"
DERIVATIVE_METHOD = "central_difference_numpy_gradient_v1"


class BuildError(RuntimeError):
    pass


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def canonical_sha256(value: Mapping[str, Any]) -> str:
    raw = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    )
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise BuildError(f"could not read JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise BuildError(f"JSON root must be an object: {path}")
    return value


def write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            raise BuildError(f"CSV has no header: {path}")
        return list(reader.fieldnames), [dict(row) for row in reader]


def write_csv(
    path: Path, fieldnames: Sequence[str], rows: Sequence[Mapping[str, Any]]
) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(fieldnames))
        writer.writeheader()
        writer.writerows(rows)


def publish_dir(staging: Path, destination: Path, *, overwrite: bool) -> None:
    if not destination.exists():
        os.replace(staging, destination)
        return
    if not overwrite:
        raise BuildError(f"destination exists: {destination}; use --overwrite")
    backup = destination.with_name(f".{destination.name}.angular66.backup")
    if backup.exists():
        shutil.rmtree(backup)
    os.replace(destination, backup)
    try:
        os.replace(staging, destination)
    except Exception:
        os.replace(backup, destination)
        raise
    shutil.rmtree(backup, ignore_errors=True)


def spike_root(root: Path) -> Path:
    return root / "spikeEncoding" / "custom-wavelet"


def _user_sort_key(user: str) -> tuple[str, int, str]:
    prefix, sep, suffix = user.rpartition("_")
    return (prefix, int(suffix), user) if sep and suffix.isdigit() else (user, -1, user)


def list_users(source_root: Path) -> tuple[str, ...]:
    root = spike_root(source_root)
    if not root.is_dir():
        raise BuildError(f"missing source spike root: {root}")
    users: set[str] = set()
    for path in root.rglob("metadata.json"):
        parts = path.parent.relative_to(root).parts
        if len(parts) >= 3:
            users.add(parts[-3])
    if not users:
        raise BuildError(f"no recordings found below {root}")
    return tuple(sorted(users, key=_user_sort_key))


def infer_action(source_root: Path) -> str:
    root = spike_root(source_root)
    actions: set[str] = set()
    for path in root.rglob("metadata.json"):
        parts = path.parent.relative_to(root).parts
        if len(parts) >= 3:
            actions.add(parts[-2].removeprefix("action_"))
    if len(actions) != 1:
        raise BuildError(f"expected exactly one action; found {sorted(actions)}")
    return next(iter(actions))


def angular_acceleration_from_gyro(
    gyro_rad_s: np.ndarray, *, sampling_rate_hz: float = RATE_HZ
) -> np.ndarray:
    gyro = np.asarray(gyro_rad_s, dtype=np.float64)
    if (
        gyro.ndim != 2
        or gyro.shape[1] != 3
        or len(gyro) < 3
        or not np.isfinite(gyro).all()
    ):
        raise BuildError("gyroscope must be finite shape (N, 3), N >= 3")
    rate = float(sampling_rate_hz)
    if not math.isfinite(rate) or rate <= 0:
        raise BuildError("sampling_rate_hz must be finite and positive")
    result = np.gradient(gyro, 1.0 / rate, axis=0, edge_order=2)
    if result.shape != gyro.shape or not np.isfinite(result).all():
        raise BuildError("angular acceleration derivative is invalid")
    return np.asarray(result, dtype=np.float64)


def combine_event_branches(
    source_spike_imu: np.ndarray, angular_events: np.ndarray
) -> np.ndarray:
    source = np.asarray(source_spike_imu)
    angular = np.asarray(angular_events)
    if source.ndim != 2 or source.shape[1] != SOURCE_CHANNELS:
        raise BuildError("source SpikeIMU must have shape (N, 36)")
    if angular.shape != (len(source), 30):
        raise BuildError("angular event matrix must have shape (N, 30)")
    if not np.isfinite(source).all() or not np.isfinite(angular).all():
        raise BuildError("feature matrices must be finite")
    if np.any(source[:, :30] < 0) or np.any(angular < 0):
        raise BuildError("PolaritySplitAbs event channels must be non-negative")
    output = np.empty((len(source), OUTPUT_CHANNELS), dtype=source.dtype)
    output[:, :30] = source[:, :30]
    output[:, 30:60] = angular.astype(source.dtype, copy=False)
    output[:, 60:] = source[:, 30:]
    return output


def angular_encoder(dtype: str) -> CustomWaveletEncoder:
    return CustomWaveletEncoder(
        CustomWaveletSettings(
            frequencies_hz=FREQUENCIES_HZ,
            sampling_rate_hz=RATE_HZ,
            max_filter_time_s=MAX_FILTER_TIME_S,
            output_dtype=dtype,
            post_encode_transform="PolaritySplitAbs",
        )
    )


def source_encoder(metadata: Mapping[str, Any]) -> tuple[dict[str, Any], str]:
    encoder = metadata.get("encoder")
    if not isinstance(encoder, Mapping):
        raise BuildError("source metadata lacks encoder")
    spec = encoder.get("spike_encoder")
    digest = encoder.get("spike_encoder_spec_sha256")
    if (
        not isinstance(spec, Mapping)
        or not isinstance(digest, str)
        or canonical_sha256(spec) != digest
    ):
        raise BuildError("source encoder identity is missing or invalid")
    return dict(spec), digest


def validate_source_recording(
    directory: Path,
) -> tuple[np.ndarray, dict[str, Any], dict[str, Any], str, list[str], list[str]]:
    values_path = directory / "spikeIMU.npy"
    metadata_path = directory / "metadata.json"
    metadata = load_json(metadata_path)
    section = metadata.get("spike_imu")
    if not isinstance(section, Mapping):
        raise BuildError(f"source metadata lacks spike_imu: {metadata_path}")
    if section.get("schema") != POLARITY_SPLIT_WAVELET_SPIKE_IMU_FEATURE_SCHEMA:
        raise BuildError("source must use polarity_split_wavelet_events_plus_imu_v1")
    if (
        section.get("channel_count") != 36
        or section.get("event_channel_count") != 30
        or section.get("event_representation") != "unsigned"
    ):
        raise BuildError("source must be the canonical 30-event + 6-IMU layout")
    if not math.isclose(
        float(metadata.get("sampling_rate_hz", 0.0)), RATE_HZ, rel_tol=0.0, abs_tol=1e-12
    ):
        raise BuildError("source sampling_rate_hz must be 64")
    spec, digest = source_encoder(metadata)
    if tuple(float(v) for v in spec.get("frequencies_hz", ())) != FREQUENCIES_HZ:
        raise BuildError("source wavelet frequencies must be 0.5,1,2,4,8 Hz")
    if not math.isclose(
        float(spec.get("max_filter_time_s", 0.0)), MAX_FILTER_TIME_S,
        rel_tol=0.0, abs_tol=1e-12,
    ):
        raise BuildError("source max_filter_time_s must be 0.3")
    if spec.get("post_encode_transform") != "PolaritySplitAbs":
        raise BuildError("source must use PolaritySplitAbs")
    values = np.load(values_path, allow_pickle=False, mmap_mode="r")
    if (
        values.ndim != 2
        or values.shape[1] != 36
        or len(values) < 3
        or not np.isfinite(values).all()
    ):
        raise BuildError(f"invalid source SpikeIMU: {values_path}")
    names, units = section.get("channel_names"), section.get("units")
    if (
        not isinstance(names, list)
        or len(names) != 36
        or not isinstance(units, list)
        or len(units) != 36
    ):
        raise BuildError("source channel_names/units must contain 36 entries")
    if tuple(names[-6:]) != SPIKE_IMU_TRANSIENT_CHANNEL_NAMES:
        raise BuildError("source trailing six channel names are not canonical")
    if tuple(units[-6:]) != ("m/s^2",) * 3 + ("rad/s",) * 3:
        raise BuildError("source trailing six units are not canonical")
    declared = section.get("sha256")
    if isinstance(declared, str) and declared != sha256_file(values_path):
        raise BuildError("source SpikeIMU SHA-256 mismatch")
    return values, metadata, spec, digest, names, units
