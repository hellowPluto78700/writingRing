"""Validate, analyze, and right-pad completed variable-length IMU exports.

This module deliberately consumes only the files written by the segmentation
exporters.  It neither discovers recordings nor reads raw Ring, Board, label,
or alignment inputs.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any, Final, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd


DEFAULT_CANDIDATE_LENGTHS: Final[tuple[int, ...]] = (
    256, 320, 400, 480, 512, 600, 640, 768, 800, 1024,
)
DEFAULT_SAMPLING_RATE_HZ: Final[float] = 200.0


class SegmentPaddingError(ValueError):
    """Raised when a completed segmentation export cannot be safely used."""


@dataclass(frozen=True, slots=True)
class SegmentedDatasetPaths:
    """All known paths belonging to one variable-length user/action export."""

    input_dir: Path
    user: str
    action: str
    stem: str
    raw_imu_path: Path
    labels_path: Path
    segment_offsets_path: Path
    segment_lengths_path: Path
    manifest_path: Path | None
    board_event_targets_path: Path | None


@dataclass(frozen=True, slots=True)
class SegmentLengthRecord:
    """One globally addressable segment-length observation."""

    global_segment_index: int
    local_segment_index: int
    user: str
    action: str
    dataset_id: int | None
    label: str
    sample_count: int
    source_directory: Path


@dataclass(frozen=True, slots=True)
class ValidatedSegmentedDataset:
    """Validated arrays and metadata from a single segmentation export."""

    paths: SegmentedDatasetPaths
    raw_imu: np.ndarray
    labels: np.ndarray
    segment_offsets: np.ndarray
    segment_lengths: np.ndarray
    board_event_targets: np.ndarray | None
    manifest: pd.DataFrame | None


@dataclass(frozen=True, slots=True)
class PaddingPackageResult:
    """Generated arrays and audit rows for one fixed-length package."""

    dataset: ValidatedSegmentedDataset
    padded_imu: np.ndarray
    labels: np.ndarray
    valid_lengths: np.ndarray
    valid_mask: np.ndarray
    padded_board_event_targets: np.ndarray | None
    manifest: pd.DataFrame
    summary: dict[str, object]


def discover_segmented_datasets(input_root: Path) -> tuple[SegmentedDatasetPaths, ...]:
    """Find complete variable-length exports below ``input_root``.

    The discovery key is the length-array path, so a missing sibling is a
    fatal error rather than an opportunity to silently skip a user/action.
    """

    root = Path(input_root)
    if not root.is_dir():
        raise SegmentPaddingError(f"input root is not a directory: {root}")
    found: list[SegmentedDatasetPaths] = []
    seen_stems: set[tuple[Path, str]] = set()
    for lengths_path in sorted(root.rglob("*_segment_lengths.npy")):
        if _is_generated_directory(lengths_path.relative_to(root).parts):
            continue
        stem = lengths_path.name.removesuffix("_segment_lengths.npy")
        directory = lengths_path.parent
        try:
            user, action = _identity_from_directory(directory, stem=stem)
        except SegmentPaddingError:
            raise
        key = (directory, stem)
        if key in seen_stems:
            raise SegmentPaddingError(f"duplicate segmentation stem {stem!r} in {directory}")
        seen_stems.add(key)
        raw = directory / f"{stem}_rawIMU.npy"
        labels = directory / f"{stem}_labels.npy"
        offsets = directory / f"{stem}_segment_offsets.npy"
        missing = [path for path in (raw, labels, offsets) if not path.is_file()]
        if missing:
            raise SegmentPaddingError(
                f"user={user!r}, action={action!r}, lengths={lengths_path}: "
                "missing required sibling file(s): " + ", ".join(map(str, missing))
            )
        manifest = directory / f"{stem}_segments.csv"
        targets = directory / f"{stem}_board_event_targets.npy"
        found.append(
            SegmentedDatasetPaths(
                input_dir=directory, user=user, action=action, stem=stem,
                raw_imu_path=raw, labels_path=labels,
                segment_offsets_path=offsets, segment_lengths_path=lengths_path,
                manifest_path=manifest if manifest.is_file() else None,
                board_event_targets_path=targets if targets.is_file() else None,
            )
        )
    if not found:
        raise SegmentPaddingError(f"no *_segment_lengths.npy files found below {root}")
    return tuple(found)


def load_and_validate_segmented_dataset(paths: SegmentedDatasetPaths) -> ValidatedSegmentedDataset:
    """Load one package without pickle support and enforce array invariants."""

    raw_imu = _load_array(paths.raw_imu_path, paths=paths, expected="a 2-D (N, 6) array")
    labels = _load_array(paths.labels_path, paths=paths, expected="a 1-D labels array")
    offsets = _load_array(paths.segment_offsets_path, paths=paths, expected="a 1-D offsets array")
    lengths = _load_array(paths.segment_lengths_path, paths=paths, expected="a 1-D lengths array")
    _validate_dataset_arrays(paths, raw_imu, labels, offsets, lengths)
    targets: np.ndarray | None = None
    if paths.board_event_targets_path is not None:
        targets = _load_array(
            paths.board_event_targets_path, paths=paths, expected="a 2-D (N, 4) Board-target array"
        )
        actual = tuple(targets.shape)
        expected = (len(raw_imu), 4)
        if targets.ndim != 2 or actual != expected:
            _shape_error(paths, paths.board_event_targets_path, expected, actual)
        if not np.can_cast(targets.dtype, np.bool_, casting="safe"):
            raise SegmentPaddingError(
                f"user={paths.user!r}, action={paths.action!r}, file={paths.board_event_targets_path}: "
                f"expected dtype safely castable to bool, actual dtype={targets.dtype}"
            )
        targets = targets.astype(np.bool_, copy=False)
    manifest = _load_manifest(paths)
    return ValidatedSegmentedDataset(
        paths=paths, raw_imu=raw_imu, labels=labels, segment_offsets=offsets,
        segment_lengths=lengths, board_event_targets=targets, manifest=manifest,
    )


def validate_segmented_root(input_root: Path) -> tuple[ValidatedSegmentedDataset, ...]:
    """Discover and fully validate every package before any output is written."""

    return tuple(load_and_validate_segmented_dataset(item) for item in discover_segmented_datasets(input_root))


def collect_segment_length_records(
    datasets: Sequence[ValidatedSegmentedDataset],
) -> tuple[SegmentLengthRecord, ...]:
    """Return lengths with label and manifest provenance, in deterministic order."""

    records: list[SegmentLengthRecord] = []
    for dataset in datasets:
        metadata = _manifest_metadata_by_index(dataset)
        for local_index, length in enumerate(dataset.segment_lengths):
            row = metadata.get(local_index, {})
            dataset_id = _optional_int(row.get("dataset_id"))
            label = str(dataset.labels[local_index])
            records.append(
                SegmentLengthRecord(
                    global_segment_index=len(records), local_segment_index=local_index,
                    user=dataset.paths.user, action=dataset.paths.action,
                    dataset_id=dataset_id, label=label, sample_count=int(length),
                    source_directory=dataset.paths.input_dir,
                )
            )
    if not records:
        raise SegmentPaddingError("segmentation root contains no segments")
    return tuple(records)


def analyze_segment_lengths(
    datasets: Sequence[ValidatedSegmentedDataset],
    *,
    sampling_rate_hz: float = DEFAULT_SAMPLING_RATE_HZ,
    candidate_lengths: Sequence[int] = DEFAULT_CANDIDATE_LENGTHS,
    minimum_coverage: float = 0.99,
    round_to: int = 1,
) -> dict[str, object]:
    """Compute length statistics, candidate tradeoffs, recommendations, and outliers."""

    rate = _positive_finite(sampling_rate_hz, name="sampling_rate_hz")
    candidates = _positive_integer_sequence(candidate_lengths, name="candidate_lengths")
    if not 0.0 < minimum_coverage <= 1.0:
        raise SegmentPaddingError("minimum_coverage must be in (0, 1]")
    if not isinstance(round_to, int) or isinstance(round_to, bool) or round_to <= 0:
        raise SegmentPaddingError("round_to must be a positive integer")
    records = collect_segment_length_records(datasets)
    lengths = np.asarray([record.sample_count for record in records], dtype=np.int64)
    stats = _length_statistics(lengths)
    maximum = int(stats["maximum"])
    candidates_rows = [_candidate_row(lengths, target) for target in candidates]
    pure_target = _round_up(maximum, round_to)
    p99_target = _round_up(int(math.ceil(float(stats["p99"]))), round_to)
    balanced = next((row for row in candidates_rows if row["coverage"] >= minimum_coverage), None)
    recommendations: dict[str, object] = {
        "pure_padding": _recommendation(lengths, pure_target),
        "p99": _recommendation(lengths, p99_target),
        "balanced": (
            None if balanced is None else {
                "target_length": balanced["target_length"], "coverage": balanced["coverage"],
                "overflow_count": balanced["overflow_count"],
                "requires_overflow_handling": balanced["overflow_count"] > 0,
            }
        ),
    }
    p75, p99, q1, q3 = (float(stats[key]) for key in ("p75", "p99", "p25", "p75"))
    iqr_threshold = q3 + 3.0 * (q3 - q1)
    quantile_outliers = [record for record in records if record.sample_count > p99]
    iqr_outliers = [record for record in records if record.sample_count > iqr_threshold]
    return {
        "sampling_rate_hz": rate,
        "segment_count": len(records),
        "user_action_count": len(datasets),
        "length_statistics": stats,
        "candidate_lengths": candidates_rows,
        "minimum_coverage": minimum_coverage,
        "round_to": round_to,
        "recommendations": recommendations,
        "outlier_thresholds": {"quantile_p99": p99, "iqr": iqr_threshold},
        "records": records,
        "quantile_outliers": quantile_outliers,
        "iqr_outliers": iqr_outliers,
    }


def write_segment_length_analysis(
    analysis: Mapping[str, object],
    *,
    input_root: Path,
    output_dir: Path,
    datasets: Sequence[ValidatedSegmentedDataset],
    overwrite: bool = False,
) -> dict[str, Path]:
    """Write JSON/CSV/Matplotlib analysis artifacts with an explicit overwrite policy."""

    destination = Path(output_dir)
    files = {
        "json": destination / "segment_length_analysis.json",
        "statistics": destination / "segment_length_analysis.csv",
        "candidates": destination / "segment_length_candidates.csv",
        "outliers": destination / "segment_length_outliers.csv",
        "histogram": destination / "segment_length_histogram.png",
        "ecdf": destination / "segment_length_ecdf.png",
    }
    existing = [path for path in files.values() if path.exists()]
    if existing and not overwrite:
        raise SegmentPaddingError("analysis output already exists; use --overwrite: " + ", ".join(map(str, existing)))
    destination.mkdir(parents=True, exist_ok=True)
    records = _analysis_records(analysis)
    payload = {
        "input_root": str(Path(input_root).resolve()),
        "sampling_rate_hz": analysis["sampling_rate_hz"],
        "segment_count": analysis["segment_count"],
        "user_action_count": analysis["user_action_count"],
        "length_statistics": analysis["length_statistics"],
        "minimum_coverage": analysis["minimum_coverage"],
        "round_to": analysis["round_to"],
        "recommendations": analysis["recommendations"],
        "outlier_thresholds": analysis["outlier_thresholds"],
        "packages": _package_fingerprints(datasets, input_root=Path(input_root)),
    }
    _atomic_write_text(files["json"], json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    statistics = dict(analysis["length_statistics"])
    statistics.update({"segment_count": analysis["segment_count"], "user_action_count": analysis["user_action_count"]})
    _atomic_write_csv(files["statistics"], [statistics])
    _atomic_write_csv(files["candidates"], list(analysis["candidate_lengths"]))
    outlier_rows = _outlier_rows(analysis, sampling_rate_hz=float(analysis["sampling_rate_hz"]))
    _atomic_write_csv(files["outliers"], outlier_rows, fieldnames=_OUTLIER_FIELDS)
    _write_analysis_figures(records, files["histogram"], files["ecdf"])
    return files


def resolve_target_length(
    *, target_length: int | None,
    analysis_report: Path | None,
    recommendation: str = "pure-padding",
    datasets: Sequence[ValidatedSegmentedDataset],
    input_root: Path,
) -> int:
    """Resolve an explicit or report-derived target and reject stale reports."""

    if (target_length is None) == (analysis_report is None):
        raise SegmentPaddingError("provide exactly one of target_length or analysis_report")
    if target_length is not None:
        return _positive_integer(target_length, name="target_length")
    assert analysis_report is not None
    report = _read_analysis_report(analysis_report)
    _validate_analysis_report(report, datasets=datasets, input_root=Path(input_root))
    key = {"pure-padding": "pure_padding", "p99": "p99", "balanced": "balanced"}.get(recommendation)
    if key is None:
        raise SegmentPaddingError("recommendation must be pure-padding, p99, or balanced")
    entry = report.get("recommendations", {}).get(key)
    if not isinstance(entry, Mapping):
        raise SegmentPaddingError(f"analysis report has no {recommendation!r} recommendation")
    value = entry.get("target_length")
    target = _positive_integer(value, name=f"{recommendation} target_length")
    return target


def build_padding_package(
    dataset: ValidatedSegmentedDataset, *, target_length: int, padding_value: float = 0.0
) -> PaddingPackageResult:
    """Create right-padded arrays for a validated package without writing files."""

    target = _positive_integer(target_length, name="target_length")
    requested_value = _finite_float(padding_value, name="padding_value")
    stored_value = _stored_padding_value(requested_value, dtype=dataset.raw_imu.dtype)
    lengths = dataset.segment_lengths
    maximum = int(np.max(lengths))
    retained_indices = np.flatnonzero(lengths <= target)
    retained_count = len(retained_indices)
    padded = np.full(
        (retained_count, target, 6), stored_value, dtype=dataset.raw_imu.dtype
    )
    exported_labels = dataset.labels[retained_indices].copy()
    valid_lengths = lengths[retained_indices].astype(np.int32, copy=True)
    valid_mask = np.zeros((retained_count, target), dtype=np.bool_)
    padded_targets = (
        np.zeros((retained_count, target, 4), dtype=np.bool_)
        if dataset.board_event_targets is not None else None
    )
    for output_index, source_index in enumerate(retained_indices):
        start = int(dataset.segment_offsets[source_index])
        stop = int(dataset.segment_offsets[source_index + 1])
        length = int(lengths[source_index])
        padded[output_index, :length] = dataset.raw_imu[start:stop]
        valid_mask[output_index, :length] = True
        if padded_targets is not None:
            assert dataset.board_event_targets is not None
            padded_targets[output_index, :length] = dataset.board_event_targets[start:stop]
    metadata = _manifest_metadata_by_index(dataset)
    rows: list[dict[str, object]] = []
    retained_positions = {int(source): output for output, source in enumerate(retained_indices)}
    for source_index, length_value in enumerate(lengths):
        item = metadata.get(source_index, {})
        length = int(length_value)
        output_index = retained_positions.get(source_index)
        exported = output_index is not None
        rows.append({
            "segment_index": source_index, "output_segment_index": output_index,
            "exported": exported, "skip_reason": None if exported else "length_exceeds_target",
            "user": dataset.paths.user, "action": dataset.paths.action,
            "dataset_id": _optional_int(item.get("dataset_id")), "label": str(dataset.labels[source_index]),
            "original_length": length, "target_length": target,
            "padding_length": target - length if exported else None,
            "valid_fraction": length / target if exported else None,
            "was_padded": length < target if exported else False,
            "board_event_targets_present": padded_targets is not None,
            "source_input_directory": str(dataset.paths.input_dir),
        })
    total_valid = int(np.sum(valid_lengths, dtype=np.int64))
    summary: dict[str, object] = {
        "target_length": target, "source_segment_count": len(lengths),
        "segment_count": retained_count, "skipped_segment_count": len(lengths) - retained_count,
        "minimum_original_length": int(np.min(lengths)), "maximum_original_length": maximum,
        "padded_segment_count": int(np.count_nonzero(valid_lengths < target)),
        "exact_length_segment_count": int(np.count_nonzero(valid_lengths == target)),
        "total_valid_samples": total_valid,
        "total_padding_samples": retained_count * target - total_valid,
        "valid_sample_ratio": (
            0.0 if retained_count == 0 else total_valid / (retained_count * target)
        ),
        "padding_side": "right", "padding_value": stored_value,
        "overflow_policy": "skip",
        "board_event_targets_present": padded_targets is not None,
    }
    return PaddingPackageResult(
        dataset=dataset, padded_imu=padded, labels=exported_labels,
        valid_lengths=valid_lengths, valid_mask=valid_mask,
        padded_board_event_targets=padded_targets, manifest=pd.DataFrame(rows), summary=summary,
    )


def publish_padded_root(
    datasets: Sequence[ValidatedSegmentedDataset],
    *, input_root: Path, output_root: Path, target_length: int,
    sampling_rate_hz: float = DEFAULT_SAMPLING_RATE_HZ, padding_value: float = 0.0,
    overwrite: bool = False,
) -> dict[str, object]:
    """Build all padded packages in staging and publish them as one directory."""

    target = _positive_integer(target_length, name="target_length")
    rate = _positive_finite(sampling_rate_hz, name="sampling_rate_hz")
    destination = Path(output_root)
    _validate_output_root_isolated(input_root=Path(input_root), output_root=destination)
    if destination.exists() and not overwrite:
        raise SegmentPaddingError(f"output root already exists; use --overwrite: {destination}")
    results = [build_padding_package(dataset, target_length=target, padding_value=padding_value) for dataset in datasets]
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{destination.name}.tmp.", dir=destination.parent))
    try:
        for result in results:
            relative = result.dataset.paths.input_dir.relative_to(Path(input_root))
            _write_padding_package(staging / relative, result)
        summary = _root_padding_summary(
            results, input_root=Path(input_root), output_root=destination,
            target_length=target, sampling_rate_hz=rate,
        )
        _atomic_write_text(staging / "padding_dataset_summary.json", json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
        root_rows = [row for result in results for row in result.manifest.to_dict(orient="records")]
        _atomic_write_csv(staging / "padding_dataset_manifest.csv", root_rows, fieldnames=_PADDING_MANIFEST_FIELDS)
        _publish_directory(staging, destination, overwrite=overwrite)
    except Exception:
        if staging.exists():
            shutil.rmtree(staging, ignore_errors=True)
        raise
    return summary


_OUTLIER_FIELDS: Final[tuple[str, ...]] = (
    "rule", "global_segment_index", "local_segment_index", "user", "action", "dataset_id",
    "label", "sample_count", "duration_seconds", "source_directory",
)
_PADDING_MANIFEST_FIELDS: Final[tuple[str, ...]] = (
    "segment_index", "output_segment_index", "exported", "skip_reason", "user", "action",
    "dataset_id", "label", "original_length", "target_length", "padding_length",
    "valid_fraction", "was_padded", "board_event_targets_present", "source_input_directory",
)


def _identity_from_directory(directory: Path, *, stem: str) -> tuple[str, str]:
    if not directory.name.startswith("action_") or not directory.parent.name:
        raise SegmentPaddingError(
            f"lengths stem {stem!r} must be below user/action_<action>: {directory}"
        )
    user = directory.parent.name
    action = directory.name.removeprefix("action_")
    expected_stem = f"{user}_action_{action}"
    if stem != expected_stem:
        raise SegmentPaddingError(
            f"lengths file {directory / (stem + '_segment_lengths.npy')}: expected stem "
            f"{expected_stem!r} from its user/action directory, actual {stem!r}"
        )
    return user, action


def _is_generated_directory(parts: Iterable[str]) -> bool:
    return any(part == "padding_analysis" or "_padded_" in part or part.startswith("padded_") for part in parts)


def _load_array(path: Path, *, paths: SegmentedDatasetPaths, expected: str) -> np.ndarray:
    try:
        value = np.load(path, allow_pickle=False)
    except (OSError, ValueError) as error:
        raise SegmentPaddingError(
            f"user={paths.user!r}, action={paths.action!r}, file={path}: could not load {expected}: {error}"
        ) from error
    if not isinstance(value, np.ndarray):
        raise SegmentPaddingError(f"user={paths.user!r}, action={paths.action!r}, file={path}: expected ndarray")
    return value


def _validate_dataset_arrays(paths: SegmentedDatasetPaths, raw: np.ndarray, labels: np.ndarray, offsets: np.ndarray, lengths: np.ndarray) -> None:
    if raw.ndim != 2 or raw.shape[1:] != (6,):
        _shape_error(paths, paths.raw_imu_path, "(sample_count, 6)", tuple(raw.shape))
    if labels.ndim != 1:
        _shape_error(paths, paths.labels_path, "(segment_count,)", tuple(labels.shape))
    if lengths.ndim != 1:
        _shape_error(paths, paths.segment_lengths_path, "(segment_count,)", tuple(lengths.shape))
    if offsets.ndim != 1:
        _shape_error(paths, paths.segment_offsets_path, "(segment_count + 1,)", tuple(offsets.shape))
    for path, values in ((paths.segment_lengths_path, lengths), (paths.segment_offsets_path, offsets)):
        if not np.issubdtype(values.dtype, np.integer):
            raise SegmentPaddingError(
                f"user={paths.user!r}, action={paths.action!r}, file={path}: "
                f"expected integer array, actual dtype={values.dtype}"
            )
    count = len(lengths)
    if len(labels) != count:
        raise SegmentPaddingError(
            f"user={paths.user!r}, action={paths.action!r}, file={paths.labels_path}: "
            f"expected shape ({count},), actual shape {tuple(labels.shape)}"
        )
    if len(offsets) != count + 1:
        raise SegmentPaddingError(
            f"user={paths.user!r}, action={paths.action!r}, file={paths.segment_offsets_path}: "
            f"expected shape ({count + 1},), actual shape {tuple(offsets.shape)}"
        )
    if count == 0:
        raise SegmentPaddingError(f"user={paths.user!r}, action={paths.action!r}, file={paths.segment_lengths_path}: no segments")
    if int(offsets[0]) != 0 or int(offsets[-1]) != len(raw):
        raise SegmentPaddingError(
            f"user={paths.user!r}, action={paths.action!r}, file={paths.segment_offsets_path}: "
            f"expected first=0 and last={len(raw)}, actual first={int(offsets[0])}, last={int(offsets[-1])}"
        )
    if np.any(np.diff(offsets) < 0):
        raise SegmentPaddingError(f"user={paths.user!r}, action={paths.action!r}, file={paths.segment_offsets_path}: offsets must be nondecreasing")
    actual_lengths = np.diff(offsets)
    if not np.array_equal(lengths, actual_lengths):
        raise SegmentPaddingError(
            f"user={paths.user!r}, action={paths.action!r}, file={paths.segment_lengths_path}: "
            f"expected np.diff({paths.segment_offsets_path.name}), actual values differ"
        )
    if np.any(lengths <= 0):
        raise SegmentPaddingError(f"user={paths.user!r}, action={paths.action!r}, file={paths.segment_lengths_path}: all segment lengths must be positive")


def _shape_error(paths: SegmentedDatasetPaths, path: Path, expected: object, actual: object) -> None:
    raise SegmentPaddingError(
        f"user={paths.user!r}, action={paths.action!r}, file={path}: expected shape {expected}, actual shape {actual}"
    )


def _load_manifest(paths: SegmentedDatasetPaths) -> pd.DataFrame | None:
    if paths.manifest_path is None:
        return None
    try:
        manifest = pd.read_csv(paths.manifest_path)
    except (OSError, UnicodeError, pd.errors.ParserError) as error:
        raise SegmentPaddingError(
            f"user={paths.user!r}, action={paths.action!r}, file={paths.manifest_path}: could not read manifest: {error}"
        ) from error
    if "segment_index" not in manifest.columns:
        raise SegmentPaddingError(
            f"user={paths.user!r}, action={paths.action!r}, file={paths.manifest_path}: expected column segment_index"
        )
    return manifest


def _manifest_metadata_by_index(dataset: ValidatedSegmentedDataset) -> dict[int, Mapping[str, Any]]:
    if dataset.manifest is None:
        return {}
    rows: dict[int, Mapping[str, Any]] = {}
    for _, row in dataset.manifest.iterrows():
        value = row.get("segment_index")
        index = _optional_int(value)
        if index is None:
            continue
        if not 0 <= index < len(dataset.segment_lengths):
            raise SegmentPaddingError(
                f"user={dataset.paths.user!r}, action={dataset.paths.action!r}, file={dataset.paths.manifest_path}: "
                f"segment_index {index} is outside [0, {len(dataset.segment_lengths)})"
            )
        if index in rows:
            raise SegmentPaddingError(
                f"user={dataset.paths.user!r}, action={dataset.paths.action!r}, file={dataset.paths.manifest_path}: "
                f"duplicate segment_index {index}"
            )
        rows[index] = row.to_dict()
    return rows


def _length_statistics(lengths: np.ndarray) -> dict[str, int | float]:
    return {
        "minimum": int(np.min(lengths)), "maximum": int(np.max(lengths)),
        "mean": float(np.mean(lengths)), "standard_deviation": float(np.std(lengths)),
        "median": float(np.median(lengths)), "p25": float(np.quantile(lengths, 0.25)),
        "p75": float(np.quantile(lengths, 0.75)), "p90": float(np.quantile(lengths, 0.90)),
        "p95": float(np.quantile(lengths, 0.95)), "p97_5": float(np.quantile(lengths, 0.975)),
        "p99": float(np.quantile(lengths, 0.99)), "p99_5": float(np.quantile(lengths, 0.995)),
    }


def _candidate_row(lengths: np.ndarray, target: int) -> dict[str, int | float | bool]:
    overflow = int(np.count_nonzero(lengths > target))
    efficiency = float(np.sum(lengths, dtype=np.int64) / (len(lengths) * target))
    return {
        "target_length": target, "coverage": float(1.0 - overflow / len(lengths)),
        "overflow_count": overflow, "valid_sample_ratio": efficiency,
        "padding_ratio": float(1.0 - efficiency), "requires_overflow_handling": overflow > 0,
    }


def _recommendation(lengths: np.ndarray, target: int) -> dict[str, int | float | bool]:
    row = _candidate_row(lengths, target)
    return {
        "target_length": target,
        "coverage": row["coverage"],
        "overflow_count": row["overflow_count"],
        "requires_overflow_handling": bool(row["overflow_count"] > 0),
    }


def _round_up(value: int, multiple: int) -> int:
    return int(math.ceil(value / multiple) * multiple)


def _analysis_records(analysis: Mapping[str, object]) -> tuple[SegmentLengthRecord, ...]:
    records = analysis.get("records")
    if not isinstance(records, tuple) or not all(isinstance(item, SegmentLengthRecord) for item in records):
        raise SegmentPaddingError("analysis records are invalid")
    return records


def _outlier_rows(analysis: Mapping[str, object], *, sampling_rate_hz: float) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    seen: set[tuple[str, int]] = set()
    for rule, records in (("quantile_p99", analysis["quantile_outliers"]), ("iqr_3x", analysis["iqr_outliers"])):
        assert isinstance(records, list)
        for record in records:
            assert isinstance(record, SegmentLengthRecord)
            key = (rule, record.global_segment_index)
            if key in seen:
                continue
            seen.add(key)
            rows.append({
                "rule": rule, "global_segment_index": record.global_segment_index,
                "local_segment_index": record.local_segment_index, "user": record.user,
                "action": record.action, "dataset_id": record.dataset_id, "label": record.label,
                "sample_count": record.sample_count, "duration_seconds": record.sample_count / sampling_rate_hz,
                "source_directory": str(record.source_directory),
            })
    return rows


def _package_fingerprints(datasets: Sequence[ValidatedSegmentedDataset], *, input_root: Path) -> list[dict[str, object]]:
    root = input_root.resolve()
    return [{
        "relative_lengths_path": str(dataset.paths.segment_lengths_path.resolve().relative_to(root)),
        "segment_count": len(dataset.segment_lengths), "maximum_length": int(np.max(dataset.segment_lengths)),
    } for dataset in datasets]


def _write_analysis_figures(records: Sequence[SegmentLengthRecord], histogram: Path, ecdf: Path) -> None:
    import matplotlib.pyplot as plt

    lengths = np.asarray([record.sample_count for record in records], dtype=np.int64)
    figure, axis = plt.subplots(figsize=(8, 5))
    axis.hist(lengths, bins="auto", color="#4477aa", edgecolor="white")
    axis.set(xlabel="Segment length (samples)", ylabel="Segment count", title="Segment-length distribution")
    figure.tight_layout()
    figure.savefig(histogram, dpi=160)
    plt.close(figure)
    sorted_lengths = np.sort(lengths)
    figure, axis = plt.subplots(figsize=(8, 5))
    axis.step(sorted_lengths, np.arange(1, len(sorted_lengths) + 1) / len(sorted_lengths), where="post", color="#4477aa")
    axis.set(xlabel="Segment length (samples)", ylabel="Empirical cumulative probability", ylim=(0.0, 1.02), title="Segment-length ECDF")
    axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(ecdf, dpi=160)
    plt.close(figure)


def _read_analysis_report(path: Path) -> Mapping[str, object]:
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise SegmentPaddingError(f"could not read analysis report {path}: {error}") from error
    if not isinstance(data, dict):
        raise SegmentPaddingError(f"analysis report must be a JSON object: {path}")
    return data


def _validate_analysis_report(report: Mapping[str, object], *, datasets: Sequence[ValidatedSegmentedDataset], input_root: Path) -> None:
    report_root = report.get("input_root")
    if report_root != str(input_root.resolve()):
        raise SegmentPaddingError(
            f"analysis report input_root={report_root!r} does not match current input root {str(input_root.resolve())!r}; rerun analysis"
        )
    if report.get("segment_count") != sum(len(dataset.segment_lengths) for dataset in datasets):
        raise SegmentPaddingError("analysis report segment count differs from current input; rerun analysis")
    expected = _package_fingerprints(datasets, input_root=input_root)
    if report.get("packages") != expected:
        raise SegmentPaddingError("analysis report package lengths differ from current input; rerun analysis")
    stats = report.get("length_statistics")
    if not isinstance(stats, Mapping) or stats.get("maximum") != _global_maximum(datasets):
        raise SegmentPaddingError("analysis report global maximum differs from current input; rerun analysis")


def _global_maximum(datasets: Sequence[ValidatedSegmentedDataset]) -> int:
    return max(int(np.max(dataset.segment_lengths)) for dataset in datasets)


def _validate_output_root_isolated(*, input_root: Path, output_root: Path) -> None:
    """Reject destinations that could replace the source segmentation tree."""

    resolved_input = input_root.resolve()
    resolved_output = output_root.resolve()
    if resolved_output == resolved_input or resolved_input.is_relative_to(resolved_output):
        raise SegmentPaddingError("output root must not equal or contain the input root")


def _stored_padding_value(requested_value: float, *, dtype: np.dtype) -> object:
    """Convert a requested scalar exactly as ``np.full`` will store it."""

    try:
        with np.errstate(over="ignore", invalid="ignore"):
            stored_value = np.asarray(requested_value, dtype=dtype).item()
    except (OverflowError, TypeError, ValueError) as error:
        raise SegmentPaddingError(
            f"padding value is not representable as a finite {dtype} value"
        ) from error
    try:
        finite = bool(np.isfinite(stored_value))
    except TypeError as error:
        raise SegmentPaddingError(
            f"padding value is not representable as a finite {dtype} value"
        ) from error
    if not finite:
        raise SegmentPaddingError(
            f"padding value is not representable as a finite {dtype} value"
        )
    return stored_value


def _write_padding_package(directory: Path, result: PaddingPackageResult) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    stem = result.dataset.paths.stem
    _save_npy(directory / f"{stem}_paddedIMU.npy", result.padded_imu)
    _save_npy(directory / f"{stem}_labels.npy", result.labels)
    _save_npy(directory / f"{stem}_valid_lengths.npy", result.valid_lengths)
    _save_npy(directory / f"{stem}_valid_mask.npy", result.valid_mask)
    if result.padded_board_event_targets is not None:
        _save_npy(directory / f"{stem}_padded_board_event_targets.npy", result.padded_board_event_targets)
    _atomic_write_csv(directory / f"{stem}_padding_manifest.csv", result.manifest.to_dict(orient="records"), fieldnames=_PADDING_MANIFEST_FIELDS)
    _atomic_write_text(directory / f"{stem}_padding_summary.json", json.dumps(result.summary, indent=2, ensure_ascii=False) + "\n")


def _root_padding_summary(results: Sequence[PaddingPackageResult], *, input_root: Path, output_root: Path, target_length: int, sampling_rate_hz: float) -> dict[str, object]:
    valid = sum(int(result.summary["total_valid_samples"]) for result in results)
    padding = sum(int(result.summary["total_padding_samples"]) for result in results)
    source_segments = sum(int(result.summary["source_segment_count"]) for result in results)
    exported_segments = sum(len(result.valid_lengths) for result in results)
    board_count = sum(result.padded_board_event_targets is not None for result in results)
    return {
        "input_root": str(input_root.resolve()), "output_root": str(output_root.resolve()),
        "target_length": target_length, "sampling_rate_hz": sampling_rate_hz,
        "processed_user_action_count": len(results), "source_segment_count": source_segments,
        "segment_count": exported_segments,
        "skipped_segment_count": source_segments - exported_segments,
        "total_valid_samples": valid, "total_padding_samples": padding,
        "global_valid_sample_ratio": 0.0 if exported_segments == 0 else valid / (valid + padding),
        "board_assisted_package_count": board_count, "label_only_package_count": len(results) - board_count,
        "failed_package_count": 0, "padding_side": "right", "overflow_policy": "skip",
    }


def _publish_directory(staging: Path, destination: Path, *, overwrite: bool) -> None:
    backup = destination.with_name(f".{destination.name}.backup")
    if backup.exists():
        raise SegmentPaddingError(f"cannot publish while backup exists: {backup}")
    try:
        if destination.exists():
            if not overwrite:
                raise SegmentPaddingError(f"output root already exists: {destination}")
            os.replace(destination, backup)
        os.replace(staging, destination)
    except OSError as error:
        if backup.exists() and not destination.exists():
            os.replace(backup, destination)
        raise SegmentPaddingError(f"could not publish padded output {destination}: {error}") from error
    if backup.exists():
        shutil.rmtree(backup)


def _save_npy(path: Path, values: np.ndarray) -> None:
    with path.open("wb") as stream:
        np.save(stream, values, allow_pickle=False)


def _atomic_write_text(path: Path, text: str) -> None:
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(text)
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink(missing_ok=True)


def _atomic_write_csv(path: Path, rows: Sequence[Mapping[str, object]], fieldnames: Sequence[str] | None = None) -> None:
    fields = list(fieldnames or (list(rows[0]) if rows else []))
    # Pandas handles empty reports consistently and uses the same CSV escaping as existing exports.
    pd.DataFrame(list(rows), columns=fields).to_csv(path, index=False)


def _positive_integer(value: object, *, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise SegmentPaddingError(f"{name} must be a positive integer")
    return value


def _positive_integer_sequence(values: Sequence[int], *, name: str) -> tuple[int, ...]:
    output = tuple(sorted(set(_positive_integer(value, name=name) for value in values)))
    if not output:
        raise SegmentPaddingError(f"{name} must not be empty")
    return output


def _finite_float(value: object, *, name: str) -> float:
    if isinstance(value, bool):
        raise SegmentPaddingError(f"{name} must be a finite number")
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise SegmentPaddingError(f"{name} must be a finite number") from error
    if not math.isfinite(result):
        raise SegmentPaddingError(f"{name} must be a finite number")
    return result


def _positive_finite(value: object, *, name: str) -> float:
    result = _finite_float(value, name=name)
    if result <= 0.0:
        raise SegmentPaddingError(f"{name} must be positive")
    return result


def _optional_int(value: object) -> int | None:
    if value is None or pd.isna(value):
        return None
    if isinstance(value, (int, np.integer)) and not isinstance(value, bool):
        return int(value)
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return None
