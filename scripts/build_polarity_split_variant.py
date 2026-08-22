#!/usr/bin/env python3
"""Build a lossless 30-channel polarity-split padded dataset from a completed WritingRing pipeline.

Source padded SpikeIMU:
    15 signed Custom Wavelet event channels + 6 IMU channels = 21 channels

Derived padded SpikeIMU:
    30 non-negative polarity channels + unchanged 6 IMU channels = 36 channels

For each signed event x:
    x_pos = max(x, 0)
    x_neg = max(-x, 0)

Pairwise output layout:
    e0_pos, e0_neg, e1_pos, e1_neg, ..., e14_pos, e14_neg,
    accel_x, accel_y, accel_z, gyro_x, gyro_y, gyro_z

This is intentionally different from the repository's existing AbsRectify
transform, which keeps 15 event channels and replaces x with abs(x).
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from typing import Any

import numpy as np


SOURCE_EVENT_CHANNELS = 15
SOURCE_TOTAL_CHANNELS = 21
DEST_EVENT_CHANNELS = 30
DEST_TOTAL_CHANNELS = 36

SOURCE_FEATURE_SCHEMA = "signed_wavelet_events_plus_imu_v1"
DEST_FEATURE_SCHEMA = "polarity_split_wavelet_events_plus_imu_v1"
TRANSFORM_NAME = "PolaritySplitAbs"


class PolaritySplitError(RuntimeError):
    pass


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-root",
        type=Path,
        required=True,
        help="Completed action0/action1 combination root.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        help="Defaults to <source-root>/segmentation_padded_polarity_split.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing destination after staging validation succeeds.",
    )
    parser.add_argument(
        "--skip-reuse-preflight",
        action="store_true",
        help="Skip wavelet_variant_reuse.py preflight; intended only for isolated tests.",
    )
    return parser


def load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise PolaritySplitError(f"could not read JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise PolaritySplitError(f"JSON root must be an object: {path}")
    return value


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def infer_action(source_root: Path) -> str:
    spike_root = source_root / "spikeEncoding" / "custom-wavelet"
    if not spike_root.is_dir():
        raise PolaritySplitError(f"missing spike root: {spike_root}")

    actions: set[str] = set()
    for metadata_path in sorted(spike_root.rglob("metadata.json")):
        rel = metadata_path.parent.relative_to(spike_root)
        if len(rel.parts) < 3:
            continue
        token = rel.parts[-2]
        if token.startswith("action_"):
            token = token.removeprefix("action_")
        if token in {"0", "1"}:
            actions.add(token)

    if len(actions) != 1:
        raise PolaritySplitError(
            f"expected exactly one action (0 or 1); found {sorted(actions)}"
        )
    return next(iter(actions))


def infer_boundary_mode(source_root: Path) -> str:
    boundary = source_root.name
    if boundary not in {"label", "aligned-board-events"}:
        raise PolaritySplitError(
            "source root basename must be 'label' or 'aligned-board-events'; "
            f"got {boundary!r}"
        )
    return boundary


def repository_root() -> Path:
    # Installed path:
    #   <repo>/scripts/build_polarity_split_variant.py
    return Path(__file__).resolve().parent.parent


def run_existing_source_checks(
    repo_root: Path,
    source_root: Path,
    action: str,
    boundary_mode: str,
) -> None:
    helper = repo_root / "scripts" / "wavelet_variant_reuse.py"
    if not helper.is_file():
        raise PolaritySplitError(f"missing reuse helper: {helper}")

    preflight = subprocess.run(
        [
            sys.executable,
            str(helper),
            "preflight",
            "--source-root",
            str(source_root),
            "--action",
            action,
            "--boundary-mode",
            boundary_mode,
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    if preflight.returncode != 0:
        raise PolaritySplitError(
            "source pipeline preflight failed:\n"
            + (preflight.stderr or preflight.stdout).strip()
        )
    if preflight.stdout.strip():
        print(preflight.stdout.strip())

    transform = subprocess.run(
        [
            sys.executable,
            str(helper),
            "source-transform",
            "--source-root",
            str(source_root),
            "--action",
            action,
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    if transform.returncode != 0:
        raise PolaritySplitError(
            "source transform validation failed:\n"
            + (transform.stderr or transform.stdout).strip()
        )

    source_transform = transform.stdout.strip()
    if source_transform != "none":
        raise PolaritySplitError(
            "source must be signed/unrectified; "
            f"source-transform returned {source_transform!r}"
        )


def polarity_split_events(spike_imu: np.ndarray) -> np.ndarray:
    """Convert (..., 21) signed SpikeIMU into (..., 36) polarity-split SpikeIMU."""
    if not isinstance(spike_imu, np.ndarray):
        raise TypeError("spike_imu must be a numpy.ndarray")
    if spike_imu.ndim < 2:
        raise PolaritySplitError(
            f"SpikeIMU must have at least 2 dimensions; got {spike_imu.shape}"
        )
    if spike_imu.shape[-1] != SOURCE_TOTAL_CHANNELS:
        raise PolaritySplitError(
            f"expected final dimension {SOURCE_TOTAL_CHANNELS}; got {spike_imu.shape}"
        )
    if not np.issubdtype(spike_imu.dtype, np.number):
        raise PolaritySplitError(f"SpikeIMU dtype must be numeric; got {spike_imu.dtype}")
    if not np.isfinite(spike_imu).all():
        raise PolaritySplitError("SpikeIMU contains non-finite values")

    events = spike_imu[..., :SOURCE_EVENT_CHANNELS]
    imu = spike_imu[..., SOURCE_EVENT_CHANNELS:]

    positive = np.maximum(events, 0)
    negative = np.maximum(-events, 0)

    output = np.empty(
        spike_imu.shape[:-1] + (DEST_TOTAL_CHANNELS,),
        dtype=spike_imu.dtype,
    )
    output[..., 0:DEST_EVENT_CHANNELS:2] = positive
    output[..., 1:DEST_EVENT_CHANNELS:2] = negative
    output[..., DEST_EVENT_CHANNELS:] = imu

    validate_transform(spike_imu, output)
    return output


def validate_transform(source: np.ndarray, derived: np.ndarray) -> None:
    expected_shape = source.shape[:-1] + (DEST_TOTAL_CHANNELS,)
    if derived.shape != expected_shape:
        raise PolaritySplitError(
            f"derived shape mismatch: expected {expected_shape}, got {derived.shape}"
        )
    if derived.dtype != source.dtype:
        raise PolaritySplitError(
            f"dtype changed: {source.dtype} -> {derived.dtype}"
        )

    source_events = source[..., :SOURCE_EVENT_CHANNELS]
    positive = derived[..., 0:DEST_EVENT_CHANNELS:2]
    negative = derived[..., 1:DEST_EVENT_CHANNELS:2]

    if np.any(positive < 0) or np.any(negative < 0):
        raise PolaritySplitError("derived event channels must be non-negative")
    if np.any((positive != 0) & (negative != 0)):
        raise PolaritySplitError(
            "positive and negative polarity are both active for one source event"
        )
    if not np.array_equal(positive - negative, source_events):
        raise PolaritySplitError("polarity split is not exactly invertible")
    if (
        np.count_nonzero(positive) + np.count_nonzero(negative)
        != np.count_nonzero(source_events)
    ):
        raise PolaritySplitError("event nonzero count changed")
    if not np.array_equal(
        derived[..., DEST_EVENT_CHANNELS:],
        source[..., SOURCE_EVENT_CHANNELS:],
    ):
        raise PolaritySplitError("trailing accel/gyro channels changed")


def split_channel_names(value: Any) -> list[str] | None:
    if value is None:
        return None
    if (
        not isinstance(value, list)
        or len(value) != SOURCE_TOTAL_CHANNELS
        or not all(isinstance(item, str) for item in value)
    ):
        raise PolaritySplitError(
            f"expected {SOURCE_TOTAL_CHANNELS} channel names"
        )

    output: list[str] = []
    for name in value[:SOURCE_EVENT_CHANNELS]:
        output.extend((f"{name}_pos", f"{name}_neg_abs"))
    output.extend(value[SOURCE_EVENT_CHANNELS:])
    return output


def split_units(value: Any) -> list[Any] | dict[str, Any] | None:
    if value is None:
        return None
    if isinstance(value, dict):
        if "channel_units" not in value:
            raise PolaritySplitError(
                "units metadata must contain channel_units"
            )
        output = dict(value)
        output["channel_units"] = split_units(output["channel_units"])
        return output
    if not isinstance(value, list) or len(value) != SOURCE_TOTAL_CHANNELS:
        raise PolaritySplitError(
            f"expected {SOURCE_TOTAL_CHANNELS} channel units"
        )
    output_list: list[Any] = []
    for unit in value[:SOURCE_EVENT_CHANNELS]:
        output_list.extend((unit, unit))
    output_list.extend(value[SOURCE_EVENT_CHANNELS:])
    return output_list


def patch_summary(payload: dict[str, Any]) -> None:
    source_count = payload.get("channel_count")
    if source_count is not None and source_count != SOURCE_TOTAL_CHANNELS:
        raise PolaritySplitError(
            f"unexpected source channel_count={source_count!r}"
        )

    source_schema = payload.get("feature_schema")
    if source_schema is not None and source_schema != SOURCE_FEATURE_SCHEMA:
        raise PolaritySplitError(
            f"unexpected source feature_schema={source_schema!r}"
        )

    payload["source_feature_schema"] = source_schema or SOURCE_FEATURE_SCHEMA
    payload["source_channel_count"] = SOURCE_TOTAL_CHANNELS
    payload["feature_schema"] = DEST_FEATURE_SCHEMA
    payload["channel_count"] = DEST_TOTAL_CHANNELS
    payload["event_channel_count"] = DEST_EVENT_CHANNELS
    payload["trailing_imu_channel_count"] = 6
    payload["derived_event_transform"] = {
        "name": TRANSFORM_NAME,
        "source_event_channel_count": SOURCE_EVENT_CHANNELS,
        "output_event_channel_count": DEST_EVENT_CHANNELS,
        "channel_layout": "pairwise_positive_negative",
        "positive_definition": "max(x, 0)",
        "negative_definition": "max(-x, 0)",
        "lossless": True,
        "inverse": "signed_event = positive - negative",
    }

    if "channel_names" in payload:
        names = split_channel_names(payload["channel_names"])
        if names is not None:
            payload["channel_names"] = names
    if "units" in payload:
        payload["units"] = split_units(payload["units"])
    if "channel_units" in payload:
        payload["channel_units"] = split_units(payload["channel_units"])


def patch_manifest(path: Path) -> None:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames
        rows = list(reader)

    if not fieldnames:
        return

    changed = False
    for row in rows:
        if "channel_count" in row:
            row["channel_count"] = str(DEST_TOTAL_CHANNELS)
            changed = True
        if "feature_schema" in row:
            row["feature_schema"] = DEST_FEATURE_SCHEMA
            changed = True

    if not changed:
        return

    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def validate_source_padded_root(
    padded_root: Path,
) -> tuple[dict[str, Any], list[Path]]:
    if not padded_root.is_dir():
        raise PolaritySplitError(
            f"pipeline is not complete: missing {padded_root}"
        )

    summary_path = padded_root / "padding_dataset_summary.json"
    manifest_path = padded_root / "padding_dataset_manifest.csv"

    if not summary_path.is_file():
        raise PolaritySplitError(
            f"pipeline is not complete: missing {summary_path}"
        )
    if not manifest_path.is_file():
        raise PolaritySplitError(
            f"pipeline is not complete: missing {manifest_path}"
        )

    summary = load_json(summary_path)
    if summary.get("channel_count") != SOURCE_TOTAL_CHANNELS:
        raise PolaritySplitError(
            "source padded root is not canonical 21-channel SpikeIMU: "
            f"channel_count={summary.get('channel_count')!r}"
        )
    if summary.get("feature_schema") != SOURCE_FEATURE_SCHEMA:
        raise PolaritySplitError(
            "source padded root is not signed-wavelet SpikeIMU: "
            f"feature_schema={summary.get('feature_schema')!r}"
        )

    arrays = sorted(padded_root.rglob("*_paddedSpikeIMU.npy"))
    if not arrays:
        raise PolaritySplitError(
            f"no *_paddedSpikeIMU.npy files below {padded_root}"
        )

    expected = summary.get("processed_user_action_count")
    if isinstance(expected, int) and expected != len(arrays):
        raise PolaritySplitError(
            f"package count mismatch: summary={expected}, found={len(arrays)}"
        )

    return summary, arrays


def safe_publish(staging: Path, destination: Path, overwrite: bool) -> None:
    if not destination.exists():
        os.replace(staging, destination)
        return

    if not overwrite:
        raise PolaritySplitError(
            f"destination exists: {destination}; use --overwrite"
        )

    backup = destination.with_name(f".{destination.name}.backup")
    if backup.exists():
        shutil.rmtree(backup)

    os.replace(destination, backup)
    try:
        os.replace(staging, destination)
    except Exception:
        if destination.exists():
            shutil.rmtree(destination, ignore_errors=True)
        os.replace(backup, destination)
        raise
    else:
        shutil.rmtree(backup, ignore_errors=True)


def build_variant(
    source_root: Path,
    output_root: Path,
    action: str,
    boundary_mode: str,
    *,
    overwrite: bool,
) -> dict[str, Any]:
    source_padded = source_root / "segmentation_padded"
    root_summary, source_arrays = validate_source_padded_root(source_padded)

    output_root.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(
            prefix=f".{output_root.name}.tmp.",
            dir=output_root.parent,
        )
    )
    published = False

    source_nonzero_events = 0
    source_negative_events = 0

    try:
        shutil.copytree(source_padded, staging, dirs_exist_ok=True)

        for source_array_path in source_arrays:
            relative = source_array_path.relative_to(source_padded)
            staged_array_path = staging / relative

            source_values = np.load(
                source_array_path,
                allow_pickle=False,
                mmap_mode="r",
            )
            if source_values.ndim != 3:
                raise PolaritySplitError(
                    f"expected padded shape (S,T,21), got "
                    f"{source_values.shape}: {source_array_path}"
                )

            events = source_values[..., :SOURCE_EVENT_CHANNELS]
            source_nonzero_events += int(np.count_nonzero(events))
            source_negative_events += int(np.count_nonzero(events < 0))

            derived = polarity_split_events(source_values)
            temporary = staged_array_path.with_name(
                f".{staged_array_path.name}.tmp.npy"
            )
            np.save(temporary, derived, allow_pickle=False)
            os.replace(temporary, staged_array_path)

        for summary_path in sorted(staging.rglob("*_padding_summary.json")):
            payload = load_json(summary_path)
            patch_summary(payload)
            payload["source_padding_root"] = str(source_padded)
            payload["source_combination_root"] = str(source_root)
            payload["derived_action"] = action
            payload["boundary_mode"] = boundary_mode
            write_json(summary_path, payload)

        staged_root_summary_path = staging / "padding_dataset_summary.json"
        staged_root_summary = load_json(staged_root_summary_path)
        patch_summary(staged_root_summary)
        staged_root_summary["source_padding_root"] = str(source_padded)
        staged_root_summary["source_combination_root"] = str(source_root)
        staged_root_summary["output_root"] = str(output_root)
        staged_root_summary["derived_action"] = action
        staged_root_summary["boundary_mode"] = boundary_mode
        staged_root_summary["source_event_nonzero_count"] = source_nonzero_events
        staged_root_summary["source_negative_event_count"] = source_negative_events
        write_json(staged_root_summary_path, staged_root_summary)

        for manifest_path in sorted(staging.rglob("*_padding_manifest.csv")):
            patch_manifest(manifest_path)
        root_manifest = staging / "padding_dataset_manifest.csv"
        if root_manifest.is_file():
            patch_manifest(root_manifest)

        provenance = {
            "schema_version": 1,
            "transform": TRANSFORM_NAME,
            "action": action,
            "boundary_mode": boundary_mode,
            "source_combination_root": str(source_root),
            "source_padding_root": str(source_padded),
            "output_root": str(output_root),
            "source_feature_schema": SOURCE_FEATURE_SCHEMA,
            "output_feature_schema": DEST_FEATURE_SCHEMA,
            "source_event_channel_count": SOURCE_EVENT_CHANNELS,
            "output_event_channel_count": DEST_EVENT_CHANNELS,
            "source_total_channel_count": SOURCE_TOTAL_CHANNELS,
            "output_total_channel_count": DEST_TOTAL_CHANNELS,
            "channel_layout": "pairwise_positive_negative",
            "inverse": "signed_event = positive - negative",
            "source_event_nonzero_count": source_nonzero_events,
            "source_negative_event_count": source_negative_events,
            "source_padding_target_length": root_summary.get("target_length"),
        }
        write_json(staging / "polarity_split_provenance.json", provenance)

        # Re-open and verify every staged array before publishing.
        staged_arrays = sorted(staging.rglob("*_paddedSpikeIMU.npy"))
        if len(staged_arrays) != len(source_arrays):
            raise PolaritySplitError(
                f"staged package count changed: "
                f"{len(source_arrays)} -> {len(staged_arrays)}"
            )

        for source_path, staged_path in zip(source_arrays, staged_arrays):
            source_values = np.load(
                source_path, allow_pickle=False, mmap_mode="r"
            )
            staged_values = np.load(
                staged_path, allow_pickle=False, mmap_mode="r"
            )
            validate_transform(source_values, staged_values)

        final_summary = load_json(staged_root_summary_path)
        if final_summary.get("channel_count") != DEST_TOTAL_CHANNELS:
            raise PolaritySplitError(
                "staged root summary does not declare 36 channels"
            )
        if final_summary.get("feature_schema") != DEST_FEATURE_SCHEMA:
            raise PolaritySplitError(
                "staged root summary has wrong feature_schema"
            )

        safe_publish(staging, output_root, overwrite=overwrite)
        published = True

    finally:
        if not published and staging.exists():
            shutil.rmtree(staging, ignore_errors=True)

    return {
        "status": "PASS",
        "source_root": str(source_root),
        "output_root": str(output_root),
        "action": action,
        "boundary_mode": boundary_mode,
        "package_count": len(source_arrays),
        "source_event_channels": SOURCE_EVENT_CHANNELS,
        "output_event_channels": DEST_EVENT_CHANNELS,
        "source_total_channels": SOURCE_TOTAL_CHANNELS,
        "output_total_channels": DEST_TOTAL_CHANNELS,
        "source_nonzero_event_count": source_nonzero_events,
        "source_negative_event_count": source_negative_events,
        "feature_schema": DEST_FEATURE_SCHEMA,
    }


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    try:
        source_root = args.source_root.expanduser().resolve()
        if not source_root.is_dir():
            raise PolaritySplitError(
                f"source root is not a directory: {source_root}"
            )

        action = infer_action(source_root)
        boundary_mode = infer_boundary_mode(source_root)

        if not args.skip_reuse_preflight:
            run_existing_source_checks(
                repository_root(),
                source_root,
                action,
                boundary_mode,
            )

        output_root = (
            args.output_root.expanduser().resolve()
            if args.output_root is not None
            else source_root / "segmentation_padded_polarity_split"
        )

        source_padded = source_root / "segmentation_padded"
        if output_root == source_padded:
            raise PolaritySplitError(
                "output root must not overwrite source segmentation_padded"
            )
        if output_root == source_root or source_root.is_relative_to(output_root):
            raise PolaritySplitError(
                "output root must not equal or contain the source combination root"
            )

        result = build_variant(
            source_root,
            output_root,
            action,
            boundary_mode,
            overwrite=args.overwrite,
        )
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0

    except (PolaritySplitError, OSError, ValueError, TypeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
