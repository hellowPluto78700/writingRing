from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import shutil
import sys
from typing import Any

from .builders import build_padding, build_recordings, build_segmentation
from .common import (
    DERIVATIVE_METHOD,
    FREQUENCIES_HZ,
    MAX_FILTER_TIME_S,
    OUTPUT_EVENT_SCHEMA,
    OUTPUT_SCHEMA,
    POLARITY_SPLIT_WAVELET_SPIKE_IMU_FEATURE_SCHEMA,
    RATE_HZ,
    BuildError,
    infer_action,
    list_users,
    load_json,
    read_csv,
    sha256_file,
    write_csv,
    write_json,
)
from .metadata import patch_contract


def build_user(
    source_root: Path, output_root: Path, user: str, *, overwrite: bool
) -> dict[str, Any]:
    if source_root.resolve() == output_root.resolve():
        raise BuildError("output root must differ from source root")
    if user not in list_users(source_root):
        raise BuildError(f"unknown user: {user}")
    action = infer_action(source_root)
    recordings, spec, spec_hash, names, units = build_recordings(
        source_root, output_root, user, overwrite=overwrite
    )
    segmentation_dir, segmentation_summary = build_segmentation(
        source_root, output_root, user, action, recordings, spec, spec_hash,
        names, units, overwrite=overwrite,
    )
    padding_dir, padding_summary = build_padding(
        source_root, output_root, user, action, segmentation_dir,
        segmentation_summary, overwrite=overwrite,
    )
    report = {
        "status": "PASS", "schema_version": 1, "user": user, "action": action,
        "recording_count": len(recordings), "recording_ids": sorted(recordings),
        "source_root": str(source_root.resolve()), "output_root": str(output_root.resolve()),
        "sampling_rate_hz": RATE_HZ, "frequencies_hz": list(FREQUENCIES_HZ),
        "max_filter_time_s": MAX_FILTER_TIME_S, "derivative_method": DERIVATIVE_METHOD,
        "feature_schema": OUTPUT_SCHEMA, "event_representation": "unsigned",
        "event_feature_schema": OUTPUT_EVENT_SCHEMA, "channel_count": 66,
        "event_channel_count": 60,
        "composite_encoder_spec_sha256": spec_hash,
        "segmentation_geometry_reused": True, "segmentation_boundaries_recomputed": False,
        "padding_geometry_reused": True,
        "padding_target_length": padding_summary.get("target_length"),
        "segmentation_directory": str(segmentation_dir), "padding_directory": str(padding_dir),
    }
    report_root = output_root / "user_reports"
    report_root.mkdir(parents=True, exist_ok=True)
    report_path = report_root / f"{user}.json"
    if report_path.exists() and not overwrite:
        raise BuildError(f"user report exists: {report_path}; use --overwrite")
    write_json(report_path, report)
    return report


def _directory_digest(root: Path) -> tuple[str, int]:
    import hashlib

    digest, count = hashlib.sha256(), 0
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        digest.update(path.relative_to(root).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(sha256_file(path).encode("ascii"))
        digest.update(b"\n")
        count += 1
    return digest.hexdigest(), count


def finalize(source_root: Path, output_root: Path, *, overwrite: bool) -> dict[str, Any]:
    users, action = list_users(source_root), infer_action(source_root)
    reports: list[dict[str, Any]] = []
    for user in users:
        path = output_root / "user_reports" / f"{user}.json"
        if not path.is_file():
            raise BuildError(f"missing completed user report: {path}")
        report = load_json(path)
        if report.get("status") != "PASS":
            raise BuildError(f"user report is not PASS: {path}")
        reports.append(report)
    if len({report["composite_encoder_spec_sha256"] for report in reports}) != 1:
        raise BuildError("workers produced mixed composite encoder identities")

    first_summary = load_json(
        output_root / "segmentation" / users[0] / f"action_{action}"
        / f"{users[0]}_action_{action}_segmentation_summary.json"
    )
    spec = first_summary["spike_encoder"]
    spec_hash = first_summary["spike_encoder_spec_sha256"]
    names, units = first_summary["channel_names"], first_summary["units"]

    source_padding = source_root / "segmentation_padded"
    output_padding = output_root / "segmentation_padded"
    root_summary = load_json(source_padding / "padding_dataset_summary.json")
    if root_summary.get("feature_schema") != POLARITY_SPLIT_WAVELET_SPIKE_IMU_FEATURE_SCHEMA:
        raise BuildError("source root padding summary is not 36-channel polarity split")
    patch_contract(
        root_summary, names=names, units=units, spec=spec, spec_hash=spec_hash,
        source_schema=POLARITY_SPLIT_WAVELET_SPIKE_IMU_FEATURE_SCHEMA,
    )
    root_summary.update({
        "source_combination_root": str(source_root.resolve()),
        "output_root": str(output_root.resolve()),
        "processed_user_action_count": len(users), "segmentation_geometry_reused": True,
        "padding_geometry_reused": True, "alignment_recomputed": False,
    })
    summary_path = output_padding / "padding_dataset_summary.json"
    manifest_path = output_padding / "padding_dataset_manifest.csv"
    if (summary_path.exists() or manifest_path.exists()) and not overwrite:
        raise BuildError("destination root padding metadata exists; use --overwrite")
    write_json(summary_path, root_summary)
    fields, rows = read_csv(source_padding / "padding_dataset_manifest.csv")
    patched_rows: list[dict[str, str]] = []
    for row in rows:
        item = dict(row)
        for key, value in {
            "feature_schema": OUTPUT_SCHEMA, "event_representation": "unsigned",
            "event_feature_schema": OUTPUT_EVENT_SCHEMA, "event_channel_count": "60",
            "channel_count": "66",
        }.items():
            if key in item:
                item[key] = value
        patched_rows.append(item)
    write_csv(manifest_path, fields, patched_rows)

    source_alignment = source_root / "alignment"
    output_alignment = output_root / "alignment"
    alignment: dict[str, Any] = {
        "source_present": source_alignment.is_dir(), "copied_byte_for_byte": False,
        "recomputed": False,
    }
    if source_alignment.is_dir():
        if output_alignment.exists():
            if not overwrite:
                raise BuildError(f"destination alignment exists: {output_alignment}; use --overwrite")
            shutil.rmtree(output_alignment)
        shutil.copytree(source_alignment, output_alignment)
        source_digest = _directory_digest(source_alignment)
        output_digest = _directory_digest(output_alignment)
        if source_digest != output_digest:
            raise BuildError("alignment copy differs from source")
        alignment.update({
            "copied_byte_for_byte": True, "directory_sha256": source_digest[0],
            "file_count": source_digest[1],
            "note": (
                "Copied as immutable geometry evidence; source 36-channel alignment "
                "provenance is intentionally not rebound."
            ),
        })

    result = {
        "status": "PASS", "schema_version": 1,
        "source_root": str(source_root.resolve()), "output_root": str(output_root.resolve()),
        "action": action, "users": list(users), "user_count": len(users),
        "sampling_rate_hz": RATE_HZ, "frequencies_hz": list(FREQUENCIES_HZ),
        "max_filter_time_s": MAX_FILTER_TIME_S, "derivative_method": DERIVATIVE_METHOD,
        "feature_schema": OUTPUT_SCHEMA, "event_representation": "unsigned",
        "event_feature_schema": OUTPUT_EVENT_SCHEMA,
        "channel_count": 66, "event_channel_count": 60,
        "channel_layout": {
            "linear_acceleration_events": [0, 30], "angular_acceleration_events": [30, 60],
            "acceleration_m_s2": [60, 63], "gyroscope_rad_s": [63, 66],
        },
        "composite_encoder_spec": spec, "composite_encoder_spec_sha256": spec_hash,
        "alignment": alignment,
        "segmentation": {"geometry_reused": True, "boundaries_recomputed": False},
        "padding": {
            "geometry_reused": True, "target_recomputed": False,
            "target_length": root_summary.get("target_length"),
        },
    }
    dataset_summary = output_root / "angular_accel66_dataset_summary.json"
    if dataset_summary.exists() and not overwrite:
        raise BuildError(f"dataset summary exists: {dataset_summary}; use --overwrite")
    write_json(dataset_summary, result)
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build the 64 Hz 66-channel linear/angular-acceleration SpikeIMU variant."
    )
    sub = parser.add_subparsers(dest="command", required=True)
    users = sub.add_parser("list-users")
    users.add_argument("--source-root", type=Path, required=True)
    for name in ("build-user", "finalize"):
        item = sub.add_parser(name)
        item.add_argument("--source-root", type=Path, required=True)
        item.add_argument("--output-root", type=Path, required=True)
        item.add_argument("--sampling-rate-hz", type=float, default=RATE_HZ)
        item.add_argument("--frequencies-hz", type=float, nargs=5, default=FREQUENCIES_HZ)
        item.add_argument("--max-filter-time-s", type=float, default=MAX_FILTER_TIME_S)
        item.add_argument("--overwrite", action="store_true")
        if name == "build-user":
            item.add_argument("--user", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        source_root = args.source_root.expanduser().resolve()
        if args.command == "list-users":
            for user in list_users(source_root):
                print(user)
            return 0
        output_root = args.output_root.expanduser().resolve()
        if not math.isclose(args.sampling_rate_hz, RATE_HZ, rel_tol=0.0, abs_tol=1e-12):
            raise BuildError("sampling-rate-hz is fixed at 64")
        if tuple(args.frequencies_hz) != FREQUENCIES_HZ:
            raise BuildError("frequencies-hz is fixed at 0.5 1 2 4 8")
        if not math.isclose(
            args.max_filter_time_s, MAX_FILTER_TIME_S, rel_tol=0.0, abs_tol=1e-12
        ):
            raise BuildError("max-filter-time-s is fixed at 0.3")
        if args.command == "build-user":
            result = build_user(source_root, output_root, args.user, overwrite=args.overwrite)
        else:
            result = finalize(source_root, output_root, overwrite=args.overwrite)
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    except (BuildError, OSError, ValueError, TypeError, KeyError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
