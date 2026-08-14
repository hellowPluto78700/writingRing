#!/usr/bin/env python3
"""Plot one Board trajectory image for every exported Board-assisted segment.

Examples:
    # Plot one recording only.
    python scripts/plot_board_segment_trajectories.py \
        --data-root data \
        --user user_0 \
        --recording 0

    # Omit --recording to plot every discovered recording for this user/action.
    python scripts/plot_board_segment_trajectories.py \
        --data-root data \
        --user user_0

    # Omit --user to plot every discovered user for the selected action.
    python scripts/plot_board_segment_trajectories.py \
        --data-root data

By default, the segmentation and plotting roots are derived from the selected
action. For `--action 0`:

    segmentation root:
        outputs/action0_rectified/low-pass/aligned-board-events/segmentation

    plotting output root:
        outputs/plotting_verification/action0

For user_0 / recording 0, images are written under:

    outputs/plotting_verification/action0/user_0/recording_0/

For `--action 1`, the corresponding roots are `action1_rectified` and
`outputs/plotting_verification/action1`.

Each filename uses the exported segment's elapsed Ring-relative start/end time,
rounded to one decimal place, plus the label, for example:

    8.1_9.3s_i.png

The script does not re-segment the recording. It reads the published
Board-assisted segmentation manifest and uses its final exported segment rows,
row indices, alignment offset domain, and offset value.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
import re
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from writingring.alignment_io import (
    ALIGNMENT_WORK_AXIS_DOMAIN,
    CANONICAL_TIMESTAMP_DOMAIN,
    build_offset_domain_timestamps,
)
from writingring.board_event_segmentation import trim_board_stale_tail
from writingring.board_loader import BoardLoadError, load_board
from writingring.discovery import DiscoveryError, Recording, discover_recordings
from writingring.ring_loader import RingLoadError, load_ring


DEFAULT_OUTPUT_ROOT = Path("outputs/plotting_verification")
SCRIPT_VERSION = "2.2"


REQUIRED_MANIFEST_COLUMNS = {
    "dataset_id",
    "segment_index",
    "exported",
    "label",
    "start_sample_index",
    "stop_sample_index_exclusive",
    "final_start_timestamp_us",
    "final_end_timestamp_us",
    "alignment_offset_us",
    "work_axis_offset_us",
    "alignment_offset_domain",
    "input_kind",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", action="version", version=f"%(prog)s {SCRIPT_VERSION}")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument(
        "--user",
        help=(
            "Optional user directory, e.g. user_0. A bare integer such as 0 is also "
            "accepted. When omitted, process all discovered users for the selected action."
        ),
    )
    parser.add_argument(
        "--recording",
        "--dataset-id",
        dest="dataset_id",
        type=int,
        help=(
            "Optional recording/dataset ID (the integer prefix in <id>_ring_0.bin). "
            "When omitted, process every discovered recording matching the selected action. "
            "If --user is omitted, the selection spans all users."
        ),
    )
    parser.add_argument(
        "--action",
        default="0",
        help="Action directory (default: 0).",
    )
    parser.add_argument(
        "--segmentation-root",
        type=Path,
        default=None,
        help=(
            "Root containing user/action Board-assisted segmentation packages "
            "(default: derived from --action as "
            "outputs/action{action}_rectified/low-pass/aligned-board-events/segmentation)."
        ),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help=(
            "Base plot output root; action{action} is appended "
            f"(default: {DEFAULT_OUTPUT_ROOT}/action{{action}})."
        ),
    )
    parser.add_argument("--dpi", type=int, default=200)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing trajectory PNGs with the same generated names.",
    )
    return parser


def default_segmentation_root(action: str) -> Path:
    """Return the default aligned-Board segmentation root for ``action``."""

    return (
        Path("outputs")
        / f"action{action}_rectified"
        / "low-pass"
        / "aligned-board-events"
        / "segmentation"
    )


def action_output_root(output_root: Path, action: str) -> Path:
    """Return the action-isolated plotting root under ``output_root``."""

    return output_root / f"action{action}"


def normalize_user(value: str) -> str:
    value = value.strip()
    if not value:
        raise ValueError("--user must not be empty")
    return value if value.startswith("user_") else f"user_{value}"


def resolve_recordings(
    data_root: Path,
    *,
    user: str | None,
    action: str,
    dataset_id: int | None,
) -> list[Recording]:
    """Discover recordings once and apply the optional user/dataset filters."""
    recordings = [
        recording
        for recording in discover_recordings(data_root)
        if recording.action == str(action)
        and (user is None or recording.user == user)
        and (dataset_id is None or recording.dataset_id == dataset_id)
    ]
    recordings.sort(key=lambda recording: (recording.user, recording.dataset_id))

    if not recordings:
        user_description = user if user is not None else "all users"
        recording_description = (
            "all recordings"
            if dataset_id is None
            else f"dataset_id={dataset_id}"
        )
        raise ValueError(
            f"no recordings found for {user_description}, action={action}, "
            f"{recording_description}"
        )

    identities = [
        (recording.user, recording.dataset_id)
        for recording in recordings
    ]
    if len(identities) != len(set(identities)):
        raise ValueError(
            "recording identity is ambiguous after filtering; duplicate "
            "(user, dataset_id) entries were discovered"
        )

    return recordings


def segmentation_manifest_path(
    segmentation_root: Path, *, user: str, action: str
) -> Path:
    return (
        segmentation_root
        / user
        / f"action_{action}"
        / f"{user}_action_{action}_segments.csv"
    )


def read_segmentation_manifest(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise ValueError(f"segmentation manifest does not exist: {path}")

    manifest = pd.read_csv(path)
    missing = sorted(REQUIRED_MANIFEST_COLUMNS.difference(manifest.columns))
    if missing:
        raise ValueError(
            "segmentation manifest is missing required column(s): "
            + ", ".join(missing)
        )

    manifest = manifest.copy()
    manifest["dataset_id"] = pd.to_numeric(
        manifest["dataset_id"], errors="coerce"
    )
    manifest["_exported_bool"] = _boolean_series(manifest["exported"])
    return manifest


def exported_segments_for_recording(
    manifest: pd.DataFrame, *, dataset_id: int
) -> pd.DataFrame:
    selected = manifest.loc[
        (manifest["dataset_id"] == dataset_id) & manifest["_exported_bool"]
    ].copy()

    if selected.empty:
        return selected

    selected["segment_index"] = pd.to_numeric(
        selected["segment_index"], errors="raise"
    ).astype(np.int64)
    selected["start_sample_index"] = pd.to_numeric(
        selected["start_sample_index"], errors="raise"
    ).astype(np.int64)
    selected["stop_sample_index_exclusive"] = pd.to_numeric(
        selected["stop_sample_index_exclusive"], errors="raise"
    ).astype(np.int64)

    for column in ("final_start_timestamp_us", "final_end_timestamp_us"):
        selected[column] = pd.to_numeric(selected[column], errors="raise")
        if not np.isfinite(selected[column].to_numpy(dtype=np.float64)).all():
            raise ValueError(
                f"exported segment column {column} contains non-finite values"
            )

    selected = selected.sort_values(
        "segment_index", kind="stable"
    ).reset_index(drop=True)
    return selected


def _boolean_series(values: pd.Series) -> pd.Series:
    if values.dtype == bool:
        return values.copy()
    normalized = values.astype(str).str.strip().str.lower()
    known = normalized.isin({"true", "false", "1", "0"})
    if not bool(known.all()):
        bad = sorted(set(normalized.loc[~known].tolist()))
        raise ValueError(
            "manifest exported column contains unsupported value(s): "
            + ", ".join(bad[:5])
        )
    return normalized.isin({"true", "1"})


def unique_string(rows: pd.DataFrame, column: str) -> str:
    values = [str(value) for value in rows[column].dropna().unique().tolist()]
    if len(values) != 1:
        raise ValueError(
            f"expected one {column} value for selected recording, got {values}"
        )
    return values[0]


def unique_float(rows: pd.DataFrame, column: str) -> float:
    values = pd.to_numeric(rows[column], errors="coerce").dropna().unique()
    if len(values) != 1:
        raise ValueError(
            f"expected one finite {column} value for selected recording, got {values.tolist()}"
        )
    value = float(values[0])
    if not math.isfinite(value):
        raise ValueError(f"{column} must be finite")
    return value


def resolve_timestamp_source_path(rows: pd.DataFrame) -> Path | None:
    if "timestamp_source_path" not in rows.columns:
        return None
    raw_values = [
        str(value).strip()
        for value in rows["timestamp_source_path"].dropna().unique().tolist()
        if str(value).strip()
    ]
    if not raw_values:
        return None
    if len(raw_values) != 1:
        raise ValueError(
            "selected recording has multiple timestamp_source_path values: "
            + ", ".join(raw_values)
        )
    path = Path(raw_values[0])
    if path.is_file():
        return path
    if not path.is_absolute():
        candidate = Path.cwd() / path
        if candidate.is_file():
            return candidate
    return None


def load_canonical_feature_timestamps(
    rows: pd.DataFrame,
    *,
    recording: Recording,
) -> tuple[np.ndarray, str]:
    source_path = resolve_timestamp_source_path(rows)
    if source_path is not None:
        try:
            timestamps = np.asarray(np.load(source_path), dtype=np.float64)
        except (OSError, ValueError) as error:
            raise ValueError(
                f"could not load timestamp sidecar {source_path}: {error}"
            ) from error
        source_description = str(source_path)
    else:
        ring = load_ring(recording)
        timestamps = ring.dataframe["timestamp"].to_numpy(dtype=np.float64)
        source_description = f"fallback Ring timestamps: {recording.ring_0_path}"

    if (
        timestamps.ndim != 1
        or timestamps.size < 2
        or not np.isfinite(timestamps).all()
        or np.any(np.diff(timestamps) < 0.0)
    ):
        raise ValueError(
            "feature timestamps must be a finite nondecreasing vector with at least two rows"
        )

    maximum_stop = int(rows["stop_sample_index_exclusive"].max())
    if maximum_stop > timestamps.size:
        raise ValueError(
            f"segment stop index {maximum_stop} exceeds timestamp row count {timestamps.size}"
        )
    return timestamps, source_description


def alignment_configuration(rows: pd.DataFrame) -> tuple[str, str, float]:
    input_kind = unique_string(rows, "input_kind")
    offset_domain = unique_string(rows, "alignment_offset_domain")

    if offset_domain == ALIGNMENT_WORK_AXIS_DOMAIN:
        offset_us = unique_float(rows, "work_axis_offset_us")
    elif offset_domain == CANONICAL_TIMESTAMP_DOMAIN:
        offset_us = unique_float(rows, "alignment_offset_us")
    else:
        raise ValueError(f"unsupported alignment offset domain: {offset_domain}")

    return input_kind, offset_domain, offset_us


def split_contact_tracks(contacts: pd.DataFrame) -> list[pd.DataFrame]:
    """Split reused contact IDs into separate visible trajectory tracks."""
    if contacts.empty:
        return []

    ordered = contacts.sort_values(
        ["global_frame_index", "contact_index"], kind="stable"
    ).copy()
    tracks: list[pd.DataFrame] = []
    active: dict[int, list[int]] = {}
    previous_frame: dict[int, int] = {}

    for row_index, row in ordered.iterrows():
        contact_id = int(row["contact_id"])
        state = int(row["state"])
        frame_index = int(row["global_frame_index"])

        restart = False
        if contact_id in active:
            if state == 1:
                restart = True
            elif frame_index - previous_frame[contact_id] > 5:
                restart = True

        if restart:
            indices = active.pop(contact_id)
            if indices:
                tracks.append(ordered.loc[indices].copy())

        active.setdefault(contact_id, []).append(row_index)
        previous_frame[contact_id] = frame_index

        if state == 3:
            indices = active.pop(contact_id)
            tracks.append(ordered.loc[indices].copy())
            previous_frame.pop(contact_id, None)

    for indices in active.values():
        if indices:
            tracks.append(ordered.loc[indices].copy())

    tracks.sort(key=lambda table: int(table["global_frame_index"].iloc[0]))
    return tracks


def marker_sizes_from_force(force: np.ndarray) -> np.ndarray:
    finite = np.isfinite(force)
    if not finite.any():
        return np.full(len(force), 24.0)
    positive = np.maximum(force[finite], 0.0)
    scale = float(np.quantile(positive, 0.95)) if positive.size else 0.0
    if not math.isfinite(scale) or scale <= 0.0:
        return np.full(len(force), 24.0)
    normalized = np.zeros(len(force), dtype=np.float64)
    normalized[finite] = np.clip(np.maximum(force[finite], 0.0) / scale, 0.0, 1.0)
    return 18.0 + 70.0 * normalized


def sanitize_label(label: object) -> str:
    text = str(label).strip()
    if not text:
        text = "unlabeled"
    text = re.sub(r"[\\/:*?\"<>|]+", "_", text)
    text = re.sub(r"\s+", "_", text)
    text = text.strip("._")
    return text or "unlabeled"


def unique_output_path(
    output_directory: Path,
    *,
    start_s: float,
    end_s: float,
    label: object,
    segment_index: int,
    used_names: set[str],
) -> Path:
    safe_label = sanitize_label(label)
    base = f"{start_s:.1f}_{end_s:.1f}s_{safe_label}"
    name = f"{base}.png"
    if name in used_names:
        name = f"{base}_seg{segment_index:03d}.png"
    used_names.add(name)
    return output_directory / name


def segment_boundary_interval(
    boundary_timestamps_us: np.ndarray,
    *,
    start_index: int,
    stop_index_exclusive: int,
) -> tuple[float, float]:
    sample_count = int(boundary_timestamps_us.size)
    if not 0 <= start_index < sample_count:
        raise ValueError(f"invalid segment start sample index: {start_index}")
    if not start_index < stop_index_exclusive <= sample_count:
        raise ValueError(
            f"invalid segment stop sample index: {stop_index_exclusive} for start {start_index}"
        )

    start_us = float(boundary_timestamps_us[start_index])
    if stop_index_exclusive < sample_count:
        stop_us = float(boundary_timestamps_us[stop_index_exclusive])
    else:
        stop_us = float(np.nextafter(boundary_timestamps_us[-1], np.inf))
    return start_us, stop_us


def select_segment_contacts(
    contacts: pd.DataFrame,
    *,
    offset_us: float,
    start_boundary_us: float,
    stop_boundary_us: float,
    recording_start_boundary_us: float,
) -> pd.DataFrame:
    if contacts.empty:
        result = contacts.copy()
        result["aligned_boundary_timestamp_us"] = pd.Series(dtype=np.float64)
        result["aligned_elapsed_s"] = pd.Series(dtype=np.float64)
        return result

    raw = pd.to_numeric(
        contacts["frame_timestamp_raw"], errors="coerce"
    ).to_numpy(dtype=np.float64)
    aligned = raw + offset_us
    mask = (
        np.isfinite(aligned)
        & (aligned >= start_boundary_us)
        & (aligned < stop_boundary_us)
    )
    result = contacts.loc[mask].copy()
    result["aligned_boundary_timestamp_us"] = aligned[mask]
    result["aligned_elapsed_s"] = (
        aligned[mask] - recording_start_boundary_us
    ) / 1_000_000.0
    return result


def plot_segment(
    *,
    row: pd.Series,
    contacts: pd.DataFrame,
    output_path: Path,
    start_elapsed_s: float,
    end_elapsed_s: float,
    dpi: int,
) -> tuple[int, int]:
    tracks = split_contact_tracks(contacts)

    fig, ax = plt.subplots(figsize=(11.0, 6.5))
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_box_aspect(130.0 / 230.0)
    ax.set_xlabel("Board x (normalized)")
    ax.set_ylabel("Board y_display = 1 - y_raw (normalized)")
    ax.grid(True, alpha=0.25)

    if contacts.empty:
        ax.text(
            0.5,
            0.5,
            "No Board contact samples inside this exported segment",
            transform=ax.transAxes,
            ha="center",
            va="center",
        )
    else:
        elapsed = contacts["aligned_elapsed_s"].to_numpy(dtype=np.float64)
        force = pd.to_numeric(contacts["force"], errors="coerce").to_numpy(
            dtype=np.float64
        )
        scatter = ax.scatter(
            contacts["x"].to_numpy(dtype=np.float64),
            contacts["y_display"].to_numpy(dtype=np.float64),
            c=elapsed,
            s=marker_sizes_from_force(force),
            alpha=0.65,
        )

        for track in tracks:
            x = track["x"].to_numpy(dtype=np.float64)
            y = track["y_display"].to_numpy(dtype=np.float64)
            if not len(x):
                continue
            ax.plot(x, y, linewidth=1.25, alpha=0.8)
            contact_id = int(track["contact_id"].iloc[0])
            start_time = float(track["aligned_elapsed_s"].iloc[0])
            ax.text(
                float(x[0]),
                float(y[0]),
                f"id={contact_id} t={start_time:.3f}s",
                fontsize=7,
            )
            if len(x) >= 2:
                ax.annotate(
                    "",
                    xy=(float(x[-1]), float(y[-1])),
                    xytext=(float(x[-2]), float(y[-2])),
                    arrowprops={"arrowstyle": "->", "linewidth": 1.0},
                )

        colorbar = fig.colorbar(scatter, ax=ax, pad=0.02)
        colorbar.set_label("Aligned elapsed time from recording start (s)")

    label = str(row["label"])
    segment_index = int(row["segment_index"])
    ax.set_title(
        f"Board trajectory | segment {segment_index} | label={label}\n"
        f"exported segment [{start_elapsed_s:.3f}, {end_elapsed_s:.3f}) s | "
        f"contact samples={len(contacts)} | tracks={len(tracks)}"
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return len(contacts), len(tracks)


def process_recording(
    *,
    recording: Recording,
    segments: pd.DataFrame,
    manifest_path: Path,
    output_root: Path,
    dpi: int,
    overwrite: bool,
) -> dict[str, int]:
    canonical_timestamps, timestamp_source = load_canonical_feature_timestamps(
        segments, recording=recording
    )
    input_kind, offset_domain, offset_us = alignment_configuration(segments)
    boundary_timestamps = build_offset_domain_timestamps(
        canonical_timestamps,
        offset_domain=offset_domain,
        input_kind=input_kind,
    )

    board = load_board(recording)
    _, trimmed_contacts, stale_tail_start = trim_board_stale_tail(
        board.frames, board.contacts
    )

    output_directory = output_root / recording.user / f"recording_{recording.dataset_id}"
    output_directory.mkdir(parents=True, exist_ok=True)

    recording_start_canonical_us = float(canonical_timestamps[0])
    recording_start_boundary_us = float(boundary_timestamps[0])
    used_names: set[str] = set()
    total_contact_samples = 0
    total_tracks = 0
    written = 0
    skipped_existing = 0

    for _, row in segments.iterrows():
        segment_index = int(row["segment_index"])
        start_index = int(row["start_sample_index"])
        stop_index = int(row["stop_sample_index_exclusive"])

        start_boundary_us, stop_boundary_us = segment_boundary_interval(
            boundary_timestamps,
            start_index=start_index,
            stop_index_exclusive=stop_index,
        )

        final_start_us = float(row["final_start_timestamp_us"])
        final_end_us = float(row["final_end_timestamp_us"])
        start_elapsed_s = (
            final_start_us - recording_start_canonical_us
        ) / 1_000_000.0
        end_elapsed_s = (
            final_end_us - recording_start_canonical_us
        ) / 1_000_000.0

        if not (
            math.isfinite(start_elapsed_s)
            and math.isfinite(end_elapsed_s)
            and end_elapsed_s > start_elapsed_s
        ):
            raise ValueError(
                f"segment {segment_index} has invalid elapsed boundaries: "
                f"{start_elapsed_s}, {end_elapsed_s}"
            )

        output_path = unique_output_path(
            output_directory,
            start_s=start_elapsed_s,
            end_s=end_elapsed_s,
            label=row["label"],
            segment_index=segment_index,
            used_names=used_names,
        )

        if output_path.exists() and not overwrite:
            print(f"Skipping existing: {output_path}")
            skipped_existing += 1
            continue

        segment_contacts = select_segment_contacts(
            trimmed_contacts,
            offset_us=offset_us,
            start_boundary_us=start_boundary_us,
            stop_boundary_us=stop_boundary_us,
            recording_start_boundary_us=recording_start_boundary_us,
        )

        contact_count, track_count = plot_segment(
            row=row,
            contacts=segment_contacts,
            output_path=output_path,
            start_elapsed_s=start_elapsed_s,
            end_elapsed_s=end_elapsed_s,
            dpi=dpi,
        )
        total_contact_samples += contact_count
        total_tracks += track_count
        written += 1
        print(
            f"segment={segment_index} label={row['label']} "
            f"time={start_elapsed_s:.3f}-{end_elapsed_s:.3f}s "
            f"contacts={contact_count} tracks={track_count} -> {output_path}"
        )

    print()
    print(
        f"Recording: {recording.user}/action_{recording.action}/dataset_{recording.dataset_id}"
    )
    print(f"Segmentation manifest: {manifest_path}")
    print(f"Timestamp source: {timestamp_source}")
    print(f"Alignment domain: {offset_domain}")
    print(f"Boundary offset: {offset_us:.6f} us")
    print(f"Exported segments selected: {len(segments)}")
    print(f"Trajectory images written: {written}")
    print(f"Existing images skipped: {skipped_existing}")
    print(f"Total plotted Board contact samples: {total_contact_samples}")
    print(f"Total plotted trajectory tracks: {total_tracks}")
    if stale_tail_start is not None:
        print(
            f"Board stale tail trimmed from positional frame {stale_tail_start}, "
            "matching Board-assisted segmentation behavior."
        )
    print(f"Output directory: {output_directory}")

    for warning in board.warnings:
        print(
            f"Board warning [recording {recording.dataset_id}]: {warning}",
            file=sys.stderr,
        )

    return {
        "recordings": 1,
        "segments": int(len(segments)),
        "written": written,
        "skipped_existing": skipped_existing,
        "contact_samples": total_contact_samples,
        "tracks": total_tracks,
    }


def main() -> int:
    args = build_parser().parse_args()

    try:
        if args.dataset_id is not None and args.dataset_id < 0:
            raise ValueError("--recording must be nonnegative")
        if args.dpi <= 0:
            raise ValueError("--dpi must be positive")

        selected_user = normalize_user(args.user) if args.user is not None else None
        action = str(args.action)
        segmentation_root = (
            args.segmentation_root
            if args.segmentation_root is not None
            else default_segmentation_root(action)
        )
        output_root = action_output_root(args.output_root, action)
        strict_single_recording = (
            selected_user is not None and args.dataset_id is not None
        )

        recordings = resolve_recordings(
            args.data_root,
            user=selected_user,
            action=action,
            dataset_id=args.dataset_id,
        )

        recordings_by_user: dict[str, list[Recording]] = {}
        for recording in recordings:
            recordings_by_user.setdefault(recording.user, []).append(recording)

        totals = {
            "users_selected": len(recordings_by_user),
            "users_processed": 0,
            "users_without_manifest": 0,
            "recordings_discovered": len(recordings),
            "recordings_processed": 0,
            "recordings_without_segments": 0,
            "recordings_without_manifest": 0,
            "segments": 0,
            "written": 0,
            "skipped_existing": 0,
            "contact_samples": 0,
            "tracks": 0,
        }

        for user, user_recordings in recordings_by_user.items():
            manifest_path = segmentation_manifest_path(
                segmentation_root, user=user, action=action
            )

            if not manifest_path.is_file():
                message = f"segmentation manifest does not exist: {manifest_path}"
                if selected_user is not None:
                    raise ValueError(message)
                print(f"SKIP USER {user}: {message}")
                totals["users_without_manifest"] += 1
                totals["recordings_without_manifest"] += len(user_recordings)
                continue

            manifest = read_segmentation_manifest(manifest_path)
            totals["users_processed"] += 1

            print()
            print(
                f"=== plotting user {user}: {len(user_recordings)} "
                "recording(s) selected ==="
            )

            for recording in user_recordings:
                segments = exported_segments_for_recording(
                    manifest, dataset_id=recording.dataset_id
                )
                if segments.empty:
                    message = (
                        f"no exported Board-assisted segments for "
                        f"{recording.user}/action_{recording.action}/"
                        f"dataset_{recording.dataset_id}"
                    )
                    if strict_single_recording:
                        raise ValueError(message)
                    print(f"SKIP: {message}")
                    totals["recordings_without_segments"] += 1
                    continue

                print()
                print(
                    f"=== plotting {recording.user} recording "
                    f"{recording.dataset_id}: {len(segments)} exported segment(s) ==="
                )
                stats = process_recording(
                    recording=recording,
                    segments=segments,
                    manifest_path=manifest_path,
                    output_root=output_root,
                    dpi=args.dpi,
                    overwrite=args.overwrite,
                )
                totals["recordings_processed"] += 1
                for key in (
                    "segments",
                    "written",
                    "skipped_existing",
                    "contact_samples",
                    "tracks",
                ):
                    totals[key] += stats[key]

        print()
        print("=== plotting summary ===")
        if selected_user is None:
            print(f"User/action: ALL USERS/action_{action}")
        else:
            print(f"User/action: {selected_user}/action_{action}")

        if selected_user is None and args.dataset_id is None:
            mode = "all users, all discovered recordings"
        elif selected_user is None:
            mode = f"all users, recording {args.dataset_id} where present"
        elif args.dataset_id is None:
            mode = f"user {selected_user}, all discovered recordings"
        else:
            mode = f"user {selected_user}, single recording {args.dataset_id}"
        print(f"Mode: {mode}")

        print(f"Users selected: {totals['users_selected']}")
        print(f"Users with manifests processed: {totals['users_processed']}")
        print(f"Users without manifests skipped: {totals['users_without_manifest']}")
        print(f"Recordings selected: {totals['recordings_discovered']}")
        print(f"Recordings processed: {totals['recordings_processed']}")
        print(
            "Recordings skipped due to missing manifest: "
            f"{totals['recordings_without_manifest']}"
        )
        print(
            "Recordings without exported segments: "
            f"{totals['recordings_without_segments']}"
        )
        print(f"Exported segments processed: {totals['segments']}")
        print(f"Trajectory images written: {totals['written']}")
        print(f"Existing images skipped: {totals['skipped_existing']}")
        print(f"Total plotted Board contact samples: {totals['contact_samples']}")
        print(f"Total plotted trajectory tracks: {totals['tracks']}")
        output_summary = (
            output_root
            if selected_user is None
            else output_root / selected_user
        )
        print(f"Output root: {output_summary}")
        return 0

    except (
        DiscoveryError,
        BoardLoadError,
        RingLoadError,
        OSError,
        ValueError,
    ) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
