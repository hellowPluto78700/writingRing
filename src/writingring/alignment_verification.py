"""Create fixed-window visual verification for a successful Ring--Board alignment."""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Literal, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from writingring.alignment_io import apply_board_to_ring_offset
from writingring.event_alignment import SequenceAlignmentResult, smooth_transient_score


LabelTimeDomain = Literal["ring", "board", "shared"]


class AlignmentVerificationError(ValueError):
    """Raised when verification inputs or output cannot be safely produced."""


@dataclass(frozen=True, slots=True)
class AlignmentLabel:
    """One label-file marker retained with its original source line number."""

    timestamp_us: float
    label: str
    source_line_number: int


@dataclass(frozen=True, slots=True)
class AlignmentVerificationConfig:
    """Fixed six-panel verification output configuration."""

    segment_duration_s: float = 10.0
    segment_count: int = 6
    verification_start_s: float = 0.0
    label_time_domain: LabelTimeDomain = "shared"
    show_smoothed_transient: bool = False
    output_dpi: int = 200
    overwrite: bool = False


@dataclass(frozen=True, slots=True)
class AlignmentVerificationResult:
    """Output location, visibility counts, and rendering warnings."""

    output_path: Path
    displayed_start_s: float
    displayed_stop_s: float
    ring_sample_count_displayed: int
    press_count_displayed: int
    lift_count_displayed: int
    label_count_displayed: int
    press_count_outside: int
    lift_count_outside: int
    label_count_outside: int
    warnings: tuple[str, ...]


def resolve_alignment_label_path(
    recording_directory: Path,
    *,
    dataset_id: int,
    explicit_label_path: Path | None = None,
) -> Path:
    """Resolve a label file, preferring an explicit path then dataset files."""

    _nonnegative_int(dataset_id, name="dataset_id")
    if explicit_label_path is not None:
        explicit = Path(explicit_label_path)
        if explicit.is_file():
            return explicit
        raise AlignmentVerificationError(
            f"explicit label path is not a regular file: {explicit}"
        )
    directory = Path(recording_directory)
    for candidate in (
        directory / f"{dataset_id}_label.txt",
        directory / f"{dataset_id}_timestamp.txt",
    ):
        if candidate.is_file():
            return candidate
    raise AlignmentVerificationError("no label or timestamp marker file is available")


def load_alignment_labels(path: Path) -> tuple[AlignmentLabel, ...]:
    """Load ``timestamp label`` lines without losing case or source location."""

    source = Path(path)
    try:
        lines = source.read_text(encoding="utf-8").splitlines()
    except OSError as error:
        raise AlignmentVerificationError(
            f"could not read label file: {source}: {error}"
        ) from error
    labels: list[AlignmentLabel] = []
    for line_number, raw_line in enumerate(lines, start=1):
        stripped = raw_line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        parts = stripped.split(maxsplit=1)
        if len(parts) != 2 or not parts[1].strip():
            raise AlignmentVerificationError(
                f"malformed label file line {line_number}: expected timestamp and label"
            )
        timestamp = _finite_float(parts[0], name=f"label timestamp on line {line_number}")
        labels.append(
            AlignmentLabel(
                timestamp_us=timestamp,
                label=parts[1].strip(),
                source_line_number=line_number,
            )
        )
    return tuple(labels)


def build_alignment_verification_path(
    output_root: Path,
    *,
    user: str,
    action: str,
    dataset_id: int,
) -> Path:
    """Return the unique verification PNG path for one recording."""

    _validate_identity(user=user, action=action, dataset_id=dataset_id)
    return Path(output_root) / user / f"action_{action}" / (
        f"{dataset_id}_alignment_verification.png"
    )


def create_alignment_verification_figure(
    *,
    ring_dataframe: pd.DataFrame,
    ring_timestamps_us: np.ndarray,
    transient_score: np.ndarray,
    board_events: pd.DataFrame,
    alignment_result: SequenceAlignmentResult,
    labels: Sequence[AlignmentLabel],
    label_source_path: Path,
    output_path: Path,
    config: AlignmentVerificationConfig = AlignmentVerificationConfig(),
    user: str | None = None,
    action: str | None = None,
    dataset_id: int | None = None,
) -> AlignmentVerificationResult:
    """Render matching verification on the strict alignment work axis.

    Raw Board timestamps are retained in ``board_events``. Aligned timestamps
    are derived locally from the result's work-axis offset and never stored
    back into caller-owned data. The canonical exported offset is deliberately
    kept separate and is consumed by downstream Board-assisted segmentation.
    """

    _validate_config(config)
    _validate_alignment_result(alignment_result)
    timestamps = _timestamps(ring_timestamps_us)
    score = _score(transient_score, sample_count=len(timestamps))
    if not isinstance(ring_dataframe, pd.DataFrame) or len(ring_dataframe) != len(timestamps):
        raise AlignmentVerificationError(
            "ring_dataframe must be a DataFrame with one row per Ring timestamp"
        )
    events = _events(board_events)
    validated_labels = _labels(labels)
    label_path = Path(label_source_path)
    if not label_path.is_file():
        raise AlignmentVerificationError(
            f"label source path is not a regular file: {label_path}"
        )
    if user is not None or action is not None or dataset_id is not None:
        if user is None or action is None or dataset_id is None:
            raise AlignmentVerificationError(
                "user, action, and dataset_id must be supplied together"
            )
        _validate_identity(user=user, action=action, dataset_id=dataset_id)

    output = Path(output_path)
    _validate_output_path(output, overwrite=config.overwrite)
    offset_us = float(alignment_result.best_offset_us)
    ring_start = float(timestamps[0])
    elapsed = (timestamps - ring_start) / 1_000_000.0
    aligned_event_times = apply_board_to_ring_offset(
        events["frame_timestamp_raw"].to_numpy(dtype=np.float64),
        offset_us=offset_us,
    )
    aligned_events = events.assign(
        aligned_ring_timestamp_us=aligned_event_times,
        aligned_ring_elapsed_s=(aligned_event_times - ring_start) / 1_000_000.0,
    )
    event_elapsed = aligned_events["aligned_ring_elapsed_s"].to_numpy(dtype=np.float64)
    label_elapsed = _label_elapsed(
        validated_labels,
        ring_start=ring_start,
        offset_us=offset_us,
        time_domain=config.label_time_domain,
    )
    displayed_start = config.verification_start_s
    displayed_stop = displayed_start + config.segment_duration_s * config.segment_count
    on_display = (elapsed >= displayed_start) & (elapsed <= displayed_stop)
    event_on_display = (event_elapsed >= displayed_start) & (event_elapsed <= displayed_stop)
    labels_on_display = (label_elapsed >= displayed_start) & (label_elapsed <= displayed_stop)
    press = events["event_type"].to_numpy(dtype=str) == "press"
    lift = ~press
    warnings_output: list[str] = [
        "time mapping: ring_timestamp_us = board_timestamp_us + offset_us",
        f"label time domain: {config.label_time_domain}",
    ]
    recording_end = float(elapsed[-1])
    if recording_end < displayed_stop:
        warnings_output.append(
            f"Ring recording ends at {recording_end:.3f} s before verification window end"
        )
    if recording_end > displayed_stop:
        warnings_output.append("Verification image shows only the requested time window")

    figure, axes = plt.subplots(
        config.segment_count,
        1,
        figsize=(12.0, 3.0 * config.segment_count),
        sharey=True,
        layout="constrained",
    )
    axes_array = np.asarray(axes, dtype=object).reshape(-1)
    y_max = float(np.quantile(score, 0.995) * 1.1)
    if not math.isfinite(y_max) or y_max <= 0.0:
        y_max = max(float(np.max(score)), 1.0)
    smooth = (
        smooth_transient_score(score, window_samples=5)
        if config.show_smoothed_transient
        else None
    )
    matched = _event_match_lookup(alignment_result)
    for panel_index, axis in enumerate(axes_array):
        panel_start = displayed_start + panel_index * config.segment_duration_s
        panel_stop = panel_start + config.segment_duration_s
        final_panel = panel_index == config.segment_count - 1
        panel_mask = _panel_mask(elapsed, panel_start, panel_stop, final_panel)
        axis.plot(
            elapsed[panel_mask], score[panel_mask], color="tab:blue", linewidth=0.8,
            label="Ring transient score" if panel_index == 0 else None,
        )
        if smooth is not None:
            axis.plot(
                elapsed[panel_mask], smooth[panel_mask], color="tab:cyan", linewidth=0.55,
                label="Smoothed transient" if panel_index == 0 else None,
            )
        _draw_events(
            axis,
            aligned_events,
            event_elapsed,
            matched,
            panel_start=panel_start,
            panel_stop=panel_stop,
            final_panel=final_panel,
            label_events=panel_index == 0,
        )
        _draw_labels(
            axis,
            validated_labels,
            label_elapsed,
            panel_start=panel_start,
            panel_stop=panel_stop,
            final_panel=final_panel,
        )
        axis.set_xlim(panel_start, panel_stop)
        axis.set_ylim(0.0, y_max)
        axis.set_title(f"{panel_start:g}\N{EN DASH}{panel_stop:g} s")
        if panel_index == config.segment_count - 1:
            axis.set_xlabel("Elapsed Ring time from recording start (s)")
        axis.grid(True, alpha=0.25)
        if recording_end < panel_stop and recording_end >= panel_start:
            axis.text(
                panel_stop,
                0.5,
                f"Recording ends at {recording_end:.1f} s",
                transform=axis.get_xaxis_transform(),
                ha="right",
                va="center",
                fontsize=8,
                color="0.35",
            )
    axes_array[0].set_ylabel("Transient score")
    axes_array[0].legend(loc="upper right", fontsize=8)
    identity = (
        "recording identity not supplied"
        if user is None
        else f"{user} / action {action} / dataset {dataset_id}"
    )
    figure.suptitle(
        "Ring\N{EN DASH}Board Alignment Verification\n"
        f"{identity}; ring_time = board_time + offset; "
        f"offset = {offset_us:.6f} us ({offset_us / 1_000.0:.6f} ms)\n"
        f"label source = {label_path.name}; label time domain = {config.label_time_domain}"
    )
    try:
        output.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(output, dpi=config.output_dpi)
    except OSError as error:
        raise AlignmentVerificationError(
            f"could not save alignment verification image: {output}: {error}"
        ) from error
    finally:
        plt.close(figure)
    if not output.is_file() or output.stat().st_size == 0:
        raise AlignmentVerificationError(
            f"alignment verification image was not written: {output}"
        )
    return AlignmentVerificationResult(
        output_path=output,
        displayed_start_s=displayed_start,
        displayed_stop_s=displayed_stop,
        ring_sample_count_displayed=int(np.count_nonzero(on_display)),
        press_count_displayed=int(np.count_nonzero(press & event_on_display)),
        lift_count_displayed=int(np.count_nonzero(lift & event_on_display)),
        label_count_displayed=int(np.count_nonzero(labels_on_display)),
        press_count_outside=int(np.count_nonzero(press & ~event_on_display)),
        lift_count_outside=int(np.count_nonzero(lift & ~event_on_display)),
        label_count_outside=int(np.count_nonzero(~labels_on_display)),
        warnings=tuple(warnings_output),
    )


def _validate_config(config: AlignmentVerificationConfig) -> None:
    if not isinstance(config, AlignmentVerificationConfig):
        raise AlignmentVerificationError("config must be AlignmentVerificationConfig")
    if config.segment_count != 6:
        raise AlignmentVerificationError("segment_count must be exactly six")
    if _finite_float(config.segment_duration_s, name="segment_duration_s") != 10.0:
        raise AlignmentVerificationError("segment_duration_s must be exactly 10 seconds")
    if _finite_float(config.verification_start_s, name="verification_start_s") < 0.0:
        raise AlignmentVerificationError("verification_start_s must be nonnegative")
    if config.label_time_domain not in {"ring", "board", "shared"}:
        raise AlignmentVerificationError("label_time_domain must be ring, board, or shared")
    if not isinstance(config.show_smoothed_transient, bool):
        raise AlignmentVerificationError("show_smoothed_transient must be a boolean")
    if isinstance(config.output_dpi, bool) or not isinstance(config.output_dpi, int) or config.output_dpi < 1:
        raise AlignmentVerificationError("output_dpi must be a positive integer")
    if not isinstance(config.overwrite, bool):
        raise AlignmentVerificationError("overwrite must be a boolean")


def _validate_alignment_result(result: SequenceAlignmentResult) -> None:
    if not isinstance(result, SequenceAlignmentResult):
        raise AlignmentVerificationError("alignment_result must be SequenceAlignmentResult")
    if not result.success:
        raise AlignmentVerificationError(
            "alignment did not succeed; verification image was not written"
        )
    _finite_float(result.best_offset_us, name="best_offset_us")


def _timestamps(values: np.ndarray) -> np.ndarray:
    try:
        timestamps = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise AlignmentVerificationError("ring_timestamps_us must be numeric") from error
    if timestamps.ndim != 1 or len(timestamps) == 0 or not np.isfinite(timestamps).all():
        raise AlignmentVerificationError("ring_timestamps_us must be a nonempty finite vector")
    if np.any(np.diff(timestamps) < 0.0):
        raise AlignmentVerificationError("ring_timestamps_us must be nondecreasing")
    return timestamps.copy()


def _score(values: np.ndarray, *, sample_count: int) -> np.ndarray:
    try:
        score = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise AlignmentVerificationError("transient_score must be numeric") from error
    if score.ndim != 1 or len(score) != sample_count or not np.isfinite(score).all():
        raise AlignmentVerificationError(
            "transient_score must be finite and match the Ring timestamp count"
        )
    return score.copy()


def _events(events: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(events, pd.DataFrame):
        raise AlignmentVerificationError("board_events must be a DataFrame")
    required = {"event_type", "frame_timestamp_raw"}
    missing = required - set(events.columns)
    if missing:
        raise AlignmentVerificationError(
            "board_events is missing required column(s): " + ", ".join(sorted(missing))
        )
    if not events["event_type"].isin(("press", "lift")).all():
        raise AlignmentVerificationError("board_events event_type must be press or lift")
    try:
        timestamps = events["frame_timestamp_raw"].to_numpy(dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise AlignmentVerificationError("board event timestamps must be numeric") from error
    if not np.isfinite(timestamps).all():
        raise AlignmentVerificationError("board event timestamps must be finite")
    return events.copy(deep=True).reset_index(drop=True)


def _labels(values: Sequence[AlignmentLabel]) -> tuple[AlignmentLabel, ...]:
    output: list[AlignmentLabel] = []
    for label in values:
        if not isinstance(label, AlignmentLabel):
            raise AlignmentVerificationError("labels must contain AlignmentLabel values")
        _finite_float(label.timestamp_us, name="label timestamp")
        if not isinstance(label.label, str) or not label.label:
            raise AlignmentVerificationError("label text must be nonempty")
        output.append(label)
    return tuple(output)


def _label_elapsed(
    labels: Sequence[AlignmentLabel],
    *,
    ring_start: float,
    offset_us: float,
    time_domain: LabelTimeDomain,
) -> np.ndarray:
    timestamps = np.asarray([label.timestamp_us for label in labels], dtype=np.float64)
    if time_domain == "board":
        timestamps = apply_board_to_ring_offset(timestamps, offset_us=offset_us)
    return (timestamps - ring_start) / 1_000_000.0


def _event_match_lookup(result: SequenceAlignmentResult) -> dict[int, bool]:
    matches = result.event_matches
    if not isinstance(matches, pd.DataFrame) or not {"event_index", "matched"}.issubset(matches):
        return {}
    return {
        int(row.event_index): bool(row.matched)
        for row in matches.loc[:, ["event_index", "matched"]].itertuples(index=False)
    }


def _panel_mask(values: np.ndarray, start: float, stop: float, final: bool) -> np.ndarray:
    return (values >= start) & ((values <= stop) if final else (values < stop))


def _draw_events(
    axis: plt.Axes,
    events: pd.DataFrame,
    elapsed: np.ndarray,
    matched: dict[int, bool],
    *,
    panel_start: float,
    panel_stop: float,
    final_panel: bool,
    label_events: bool,
) -> None:
    event_indices = (
        events["event_index"].to_numpy(dtype=int)
        if "event_index" in events
        else np.arange(len(events), dtype=int)
    )
    on_panel = _panel_mask(elapsed, panel_start, panel_stop, final_panel)
    for event_type, color, linestyle, label in (
        ("press", "tab:red", "-", "Board press"),
        ("lift", "tab:orange", "--", "Board lift"),
    ):
        type_mask = events["event_type"].to_numpy(dtype=str) == event_type
        first = True
        for event_index, position in zip(event_indices[type_mask & on_panel], elapsed[type_mask & on_panel], strict=True):
            accepted = matched.get(int(event_index), False)
            axis.axvline(
                float(position), color=color, linestyle=linestyle,
                alpha=0.90 if accepted else 0.35,
                linewidth=1.5 if accepted else 0.65,
                label=label if label_events and first else None,
            )
            first = False


def _draw_labels(
    axis: plt.Axes,
    labels: Sequence[AlignmentLabel],
    elapsed: np.ndarray,
    *,
    panel_start: float,
    panel_stop: float,
    final_panel: bool,
) -> None:
    for position, label in zip(elapsed, labels, strict=True):
        if not bool(_panel_mask(np.asarray([position]), panel_start, panel_stop, final_panel)[0]):
            continue
        level = 0.96 if label.source_line_number % 2 else 0.78
        axis.axvline(float(position), color="0.45", linestyle=":", linewidth=0.9)
        axis.text(
            float(position), level, label.label, rotation=90,
            transform=axis.get_xaxis_transform(), va="top", ha="right",
            fontsize=7, color="0.25",
        )


def _validate_output_path(path: Path, *, overwrite: bool) -> None:
    if path.exists():
        if not path.is_file():
            raise AlignmentVerificationError(
                f"verification output path is not a regular file: {path}"
            )
        if not overwrite:
            raise AlignmentVerificationError(
                f"verification output already exists; use overwrite=True: {path}"
            )


def _validate_identity(*, user: str, action: str, dataset_id: int) -> None:
    for name, value in (("user", user), ("action", action)):
        if not isinstance(value, str) or not value or "/" in value or "\\" in value:
            raise AlignmentVerificationError(f"{name} must be a nonempty safe path component")
    _nonnegative_int(dataset_id, name="dataset_id")


def _finite_float(value: object, *, name: str) -> float:
    if isinstance(value, bool):
        raise AlignmentVerificationError(f"{name} must be finite")
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise AlignmentVerificationError(f"{name} must be finite") from error
    if not math.isfinite(number):
        raise AlignmentVerificationError(f"{name} must be finite")
    return number


def _nonnegative_int(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise AlignmentVerificationError(f"{name} must be a nonnegative integer")
    return value
