"""Reusable Matplotlib plots for loaded WritingRing data.

Plotting functions never sort, repair, synchronize, or otherwise mutate the
loaded tables. They return open figures so interactive callers retain control.
Noninteractive callers should close a returned figure with
``matplotlib.pyplot.close(figure)`` after saving or inspecting it.
"""

from __future__ import annotations

from pathlib import Path
import textwrap
from typing import Final, Iterable

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import numpy as np
import pandas as pd

from writingring.board_loader import BoardData
from writingring.gravity import GravityRemovalResult
from writingring.ring_loader import RELATIVE_TIME_COLUMN, RingData


RING_SAMPLE_INDEX: Final[str] = "sample_index"
RING_INFERRED_TIME: Final[str] = "inferred_time"
BOARD_FRAME_INDEX: Final[str] = "frame_index"
BOARD_RAW_TIMESTAMP: Final[str] = "raw_timestamp"

_ACC_COLUMNS: Final[tuple[str, ...]] = ("acc_x", "acc_y", "acc_z")
_GYR_COLUMNS: Final[tuple[str, ...]] = ("gyr_x", "gyr_y", "gyr_z")
_MAX_WARNING_ITEMS: Final[int] = 6


class PlottingError(Exception):
    """Base exception for plotting-only failures."""


class UnsupportedTimeAxisError(PlottingError):
    """Raised when a plot does not support the requested x-axis mode."""


class MissingPlotColumnError(PlottingError):
    """Raised when loaded data lacks a column required by a plot."""


class InferredTimeUnavailableError(MissingPlotColumnError):
    """Raised when the ring inferred-time column is unavailable."""


class PlotOutputError(PlottingError):
    """Raised when a figure cannot be written to the requested path."""


class MalformedPlotDataError(PlottingError):
    """Raised when an object is not a supported loaded-data result."""


def plot_ring_imu(
    ring_data: RingData,
    *,
    time_axis: str = RING_INFERRED_TIME,
    output_path: str | Path | None = None,
    show: bool = True,
) -> Figure:
    """Plot raw accelerometer and gyroscope channels in two shared-x axes.

    ``inferred_time`` uses the loader's existing
    ``relative_time_inferred_s`` column. It never creates a synthetic time
    axis. ``sample_index`` uses the DataFrame's current row order.
    """

    if not isinstance(ring_data, RingData):
        raise MalformedPlotDataError("plot_ring_imu requires a RingData object")
    dataframe = ring_data.dataframe
    _require_columns(dataframe, (*_ACC_COLUMNS, *_GYR_COLUMNS), "ring IMU")

    if time_axis == RING_SAMPLE_INDEX:
        x_values = dataframe.index.to_numpy(copy=True)
        x_label = "Sample index"
    elif time_axis == RING_INFERRED_TIME:
        if RELATIVE_TIME_COLUMN not in dataframe.columns:
            raise InferredTimeUnavailableError(
                "inferred_time requires the existing "
                f"{RELATIVE_TIME_COLUMN!r} column; use sample_index when "
                "inferred ring time is unavailable"
            )
        x_values = dataframe[RELATIVE_TIME_COLUMN].to_numpy(copy=True)
        x_label = (
            "Inferred relative time (s; microsecond interpretation unconfirmed)"
        )
    else:
        raise UnsupportedTimeAxisError(
            f"unsupported ring time axis {time_axis!r}; expected "
            f"{RING_SAMPLE_INDEX!r} or {RING_INFERRED_TIME!r}"
        )

    figure, axes = plt.subplots(
        2,
        1,
        sharex=True,
        figsize=(11, 7),
        layout="constrained",
    )
    for column in _ACC_COLUMNS:
        axes[0].plot(x_values, dataframe[column].to_numpy(copy=True), label=column)
    for column in _GYR_COLUMNS:
        axes[1].plot(x_values, dataframe[column].to_numpy(copy=True), label=column)

    axes[0].set_ylabel("Acceleration raw value")
    axes[1].set_ylabel("Gyroscope raw value")
    axes[1].set_xlabel(x_label)
    for axis in axes:
        axis.grid(True)
        axis.legend()

    figure.suptitle(
        "WritingRing primary ring IMU — "
        f"{ring_data.source_path.name}\n(raw signal units undocumented)"
    )
    warning_items = list(ring_data.warnings)
    if time_axis == RING_INFERRED_TIME:
        warning_items.insert(
            0,
            "Ring time uses an inferred, unconfirmed microsecond interpretation",
        )
    _add_warning_text(figure, warning_items)
    _save_and_show(figure, output_path=output_path, show=show)
    return figure


def plot_ring_gravity_removal(
    ring_data: RingData,
    gravity_result: GravityRemovalResult,
    *,
    time_axis: str = RING_INFERRED_TIME,
    output_path: str | Path | None = None,
    show: bool = True,
) -> Figure:
    """Compare configured acceleration, gravity, and linear acceleration.

    The supplied result is an offline, bidirectional estimate. This function
    only visualizes existing arrays and never recalibrates or mutates either
    input.
    """

    if not isinstance(ring_data, RingData):
        raise MalformedPlotDataError(
            "plot_ring_gravity_removal requires a RingData object"
        )
    if not isinstance(gravity_result, GravityRemovalResult):
        raise MalformedPlotDataError(
            "plot_ring_gravity_removal requires a GravityRemovalResult object"
        )
    dataframe = ring_data.dataframe
    if gravity_result.sample_count != len(dataframe):
        raise MalformedPlotDataError(
            "gravity result sample count does not match RingData row count: "
            f"{gravity_result.sample_count} != {len(dataframe)}"
        )
    if time_axis == RING_SAMPLE_INDEX:
        x_values = dataframe.index.to_numpy(copy=True)
        x_label = "Sample index"
    elif time_axis == RING_INFERRED_TIME:
        if RELATIVE_TIME_COLUMN not in dataframe.columns:
            raise InferredTimeUnavailableError(
                "inferred_time requires the existing "
                f"{RELATIVE_TIME_COLUMN!r} column; use sample_index when "
                "inferred ring time is unavailable"
            )
        x_values = dataframe[RELATIVE_TIME_COLUMN].to_numpy(copy=True)
        x_label = (
            "Inferred relative time (s; microsecond interpretation unconfirmed)"
        )
    else:
        raise UnsupportedTimeAxisError(
            f"unsupported ring time axis {time_axis!r}; expected "
            f"{RING_SAMPLE_INDEX!r} or {RING_INFERRED_TIME!r}"
        )

    figure, axes = plt.subplots(
        4,
        1,
        sharex=True,
        figsize=(12, 10),
        layout="constrained",
    )
    axis_names = ("x", "y", "z")
    for axis_index, axis_name in enumerate(axis_names):
        axes[0].plot(
            x_values,
            gravity_result.acceleration_body[:, axis_index],
            label=f"acc_body_{axis_name}",
        )
        axes[1].plot(
            x_values,
            gravity_result.gravity_body[:, axis_index],
            label=f"gravity_body_{axis_name}",
        )
        axes[2].plot(
            x_values,
            gravity_result.linear_acceleration_body[:, axis_index],
            label=f"linear_acc_body_{axis_name}",
        )
    axes[3].plot(
        x_values,
        gravity_result.correction_confidence,
        label="correction confidence",
        color="tab:blue",
    )
    axes[3].step(
        x_values,
        gravity_result.correction_used.astype(np.float64),
        where="mid",
        label="correction used",
        color="tab:orange",
        alpha=0.65,
    )

    unit = gravity_result.config.acceleration_unit_label
    axes[0].set_ylabel(f"Configured acceleration\n({unit})")
    axes[1].set_ylabel(f"Gravity contribution\n({unit})")
    axes[2].set_ylabel(f"Linear acceleration\n({unit})")
    axes[3].set_ylabel("Correction gate")
    axes[3].set_xlabel(x_label)
    axes[3].set_ylim(-0.05, 1.05)
    for axis in axes:
        axis.grid(True)
        axis.legend(loc="upper right")

    calibration = gravity_result.calibration
    config = gravity_result.config
    status = (
        "calibration passed"
        if calibration.passed
        else "PROVISIONAL calibration"
    )
    figure.suptitle(
        "WritingRing gravity-contribution removal — "
        f"{ring_data.source_path.name}\n"
        f"{status}; samples {calibration.start_sample}:"
        f"{calibration.stop_sample}; nominal {config.sampling_rate_hz:g} Hz; "
        f"profile={config.profile_name}; offline bidirectional estimate"
    )
    warning_items = list(gravity_result.diagnostics.warnings)
    if time_axis == RING_INFERRED_TIME:
        warning_items.insert(
            0,
            "Ring time uses an inferred, unconfirmed microsecond interpretation",
        )
    _add_warning_text(figure, warning_items)
    _save_and_show(figure, output_path=output_path, show=show)
    return figure


def plot_touch_trajectory(
    board_data: BoardData,
    *,
    output_path: str | Path | None = None,
    show: bool = True,
) -> Figure:
    """Plot stored contact x against the upstream ``1 - y_raw`` display value."""

    _validate_board_data(board_data)
    contacts = board_data.contacts
    _require_columns(contacts, ("x", "y_display", "force"), "touch trajectory")

    figure, axis = plt.subplots(figsize=(8, 7), layout="constrained")
    if contacts.empty:
        axis.text(
            0.5,
            0.5,
            "No board contacts are available for this recording.",
            ha="center",
            va="center",
            transform=axis.transAxes,
        )
    else:
        points = axis.scatter(
            contacts["x"].to_numpy(copy=True),
            contacts["y_display"].to_numpy(copy=True),
            c=contacts["force"].to_numpy(copy=True),
            s=12,
        )
        figure.colorbar(
            points,
            ax=axis,
            label="Contact force (raw value; units undocumented)",
        )

    axis.set_xlabel("Stored normalized x")
    axis.set_ylabel("Display y = 1 - y_raw")
    axis.set_aspect("equal", adjustable="box")
    axis.grid(True)
    axis.set_title(
        "WritingRing touch trajectory — "
        f"{_board_identity(board_data)}\n"
        "(coordinates are stored/display values; physical units undocumented)"
    )
    _add_warning_text(figure, _board_warning_items(board_data))
    _save_and_show(figure, output_path=output_path, show=show)
    return figure


def plot_board_force_over_time(
    board_data: BoardData,
    *,
    time_axis: str = BOARD_FRAME_INDEX,
    output_path: str | Path | None = None,
    show: bool = True,
) -> Figure:
    """Plot contact force in original contact-row order.

    Raw board timestamps remain discontinuous when the loaded data is
    discontinuous. No relationship to ring timestamps is assumed.
    """

    _validate_board_data(board_data)
    contacts = board_data.contacts
    _require_columns(
        contacts,
        ("global_frame_index", "frame_timestamp_raw", "force"),
        "board force",
    )

    if time_axis == BOARD_FRAME_INDEX:
        x_values = contacts["global_frame_index"].to_numpy(copy=True)
        x_label = "Global frame index (numeric chunk order)"
    elif time_axis == BOARD_RAW_TIMESTAMP:
        x_values = contacts["frame_timestamp_raw"].to_numpy(copy=True)
        x_label = "Raw board frame timestamp (stored value)"
    else:
        raise UnsupportedTimeAxisError(
            f"unsupported board time axis {time_axis!r}; expected "
            f"{BOARD_FRAME_INDEX!r} or {BOARD_RAW_TIMESTAMP!r}"
        )

    figure, axis = plt.subplots(figsize=(11, 5), layout="constrained")
    if contacts.empty:
        axis.text(
            0.5,
            0.5,
            "No board contacts are available for force plotting.",
            ha="center",
            va="center",
            transform=axis.transAxes,
        )
    else:
        axis.scatter(
            x_values,
            contacts["force"].to_numpy(copy=True),
            s=10,
            alpha=0.75,
        )

    axis.set_xlabel(x_label)
    axis.set_ylabel("Contact force (raw value; units undocumented)")
    axis.grid(True)
    axis.set_title(
        "WritingRing board contact force — "
        f"{_board_identity(board_data)}\n"
        "(board and ring clocks are not assumed synchronized)"
    )
    _add_warning_text(figure, _board_warning_items(board_data))
    _save_and_show(figure, output_path=output_path, show=show)
    return figure


def concise_warning_text(warnings: Iterable[str]) -> str:
    """Return bounded warning text suitable for display inside a figure."""

    unique: list[str] = []
    for warning in warnings:
        normalized = " ".join(str(warning).split())
        if normalized and normalized not in unique:
            unique.append(normalized)
    displayed = unique[:_MAX_WARNING_ITEMS]
    if len(unique) > _MAX_WARNING_ITEMS:
        displayed.append(f"{len(unique) - _MAX_WARNING_ITEMS} more warning(s)")
    return "Warnings: " + " | ".join(displayed) if displayed else ""


def _board_warning_items(board_data: BoardData) -> tuple[str, ...]:
    report = board_data.validation
    warnings: list[str] = []
    if report.empty_chunk_indices:
        warnings.append(
            "Empty board chunks retained: "
            + ", ".join(str(value) for value in report.empty_chunk_indices)
        )
    if report.frames_without_contacts:
        warnings.append(
            f"{report.frames_without_contacts} frame(s) contain no contacts"
        )
    for boundary in report.cross_chunk_backward_boundaries:
        warnings.append(
            "Backward timestamp boundary "
            f"{boundary.previous_chunk_index}->{boundary.next_chunk_index} "
            f"(delta={boundary.timestamp_delta}); order retained"
        )
    covered_fragments = (
        "empty board chunk",
        "empty leading board chunk",
        "frame(s) contain no contacts",
        "backward timestamp boundary",
    )
    warnings.extend(
        warning
        for warning in board_data.warnings
        if not any(fragment in warning.lower() for fragment in covered_fragments)
    )
    return tuple(warnings)


def _add_warning_text(figure: Figure, warnings: Iterable[str]) -> None:
    text = concise_warning_text(warnings)
    if text:
        wrapped = "\n".join(textwrap.wrap(text, width=145))
        layout_engine = figure.get_layout_engine()
        if layout_engine is not None:
            layout_engine.set(rect=(0.0, 0.075, 1.0, 0.925))
        figure.text(
            0.01,
            0.01,
            wrapped,
            ha="left",
            va="bottom",
            fontsize=8,
            color="darkred",
            wrap=True,
        )


def _board_identity(board_data: BoardData) -> str:
    recording = board_data.source_recording
    if recording is not None:
        return (
            f"user={recording.user}, action={recording.action}, "
            f"dataset={recording.dataset_id}"
        )
    return f"dataset={board_data.validation.dataset_id}"


def _validate_board_data(board_data: BoardData) -> None:
    if not isinstance(board_data, BoardData):
        raise MalformedPlotDataError(
            "board plotting requires a BoardData object"
        )
    if not isinstance(board_data.contacts, pd.DataFrame):
        raise MalformedPlotDataError("BoardData.contacts must be a DataFrame")


def _require_columns(
    dataframe: pd.DataFrame,
    columns: Iterable[str],
    plot_name: str,
) -> None:
    if not isinstance(dataframe, pd.DataFrame):
        raise MalformedPlotDataError(f"{plot_name} input must be a DataFrame")
    missing = tuple(column for column in columns if column not in dataframe)
    if missing:
        raise MissingPlotColumnError(
            f"{plot_name} requires missing column(s): {', '.join(missing)}"
        )


def _save_and_show(
    figure: Figure,
    *,
    output_path: str | Path | None,
    show: bool,
) -> None:
    if output_path is not None:
        try:
            path = Path(output_path)
        except TypeError as error:
            raise PlotOutputError(
                f"invalid output path {output_path!r}"
            ) from error
        if path.exists() and not path.is_file():
            raise PlotOutputError(
                f"output path is not a regular file path: {path}"
            )
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            figure.savefig(path)
        except (OSError, ValueError) as error:
            raise PlotOutputError(
                f"cannot save plotting output to {path}: {error}"
            ) from error
    if show:
        plt.show()
