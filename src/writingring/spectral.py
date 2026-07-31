"""Windowed spectral analysis for the primary WritingRing acceleration stream.

The nominal sampling rate supplied to this module is an analysis assumption.
It is not inferred from, and does not repair or resample, Ring timestamps.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Final

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import numpy as np

from writingring.discovery import parse_recording_filename
from writingring.ring_loader import RingData


ACCELERATION_COLUMNS: Final[tuple[str, ...]] = ("acc_x", "acc_y", "acc_z")
SUPPORTED_AGGREGATES: Final[tuple[str, ...]] = ("mean", "median")


class SpectralAnalysisError(ValueError):
    """Base error for invalid spectral inputs or unsupported analysis."""


class InsufficientSpectralSamplesError(SpectralAnalysisError):
    """Raised when an input cannot provide one complete analysis window."""


class UnsupportedAggregateError(SpectralAnalysisError):
    """Raised when an aggregate other than mean or median is requested."""


class InvalidFrequencyRangeError(SpectralAnalysisError):
    """Raised when plot frequency bounds are invalid for the nominal rate."""


class SpectralPlotError(SpectralAnalysisError):
    """Raised when a spectral figure cannot be created or saved safely."""


@dataclass(frozen=True, slots=True)
class WindowedPSD:
    """One-sided periodograms for every complete fixed-length input window."""

    frequencies_hz: np.ndarray
    psd: np.ndarray
    window_size_samples: int
    hop_size_samples: int
    window_start_samples: np.ndarray

    @property
    def window_count(self) -> int:
        """Return the number of complete windows."""

        return int(self.psd.shape[0])

    @property
    def frequency_resolution_hz(self) -> float:
        """Return the spacing between adjacent frequency bins."""

        if self.frequencies_hz.size < 2:
            raise SpectralAnalysisError(
                "PSD result has fewer than two frequency bins"
            )
        return float(self.frequencies_hz[1] - self.frequencies_hz[0])


@dataclass(frozen=True, slots=True)
class FrequencySupport:
    """Frequency-bin support derived from complete-window PSDs."""

    frequencies_hz: np.ndarray
    relative_power: np.ndarray
    high_power: np.ndarray
    support_count: np.ndarray
    support_percent: np.ndarray
    supporting_strength: np.ndarray
    high_power_quantile: float
    smoothing_bins: int

    @property
    def window_count(self) -> int:
        """Return the number of analyzed PSD windows."""

        return int(self.relative_power.shape[0])


def compute_windowed_psd(
    values: np.ndarray,
    *,
    sampling_rate_hz: float,
    window_seconds: float,
    overlap_ratio: float,
) -> WindowedPSD:
    """Compute Hann-windowed, mean-detrended, one-sided periodograms.

    Only complete windows are emitted. The incomplete final tail is dropped
    without padding, and the input is never modified.
    """

    try:
        input_values = np.asarray(values)
    except (TypeError, ValueError) as error:
        raise SpectralAnalysisError(
            "spectral input must be convertible to a one-dimensional array"
        ) from error
    if input_values.ndim != 1:
        raise SpectralAnalysisError(
            "spectral input must be one-dimensional, "
            f"got shape {input_values.shape}"
        )
    if input_values.size == 0:
        raise SpectralAnalysisError("spectral input must be nonempty")
    try:
        analysis_values = input_values.astype(np.float64, copy=True)
    except (TypeError, ValueError) as error:
        raise SpectralAnalysisError(
            "spectral input must contain numeric values"
        ) from error
    if not np.isfinite(analysis_values).all():
        raise SpectralAnalysisError(
            "spectral input must contain only finite values"
        )

    sampling_rate = _finite_float(
        sampling_rate_hz,
        name="sampling rate",
        positive=True,
    )
    duration = _finite_float(
        window_seconds,
        name="window duration",
        positive=True,
    )
    overlap = _finite_float(overlap_ratio, name="overlap ratio")
    if not 0.0 <= overlap < 1.0:
        raise SpectralAnalysisError(
            "overlap ratio must satisfy 0 <= overlap_ratio < 1"
        )

    raw_window_size = sampling_rate * duration
    if not np.isfinite(raw_window_size):
        raise SpectralAnalysisError(
            "sampling rate multiplied by window duration must be finite"
        )
    window_size = int(round(raw_window_size))
    if window_size < 2:
        raise SpectralAnalysisError(
            "calculated window size must be at least two samples, "
            f"got {window_size}"
        )

    raw_hop_size = window_size * (1.0 - overlap)
    if not np.isfinite(raw_hop_size):
        raise SpectralAnalysisError("calculated hop size must be finite")
    hop_size = int(round(raw_hop_size))
    if hop_size < 1:
        raise SpectralAnalysisError(
            "calculated hop size must be at least one sample, "
            f"got {hop_size}"
        )
    if analysis_values.size < window_size:
        raise InsufficientSpectralSamplesError(
            f"spectral input has {analysis_values.size} sample(s), fewer than "
            f"one complete {window_size}-sample window"
        )

    starts = np.arange(
        0,
        analysis_values.size - window_size + 1,
        hop_size,
        dtype=np.int64,
    )
    hann = np.hanning(window_size)
    hann_energy = float(np.sum(hann**2))
    if not np.isfinite(hann_energy) or hann_energy <= 0.0:
        raise SpectralAnalysisError(
            "calculated Hann window has zero energy; increase window duration"
        )

    frequencies = np.fft.rfftfreq(window_size, d=1.0 / sampling_rate)
    periodograms = np.empty(
        (starts.size, frequencies.size),
        dtype=np.float64,
    )
    normalization = sampling_rate * hann_energy
    for window_index, start in enumerate(starts):
        segment = analysis_values[start : start + window_size]
        detrended = segment - np.mean(segment)
        spectrum = np.fft.rfft(detrended * hann)
        window_psd = (np.abs(spectrum) ** 2) / normalization
        if window_size % 2 == 0:
            window_psd[1:-1] *= 2.0
        else:
            window_psd[1:] *= 2.0
        periodograms[window_index] = window_psd

    frequencies.setflags(write=False)
    periodograms.setflags(write=False)
    starts.setflags(write=False)
    return WindowedPSD(
        frequencies_hz=frequencies,
        psd=periodograms,
        window_size_samples=window_size,
        hop_size_samples=hop_size,
        window_start_samples=starts,
    )


def aggregate_windowed_psd(
    windowed_psd: WindowedPSD,
    *,
    method: str,
) -> np.ndarray:
    """Aggregate window PSDs independently at every frequency bin."""

    if method == "mean":
        aggregate = np.mean(windowed_psd.psd, axis=0)
    elif method == "median":
        aggregate = np.median(windowed_psd.psd, axis=0)
    else:
        raise UnsupportedAggregateError(
            f"unsupported PSD aggregate {method!r}; expected "
            f"{SUPPORTED_AGGREGATES[0]!r} or {SUPPORTED_AGGREGATES[1]!r}"
        )
    aggregate.setflags(write=False)
    return aggregate


def compute_frequency_support(
    windowed_psd: WindowedPSD,
    *,
    frequency_min_hz: float = 1.0,
    frequency_max_hz: float = 100.0,
    high_power_quantile: float = 0.90,
    smoothing_bins: int = 3,
) -> FrequencySupport:
    """Calculate how often each frequency is high-power across PSD windows."""

    if not isinstance(windowed_psd, WindowedPSD):
        raise SpectralAnalysisError(
            "frequency support requires a WindowedPSD result"
        )
    if windowed_psd.psd.ndim != 2 or windowed_psd.psd.shape[0] == 0:
        raise SpectralAnalysisError(
            "frequency support requires at least one PSD window"
        )
    if windowed_psd.psd.shape[1] != windowed_psd.frequencies_hz.size:
        raise SpectralAnalysisError(
            "PSD frequency and value dimensions do not match"
        )
    if not np.isfinite(windowed_psd.psd).all() or np.any(windowed_psd.psd < 0):
        raise SpectralAnalysisError(
            "frequency support requires finite nonnegative PSD values"
        )

    nyquist = float(windowed_psd.frequencies_hz[-1])
    frequency_min, frequency_max = _validate_frequency_range(
        frequency_min_hz,
        frequency_max_hz,
        sampling_rate_hz=nyquist * 2.0,
    )
    if frequency_min <= 0.0:
        raise InvalidFrequencyRangeError(
            "frequency support minimum must be greater than 0 Hz"
        )
    quantile = _finite_float(
        high_power_quantile,
        name="high-power quantile",
    )
    if not 0.0 <= quantile <= 1.0:
        raise SpectralAnalysisError(
            "high-power quantile must satisfy 0 <= quantile <= 1"
        )
    if (
        isinstance(smoothing_bins, bool)
        or not isinstance(smoothing_bins, int)
        or smoothing_bins < 1
        or smoothing_bins % 2 == 0
    ):
        raise SpectralAnalysisError(
            "frequency smoothing bins must be a positive odd integer"
        )

    frequency_mask = (
        (windowed_psd.frequencies_hz >= frequency_min)
        & (windowed_psd.frequencies_hz <= frequency_max)
    )
    frequencies = np.array(
        windowed_psd.frequencies_hz[frequency_mask],
        dtype=np.float64,
        copy=True,
    )
    selected_psd = np.array(
        windowed_psd.psd[:, frequency_mask],
        dtype=np.float64,
        copy=True,
    )
    if frequencies.size == 0:
        raise InvalidFrequencyRangeError(
            "selected support range contains no PSD frequency bins"
        )
    if smoothing_bins > frequencies.size:
        raise SpectralAnalysisError(
            "frequency smoothing bins exceed selected frequency-bin count"
        )

    kernel = np.ones(smoothing_bins, dtype=np.float64) / smoothing_bins
    smoothed_psd = np.apply_along_axis(
        lambda row: np.convolve(row, kernel, mode="same"),
        axis=1,
        arr=selected_psd,
    )
    row_totals = smoothed_psd.sum(axis=1, keepdims=True)
    if not np.isfinite(row_totals).all() or np.any(row_totals <= 0.0):
        raise SpectralAnalysisError(
            "each PSD window must have positive power in the support range"
        )
    relative_power = smoothed_psd / row_totals
    thresholds = np.quantile(
        relative_power,
        quantile,
        axis=1,
        keepdims=True,
    )
    high_power = relative_power >= thresholds
    support_count = np.count_nonzero(high_power, axis=0)
    support_percent = support_count / relative_power.shape[0] * 100.0
    supporting_strength = np.zeros(frequencies.size, dtype=np.float64)
    for frequency_index in range(frequencies.size):
        selected_windows = high_power[:, frequency_index]
        if np.any(selected_windows):
            supporting_strength[frequency_index] = np.median(
                relative_power[selected_windows, frequency_index]
            )

    for values in (
        frequencies,
        relative_power,
        high_power,
        support_count,
        support_percent,
        supporting_strength,
    ):
        values.setflags(write=False)
    return FrequencySupport(
        frequencies_hz=frequencies,
        relative_power=relative_power,
        high_power=high_power,
        support_count=support_count,
        support_percent=support_percent,
        supporting_strength=supporting_strength,
        high_power_quantile=quantile,
        smoothing_bins=smoothing_bins,
    )


def plot_ring_acceleration_psd_overlay(
    ring_data: RingData,
    *,
    sampling_rate_hz: float = 200.0,
    window_seconds: float = 1.0,
    overlap_ratio: float = 0.5,
    aggregate: str = "mean",
    frequency_min_hz: float = 0.0,
    frequency_max_hz: float = 30.0,
    output_path: str | Path | None = None,
    show: bool = False,
) -> Figure:
    """Plot complete-window raw Ring-acceleration PSDs in three panels."""

    if not isinstance(ring_data, RingData):
        raise SpectralAnalysisError(
            "acceleration PSD plotting requires a RingData object"
        )
    missing_columns = tuple(
        column
        for column in ACCELERATION_COLUMNS
        if column not in ring_data.dataframe.columns
    )
    if missing_columns:
        raise SpectralAnalysisError(
            "acceleration PSD plotting requires missing column(s): "
            + ", ".join(missing_columns)
        )
    if aggregate not in SUPPORTED_AGGREGATES:
        raise UnsupportedAggregateError(
            f"unsupported PSD aggregate {aggregate!r}; expected "
            f"{SUPPORTED_AGGREGATES[0]!r} or {SUPPORTED_AGGREGATES[1]!r}"
        )

    return plot_acceleration_psd_overlay(
        ring_data.dataframe.loc[:, ACCELERATION_COLUMNS].to_numpy(copy=True),
        identity=_ring_identity(ring_data),
        source_label="Acceleration",
        psd_unit_label="raw acceleration units²/Hz",
        sampling_rate_hz=sampling_rate_hz,
        window_seconds=window_seconds,
        overlap_ratio=overlap_ratio,
        aggregate=aggregate,
        frequency_min_hz=frequency_min_hz,
        frequency_max_hz=frequency_max_hz,
        output_path=output_path,
        show=show,
    )


def plot_acceleration_psd_overlay(
    acceleration: np.ndarray,
    *,
    identity: str,
    source_label: str,
    psd_unit_label: str,
    sampling_rate_hz: float = 200.0,
    window_seconds: float = 1.0,
    overlap_ratio: float = 0.5,
    aggregate: str = "mean",
    frequency_min_hz: float = 0.0,
    frequency_max_hz: float = 30.0,
    output_path: str | Path | None = None,
    show: bool = False,
) -> Figure:
    """Plot PSD overlays for an explicit finite ``(N, 3)`` acceleration array."""

    acceleration_values = _validated_acceleration_matrix(acceleration)
    if not isinstance(identity, str):
        raise SpectralAnalysisError("PSD identity must be a string")
    if not isinstance(source_label, str) or not source_label:
        raise SpectralAnalysisError("PSD source label must be a nonempty string")
    if not isinstance(psd_unit_label, str) or not psd_unit_label:
        raise SpectralAnalysisError("PSD unit label must be a nonempty string")
    if aggregate not in SUPPORTED_AGGREGATES:
        raise UnsupportedAggregateError(
            f"unsupported PSD aggregate {aggregate!r}; expected "
            f"{SUPPORTED_AGGREGATES[0]!r} or {SUPPORTED_AGGREGATES[1]!r}"
        )

    sampling_rate = _finite_float(
        sampling_rate_hz,
        name="sampling rate",
        positive=True,
    )
    frequency_min, frequency_max = _validate_frequency_range(
        frequency_min_hz,
        frequency_max_hz,
        sampling_rate_hz=sampling_rate,
    )

    results: list[WindowedPSD] = []
    aggregates: list[np.ndarray] = []
    for axis_index in range(3):
        result = compute_windowed_psd(
            acceleration_values[:, axis_index],
            sampling_rate_hz=sampling_rate,
            window_seconds=window_seconds,
            overlap_ratio=overlap_ratio,
        )
        results.append(result)
        aggregates.append(aggregate_windowed_psd(result, method=aggregate))

    frequency_mask = (
        (results[0].frequencies_hz >= frequency_min)
        & (results[0].frequencies_hz <= frequency_max)
    )
    if not np.any(frequency_mask):
        raise InvalidFrequencyRangeError(
            "selected frequency range contains no PSD frequency bins"
        )

    selected_values = [
        result.psd[:, frequency_mask] for result in results
    ] + [
        aggregate_values[frequency_mask]
        for aggregate_values in aggregates
    ]
    positive_values = np.concatenate(
        [values[values > 0.0] for values in selected_values]
    )
    if positive_values.size == 0:
        raise SpectralPlotError(
            "selected frequency range contains no positive PSD values "
            "for logarithmic display"
        )
    smallest_positive = float(np.min(positive_values))
    largest_positive = float(np.max(positive_values))
    plotting_floor = smallest_positive * 0.1
    if plotting_floor <= 0.0 or not np.isfinite(plotting_floor):
        plotting_floor = smallest_positive
    y_min = plotting_floor
    y_max = largest_positive
    if y_max <= y_min:
        y_min = y_max / 10.0
        y_max *= 10.0

    figure, axes = plt.subplots(
        3,
        1,
        sharex=True,
        figsize=(11, 10),
        layout="constrained",
    )
    titles = tuple(f"{source_label} {axis}" for axis in ("X", "Y", "Z"))
    selected_frequencies = results[0].frequencies_hz[frequency_mask]
    try:
        for axis, title, result, aggregate_values in zip(
            axes,
            titles,
            results,
            aggregates,
            strict=True,
        ):
            for window_psd in result.psd[:, frequency_mask]:
                axis.plot(
                    selected_frequencies,
                    _replace_zeros_for_log(window_psd, plotting_floor),
                    color="0.55",
                    linewidth=0.7,
                    alpha=0.25,
                )
            axis.plot(
                selected_frequencies,
                _replace_zeros_for_log(
                    aggregate_values[frequency_mask],
                    plotting_floor,
                ),
                color="tab:blue",
                linewidth=2.5,
                label=f"{aggregate.capitalize()} PSD",
            )
            axis.set_title(title)
            axis.set_ylabel(f"PSD ({psd_unit_label})")
            axis.set_yscale("log")
            axis.set_xlim(frequency_min, frequency_max)
            axis.set_ylim(y_min, y_max)
            axis.grid(True)
            axis.legend()
        axes[-1].set_xlabel("Frequency (Hz)")

        first_result = results[0]
        figure.suptitle(
            f"WritingRing {source_label.lower()} PSD comparison"
            f"{identity}\n"
            f"nominal rate={sampling_rate:g} Hz; "
            f"window={float(window_seconds):g} s; "
            f"overlap={float(overlap_ratio) * 100:g}%; "
            f"windows={first_result.window_count}; "
            f"aggregate={aggregate}"
        )
        if output_path is not None:
            _save_figure(figure, output_path)
        if show:
            plt.show()
    except Exception:
        plt.close(figure)
        raise
    return figure


def plot_ring_acceleration_frequency_support(
    supports: dict[str, FrequencySupport],
    *,
    identity: str,
    source_label: str = "Acceleration",
    sampling_rate_hz: float,
    window_seconds: float,
    overlap_ratio: float,
    minimum_support_percent: float = 20.0,
    top_k_labels: int = 5,
    output_path: str | Path | None = None,
    show: bool = False,
) -> Figure:
    """Plot acceleration frequency support in one three-row scatter figure."""

    if tuple(supports) != ACCELERATION_COLUMNS:
        raise SpectralAnalysisError(
            "frequency-support plotting requires acc_x, acc_y, and acc_z "
            "in that order"
        )
    if not isinstance(source_label, str) or not source_label:
        raise SpectralAnalysisError(
            "frequency-support source label must be a nonempty string"
        )
    minimum_support = _finite_float(
        minimum_support_percent,
        name="minimum support",
    )
    if not 0.0 <= minimum_support <= 100.0:
        raise SpectralAnalysisError(
            "minimum support must be between 0 and 100 percent"
        )
    if (
        isinstance(top_k_labels, bool)
        or not isinstance(top_k_labels, int)
        or top_k_labels < 0
    ):
        raise SpectralAnalysisError(
            "top-k labels must be a nonnegative integer"
        )

    figure, axes = plt.subplots(
        3,
        1,
        sharex=True,
        figsize=(11, 10),
        layout="constrained",
    )
    titles = tuple(f"{source_label} {axis}" for axis in ("X", "Y", "Z"))
    try:
        for axis, title, column in zip(
            axes,
            titles,
            ACCELERATION_COLUMNS,
            strict=True,
        ):
            support = supports[column]
            strengths = support.supporting_strength
            maximum_strength = float(np.max(strengths))
            if maximum_strength > 0.0:
                marker_sizes = 18.0 + 110.0 * strengths / maximum_strength
            else:
                marker_sizes = np.full(strengths.shape, 18.0)
            highlighted = support.support_percent >= minimum_support
            axis.scatter(
                support.frequencies_hz,
                support.support_percent,
                s=marker_sizes,
                color="0.55",
                alpha=0.65,
                label="All frequencies",
            )
            if np.any(highlighted):
                axis.scatter(
                    support.frequencies_hz[highlighted],
                    support.support_percent[highlighted],
                    s=marker_sizes[highlighted] + 35.0,
                    color="tab:orange",
                    edgecolors="black",
                    linewidths=0.5,
                    label=f"Support ≥ {minimum_support:g}%",
                )
            top_indices = _top_support_indices(
                support,
                top_k=top_k_labels,
            )
            if top_indices:
                top_labels = "\n".join(
                    f"{support.frequencies_hz[index]:g} Hz"
                    for index in top_indices
                )
                axis.text(
                    0.98,
                    0.62,
                    f"Top support\n{top_labels}",
                    transform=axis.transAxes,
                    ha="right",
                    va="top",
                    fontsize=8,
                    bbox={
                        "boxstyle": "round",
                        "facecolor": "white",
                        "edgecolor": "0.75",
                        "alpha": 0.85,
                    },
                )
            axis.set_title(title)
            axis.set_ylabel("Windows with high power (%)")
            axis.set_yscale("linear")
            axis.set_ylim(0.0, 100.0)
            axis.set_xlim(
                float(support.frequencies_hz[0]),
                float(support.frequencies_hz[-1]),
            )
            axis.grid(True)
            axis.legend()
        axes[-1].set_xlabel("Frequency (Hz)")

        first_support = supports[ACCELERATION_COLUMNS[0]]
        figure.suptitle(
            f"WritingRing {source_label.lower()} frequency support — "
            f"{identity}\n"
            f"{float(sampling_rate_hz):g} Hz nominal; "
            f"{float(window_seconds):g} s windows; "
            f"{float(overlap_ratio) * 100:g}% overlap; "
            f"{first_support.high_power_quantile * 100:g}th-percentile "
            "high-power threshold"
        )
        if output_path is not None:
            _save_figure(figure, output_path)
        if show:
            plt.show()
    except Exception:
        plt.close(figure)
        raise
    return figure


def _validated_acceleration_matrix(acceleration: np.ndarray) -> np.ndarray:
    """Return a finite floating-point acceleration matrix without aliasing input."""

    try:
        values = np.asarray(acceleration, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise SpectralAnalysisError(
            "PSD acceleration must be convertible to a numeric (N, 3) array"
        ) from error
    if values.ndim != 2 or values.shape[1] != len(ACCELERATION_COLUMNS):
        raise SpectralAnalysisError(
            "PSD acceleration must have shape (N, 3), "
            f"got {values.shape}"
        )
    if values.shape[0] == 0:
        raise SpectralAnalysisError(
            "PSD acceleration must contain at least one sample"
        )
    if not np.isfinite(values).all():
        raise SpectralAnalysisError(
            "PSD acceleration must contain only finite values"
        )
    return np.array(values, dtype=np.float64, copy=True)


def _top_support_indices(
    support: FrequencySupport,
    *,
    top_k: int,
) -> tuple[int, ...]:
    ranked = sorted(
        range(support.frequencies_hz.size),
        key=lambda index: (
            -support.support_percent[index],
            -support.supporting_strength[index],
            support.frequencies_hz[index],
        ),
    )
    return tuple(ranked[:top_k])


def _finite_float(
    value: object,
    *,
    name: str,
    positive: bool = False,
) -> float:
    try:
        converted = float(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise SpectralAnalysisError(f"{name} must be a finite number") from error
    if not np.isfinite(converted):
        raise SpectralAnalysisError(f"{name} must be finite")
    if positive and converted <= 0.0:
        raise SpectralAnalysisError(f"{name} must be positive")
    return converted


def _validate_frequency_range(
    minimum: object,
    maximum: object,
    *,
    sampling_rate_hz: float,
) -> tuple[float, float]:
    try:
        frequency_min = _finite_float(minimum, name="minimum frequency")
        frequency_max = _finite_float(maximum, name="maximum frequency")
    except SpectralAnalysisError as error:
        raise InvalidFrequencyRangeError(str(error)) from error
    if frequency_min < 0.0:
        raise InvalidFrequencyRangeError(
            "minimum frequency must not be negative"
        )
    if frequency_max <= frequency_min:
        raise InvalidFrequencyRangeError(
            "maximum frequency must be greater than minimum frequency"
        )
    nyquist = sampling_rate_hz / 2.0
    if frequency_max > nyquist:
        raise InvalidFrequencyRangeError(
            f"maximum frequency {frequency_max:g} Hz exceeds Nyquist "
            f"frequency {nyquist:g} Hz"
        )
    return frequency_min, frequency_max


def _replace_zeros_for_log(values: np.ndarray, floor: float) -> np.ndarray:
    plotting_values = np.array(values, dtype=np.float64, copy=True)
    plotting_values[plotting_values == 0.0] = floor
    return plotting_values


def _ring_identity(ring_data: RingData) -> str:
    parsed = parse_recording_filename(ring_data.source_path.name)
    fragments: list[str] = []
    if len(ring_data.source_path.parents) >= 2:
        fragments.extend(
            (
                f"user={ring_data.source_path.parent.parent.name}",
                f"action={ring_data.source_path.parent.name}",
            )
        )
    if parsed is not None:
        fragments.append(f"dataset={parsed.dataset_id}")
    return f" — {', '.join(fragments)}" if fragments else ""


def _save_figure(figure: Figure, output_path: str | Path) -> None:
    try:
        path = Path(output_path)
    except TypeError as error:
        raise SpectralPlotError(
            f"invalid spectral output path {output_path!r}"
        ) from error
    if path.exists() and not path.is_file():
        raise SpectralPlotError(
            f"spectral output path is not a regular file path: {path}"
        )
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(path)
    except (OSError, ValueError) as error:
        raise SpectralPlotError(
            f"cannot save spectral output to {path}: {error}"
        ) from error
