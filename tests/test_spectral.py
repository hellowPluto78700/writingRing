from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from writingring.ring_loader import RingData, load_ring
from writingring.spectral import (
    InvalidFrequencyRangeError,
    SpectralAnalysisError,
    SpectralPlotError,
    aggregate_windowed_psd,
    compute_windowed_psd,
    plot_acceleration_psd_overlay,
    plot_ring_acceleration_psd_overlay,
)


def _sine(
    frequency_hz: float,
    *,
    sample_count: int = 200,
    sampling_rate_hz: float = 200.0,
    amplitude: float = 1.0,
) -> np.ndarray:
    time = np.arange(sample_count, dtype=np.float64) / sampling_rate_hz
    return amplitude * np.sin(2.0 * np.pi * frequency_hz * time)


def _ring_data(
    tmp_path: Path,
    *,
    sample_count: int = 400,
    acceleration: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None,
    gyroscope_value: float = 0.0,
) -> RingData:
    if acceleration is None:
        acceleration = (
            _sine(5.0, sample_count=sample_count),
            _sine(10.0, sample_count=sample_count),
            _sine(20.0, sample_count=sample_count),
        )
    rows = np.column_stack(
        (
            *acceleration,
            np.full(sample_count, gyroscope_value),
            np.full(sample_count, gyroscope_value),
            np.full(sample_count, gyroscope_value),
            1_000_000.0 + np.arange(sample_count) * 5_000.0,
        )
    )
    action = tmp_path / "writer_a" / "letters"
    action.mkdir(parents=True)
    path = action / "0_ring_0.bin"
    rows.astype(np.float64).tofile(path)
    return load_ring(path)


@pytest.mark.parametrize("frequency_hz", [5.0, 20.0])
def test_sine_peak_is_at_expected_frequency(frequency_hz: float) -> None:
    result = compute_windowed_psd(
        _sine(frequency_hz),
        sampling_rate_hz=200.0,
        window_seconds=1.0,
        overlap_ratio=0.5,
    )

    peak = result.frequencies_hz[np.argmax(result.psd[0])]
    assert peak == pytest.approx(frequency_hz)


def test_default_window_frequency_spacing_and_shape() -> None:
    result = compute_windowed_psd(
        _sine(5.0),
        sampling_rate_hz=200.0,
        window_seconds=1.0,
        overlap_ratio=0.5,
    )

    assert result.window_size_samples == 200
    assert result.hop_size_samples == 100
    assert result.psd.shape == (1, 101)
    assert result.frequency_resolution_hz == pytest.approx(1.0)
    np.testing.assert_allclose(result.frequencies_hz[[0, -1]], [0.0, 100.0])


@pytest.mark.parametrize(
    ("sample_count", "expected_windows"),
    [(10_000, 99), (10_247, 101)],
)
def test_default_window_counts(
    sample_count: int,
    expected_windows: int,
) -> None:
    result = compute_windowed_psd(
        _sine(5.0, sample_count=sample_count),
        sampling_rate_hz=200.0,
        window_seconds=1.0,
        overlap_ratio=0.5,
    )

    assert result.window_count == expected_windows


def test_incomplete_tail_is_dropped() -> None:
    result = compute_windowed_psd(
        _sine(5.0, sample_count=451),
        sampling_rate_hz=200.0,
        window_seconds=1.0,
        overlap_ratio=0.5,
    )

    np.testing.assert_array_equal(result.window_start_samples, [0, 100, 200])
    assert result.window_start_samples[-1] + result.window_size_samples == 400


def test_mean_removal_suppresses_large_dc_offset() -> None:
    result = compute_windowed_psd(
        1_000_000.0 + _sine(5.0),
        sampling_rate_hz=200.0,
        window_seconds=1.0,
        overlap_ratio=0.0,
    )

    assert result.psd[0, 0] < result.psd[0, 5] * 1e-8


def test_input_is_not_mutated() -> None:
    values = _sine(5.0)
    before = values.copy()

    compute_windowed_psd(
        values,
        sampling_rate_hz=200.0,
        window_seconds=1.0,
        overlap_ratio=0.5,
    )

    np.testing.assert_array_equal(values, before)


def test_even_nyquist_bin_is_not_doubled() -> None:
    values = (-1.0) ** np.arange(200)
    result = compute_windowed_psd(
        values,
        sampling_rate_hz=200.0,
        window_seconds=1.0,
        overlap_ratio=0.0,
    )
    hann = np.hanning(200)
    spectrum = np.fft.rfft((values - values.mean()) * hann)
    expected = np.abs(spectrum[-1]) ** 2 / (200.0 * np.sum(hann**2))

    assert result.psd[0, -1] == pytest.approx(expected)


def test_odd_one_sided_bins_are_doubled() -> None:
    values = _sine(20.0, sample_count=201)
    result = compute_windowed_psd(
        values,
        sampling_rate_hz=201.0,
        window_seconds=1.0,
        overlap_ratio=0.0,
    )
    hann = np.hanning(201)
    spectrum = np.fft.rfft((values - values.mean()) * hann)
    unscaled_last = np.abs(spectrum[-1]) ** 2 / (201.0 * np.sum(hann**2))

    assert result.psd[0, -1] == pytest.approx(2.0 * unscaled_last)


@pytest.mark.parametrize("overlap", [-0.1, 1.0, np.inf, np.nan])
def test_invalid_overlap_is_rejected(overlap: float) -> None:
    with pytest.raises(SpectralAnalysisError, match="overlap"):
        compute_windowed_psd(
            np.ones(200),
            sampling_rate_hz=200.0,
            window_seconds=1.0,
            overlap_ratio=overlap,
        )


@pytest.mark.parametrize(
    ("sampling_rate", "window_seconds"),
    [(0.0, 1.0), (-1.0, 1.0), (np.inf, 1.0), (200.0, 0.0), (200.0, np.nan)],
)
def test_invalid_rate_or_duration_is_rejected(
    sampling_rate: float,
    window_seconds: float,
) -> None:
    with pytest.raises(SpectralAnalysisError):
        compute_windowed_psd(
            np.ones(200),
            sampling_rate_hz=sampling_rate,
            window_seconds=window_seconds,
            overlap_ratio=0.5,
        )


def test_malformed_and_short_inputs_are_rejected() -> None:
    with pytest.raises(SpectralAnalysisError, match="one-dimensional"):
        compute_windowed_psd(
            np.ones((2, 200)),
            sampling_rate_hz=200.0,
            window_seconds=1.0,
            overlap_ratio=0.5,
        )
    with pytest.raises(SpectralAnalysisError, match="finite"):
        compute_windowed_psd(
            np.r_[np.ones(199), np.nan],
            sampling_rate_hz=200.0,
            window_seconds=1.0,
            overlap_ratio=0.5,
        )
    with pytest.raises(SpectralAnalysisError, match="fewer than"):
        compute_windowed_psd(
            np.ones(199),
            sampling_rate_hz=200.0,
            window_seconds=1.0,
            overlap_ratio=0.5,
        )


def test_three_panel_plot_layout_lines_ranges_and_units(tmp_path: Path) -> None:
    ring = _ring_data(tmp_path, sample_count=400)

    figure = plot_ring_acceleration_psd_overlay(ring, show=False)

    assert len(figure.axes) == 3
    assert [axis.get_title() for axis in figure.axes] == [
        "Acceleration X",
        "Acceleration Y",
        "Acceleration Z",
    ]
    assert all(axis.get_yscale() == "log" for axis in figure.axes)
    assert all(
        figure.axes[0].get_shared_x_axes().joined(figure.axes[0], axis)
        for axis in figure.axes[1:]
    )
    assert all(len(axis.lines) == 4 for axis in figure.axes)
    assert all(axis.get_xlim() == pytest.approx((0.0, 30.0)) for axis in figure.axes)
    assert len({axis.get_ylim() for axis in figure.axes}) == 1
    assert figure.axes[-1].get_xlabel() == "Frequency (Hz)"
    assert all("raw acceleration units²/Hz" in axis.get_ylabel() for axis in figure.axes)
    assert all("m/s" not in axis.get_ylabel() and axis.get_ylabel() != "g" for axis in figure.axes)
    plt.close(figure)


def test_explicit_acceleration_plot_uses_caller_labels() -> None:
    values = np.column_stack((_sine(5.0), _sine(10.0), _sine(20.0)))

    figure = plot_acceleration_psd_overlay(
        values,
        identity=" — test",
        source_label="Linear acceleration",
        psd_unit_label="(m/s^2)²/Hz",
        show=False,
    )

    assert [axis.get_title() for axis in figure.axes] == [
        "Linear acceleration X",
        "Linear acceleration Y",
        "Linear acceleration Z",
    ]
    assert all("(m/s^2)²/Hz" in axis.get_ylabel() for axis in figure.axes)
    assert "linear acceleration psd" in figure._suptitle.get_text().lower()
    plt.close(figure)


@pytest.mark.parametrize("method", ["mean", "median"])
def test_plotted_aggregate_is_numerically_correct(
    tmp_path: Path,
    method: str,
) -> None:
    values = np.concatenate(
        (_sine(5.0, amplitude=1.0), _sine(5.0, amplitude=2.0))
    )
    ring = _ring_data(
        tmp_path,
        sample_count=400,
        acceleration=(values, values, values),
    )
    expected_result = compute_windowed_psd(
        values,
        sampling_rate_hz=200.0,
        window_seconds=1.0,
        overlap_ratio=0.0,
    )
    expected = aggregate_windowed_psd(expected_result, method=method)

    figure = plot_ring_acceleration_psd_overlay(
        ring,
        overlap_ratio=0.0,
        aggregate=method,
        show=False,
    )

    aggregate_line = figure.axes[0].lines[-1]
    plotted_frequencies = aggregate_line.get_xdata()
    mask = (expected_result.frequencies_hz >= 0.0) & (
        expected_result.frequencies_hz <= 30.0
    )
    np.testing.assert_array_equal(
        plotted_frequencies,
        expected_result.frequencies_hz[mask],
    )
    positive = expected[mask] > 0.0
    np.testing.assert_allclose(
        aggregate_line.get_ydata()[positive],
        expected[mask][positive],
    )
    assert method.capitalize() in aggregate_line.get_label()
    plt.close(figure)


def test_one_combined_file_and_dataframe_unchanged(tmp_path: Path) -> None:
    ring = _ring_data(tmp_path)
    before = ring.dataframe.copy(deep=True)
    output = tmp_path / "nested" / "combined.png"

    figure = plot_ring_acceleration_psd_overlay(
        ring,
        output_path=output,
        show=False,
    )

    assert output.is_file() and output.stat().st_size > 0
    assert list(output.parent.iterdir()) == [output]
    pd.testing.assert_frame_equal(ring.dataframe, before)
    plt.close(figure)


def test_gyroscope_columns_are_not_analyzed(tmp_path: Path) -> None:
    ring = _ring_data(tmp_path)
    malformed_gyro = replace(
        ring,
        dataframe=ring.dataframe.assign(
            gyr_x=np.nan,
            gyr_y=np.inf,
            gyr_z=-np.inf,
        ),
    )

    figure = plot_ring_acceleration_psd_overlay(malformed_gyro, show=False)

    assert len(figure.axes) == 3
    plt.close(figure)


@pytest.mark.parametrize(
    ("minimum", "maximum", "message"),
    [
        (-1.0, 30.0, "negative"),
        (30.0, 30.0, "greater"),
        (0.0, 101.0, "Nyquist"),
        (np.nan, 30.0, "finite"),
    ],
)
def test_invalid_frequency_ranges_raise_clear_errors(
    tmp_path: Path,
    minimum: float,
    maximum: float,
    message: str,
) -> None:
    ring = _ring_data(tmp_path)

    with pytest.raises(InvalidFrequencyRangeError, match=message):
        plot_ring_acceleration_psd_overlay(
            ring,
            frequency_min_hz=minimum,
            frequency_max_hz=maximum,
            show=False,
        )


def test_all_zero_selected_spectra_raise_clear_error(tmp_path: Path) -> None:
    zeros = np.zeros(400)
    ring = _ring_data(
        tmp_path,
        acceleration=(zeros, zeros, zeros),
    )

    with pytest.raises(SpectralPlotError, match="no positive PSD"):
        plot_ring_acceleration_psd_overlay(ring, show=False)
