from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest

from writingring.spectral import (
    FrequencySupport,
    WindowedPSD,
    compute_frequency_support,
    plot_ring_acceleration_frequency_support,
)


def _synthetic_psd() -> WindowedPSD:
    frequencies = np.arange(0.0, 21.0)
    baseline = np.linspace(0.1, 1.0, frequencies.size)
    psd = np.repeat(baseline[np.newaxis, :], 6, axis=0)
    psd[:, 5] = 10.0
    psd[0, 8] = 20.0
    return WindowedPSD(
        frequencies_hz=frequencies,
        psd=psd,
        window_size_samples=20,
        hop_size_samples=10,
        window_start_samples=np.arange(6) * 10,
    )


def test_frequency_support_calculation() -> None:
    result = compute_frequency_support(
        _synthetic_psd(),
        frequency_min_hz=1.0,
        frequency_max_hz=20.0,
        high_power_quantile=0.90,
        smoothing_bins=1,
    )

    np.testing.assert_allclose(result.relative_power.sum(axis=1), 1.0)
    np.testing.assert_array_equal(
        result.support_count,
        np.count_nonzero(result.high_power, axis=0),
    )
    assert np.all(
        (result.support_percent >= 0.0)
        & (result.support_percent <= 100.0)
    )
    five_hz = int(np.flatnonzero(result.frequencies_hz == 5.0)[0])
    eight_hz = int(np.flatnonzero(result.frequencies_hz == 8.0)[0])
    assert result.support_percent[five_hz] == 100.0
    assert result.support_percent[eight_hz] == pytest.approx(100.0 / 6.0)
    assert result.support_percent[eight_hz] < 20.0


def test_frequency_support_figure_is_three_linear_scatter_axes(
    tmp_path: Path,
) -> None:
    support = compute_frequency_support(
        _synthetic_psd(),
        frequency_min_hz=1.0,
        frequency_max_hz=20.0,
        smoothing_bins=3,
    )
    supports: dict[str, FrequencySupport] = {
        "acc_x": support,
        "acc_y": support,
        "acc_z": support,
    }
    output = tmp_path / "support.png"

    figure = plot_ring_acceleration_frequency_support(
        supports,
        identity="user_0 / action 0 / dataset 0",
        sampling_rate_hz=200.0,
        window_seconds=1.0,
        overlap_ratio=0.5,
        output_path=output,
        show=False,
    )

    assert len(figure.axes) == 3
    assert all(axis.get_yscale() == "linear" for axis in figure.axes)
    assert all(axis.get_ylim() == (0.0, 100.0) for axis in figure.axes)
    assert all(not axis.lines for axis in figure.axes)
    assert all(axis.collections for axis in figure.axes)
    assert not any(
        "mean" in text.get_text().lower() or "median" in text.get_text().lower()
        for axis in figure.axes
        for text in axis.texts
    )
    assert output.is_file() and output.stat().st_size > 0
    plt.close(figure)
