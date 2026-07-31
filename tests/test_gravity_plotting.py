from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from writingring.gravity import GravityRemovalConfig, process_ring_gravity
from writingring.plotting import (
    InferredTimeUnavailableError,
    MalformedPlotDataError,
    plot_ring_gravity_removal,
)
from writingring.ring_loader import RELATIVE_TIME_COLUMN, load_ring


def _ring_and_result(
    tmp_path: Path,
    *,
    provisional: bool = False,
) -> tuple[object, object]:
    sample_count = 100
    acceleration = np.tile([0.0, 0.0, 9.8], (sample_count, 1))
    gyroscope = np.zeros((sample_count, 3))
    if provisional:
        gyroscope[:30, 0] = 1.0
    timestamps = 1_000_000.0 + np.arange(sample_count) * 10_000.0
    rows = np.column_stack((acceleration, gyroscope, timestamps))
    path = tmp_path / "0_ring_0.bin"
    rows.astype(np.float64).tofile(path)
    ring = load_ring(path)
    config = GravityRemovalConfig(
        sampling_rate_hz=100.0,
        gyro_scale_to_rad_s=1.0,
        calibration_start_sample=0,
        calibration_stop_sample=30,
        strict_calibration=not provisional,
    )
    return ring, process_ring_gravity(ring, config=config)


def test_gravity_plot_has_expected_panels_lines_and_labels(tmp_path: Path) -> None:
    ring, result = _ring_and_result(tmp_path)
    before = ring.dataframe.copy(deep=True)

    figure = plot_ring_gravity_removal(ring, result, show=False)

    assert len(figure.axes) == 4
    assert [len(axis.lines) for axis in figure.axes] == [3, 3, 3, 2]
    assert [line.get_label() for line in figure.axes[1].lines] == [
        "gravity_body_x",
        "gravity_body_y",
        "gravity_body_z",
    ]
    assert "m/s^2" in figure.axes[2].get_ylabel()
    assert "unconfirmed" in figure.axes[-1].get_xlabel()
    assert "offline bidirectional" in figure._suptitle.get_text()
    pd.testing.assert_frame_equal(ring.dataframe, before)
    plt.close(figure)


def test_gravity_plot_sample_index_uses_existing_index(tmp_path: Path) -> None:
    ring, result = _ring_and_result(tmp_path)

    figure = plot_ring_gravity_removal(
        ring,
        result,
        time_axis="sample_index",
        show=False,
    )

    np.testing.assert_array_equal(
        figure.axes[0].lines[0].get_xdata(),
        ring.dataframe.index.to_numpy(),
    )
    plt.close(figure)


def test_gravity_plot_shows_provisional_warning(tmp_path: Path) -> None:
    ring, result = _ring_and_result(tmp_path, provisional=True)

    figure = plot_ring_gravity_removal(ring, result, show=False)

    assert "PROVISIONAL" in figure._suptitle.get_text()
    assert any("provisional" in text.get_text().lower() for text in figure.texts)
    plt.close(figure)


def test_gravity_plot_validates_time_and_sample_count(tmp_path: Path) -> None:
    ring, result = _ring_and_result(tmp_path)
    without_time = replace(
        ring,
        dataframe=ring.dataframe.drop(columns=RELATIVE_TIME_COLUMN),
    )

    with pytest.raises(InferredTimeUnavailableError):
        plot_ring_gravity_removal(without_time, result, show=False)

    shorter = replace(
        ring,
        dataframe=ring.dataframe.iloc[:-1].copy(),
    )
    with pytest.raises(MalformedPlotDataError, match="sample count"):
        plot_ring_gravity_removal(shorter, result, show=False)


def test_gravity_plot_saves_output(tmp_path: Path) -> None:
    ring, result = _ring_and_result(tmp_path)
    output = tmp_path / "nested" / "gravity.png"

    figure = plot_ring_gravity_removal(
        ring,
        result,
        output_path=output,
        show=False,
    )

    assert output.is_file() and output.stat().st_size > 0
    plt.close(figure)
