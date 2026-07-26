from dataclasses import replace
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import compress_pickle
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from core.sensel_lib.frame_data import ContactData, FrameData
from writingring.board_loader import CONTACT_COLUMNS, BoardData, load_board
from writingring.plotting import (
    InferredTimeUnavailableError,
    MissingPlotColumnError,
    PlotOutputError,
    UnsupportedTimeAxisError,
    plot_board_force_over_time,
    plot_ring_imu,
    plot_touch_trajectory,
)
from writingring.ring_loader import RELATIVE_TIME_COLUMN, RingData, load_ring


def _write_ring(path: Path, timestamps: list[float]) -> RingData:
    rows = np.asarray(
        [
            [index + offset for offset in range(6)] + [timestamp]
            for index, timestamp in enumerate(timestamps)
        ],
        dtype=np.float64,
    )
    rows.tofile(path)
    return load_ring(path)


def _contact(
    *,
    x: float = 0.25,
    y: float = 0.75,
    force: float = 12.5,
) -> ContactData:
    return ContactData(
        id=3,
        state=2,
        x=x,
        y=y,
        area=4.0,
        force=force,
        major=2.0,
        minor=1.0,
        delta_x=0.1,
        delta_y=-0.1,
        delta_force=0.5,
        delta_area=0.25,
        label=0,
        frame_id=9,
    )


def _frame(timestamp: int, contacts: list[ContactData]) -> FrameData:
    frame = FrameData(np.zeros((2, 3), dtype=np.float64), timestamp)
    for contact in contacts:
        frame.append_contact(contact)
    return frame


def _write_chunk(path: Path, frames: list[FrameData]) -> Path:
    with path.open("wb") as output:
        compress_pickle.dump(frames, output)
    return path


def _board_data(tmp_path: Path) -> BoardData:
    chunk = _write_chunk(
        tmp_path / "0_board_0.gz",
        [
            _frame(100, []),
            _frame(200, [_contact(x=0.2, y=0.8, force=10.0)]),
            _frame(300, [_contact(x=0.6, y=0.3, force=30.0)]),
        ],
    )
    return load_board(chunk)


def _backward_board_data(tmp_path: Path) -> BoardData:
    chunk_6 = _write_chunk(
        tmp_path / "0_board_6.gz",
        [_frame(200, [_contact(force=20.0)])],
    )
    chunk_7 = _write_chunk(
        tmp_path / "0_board_7.gz",
        [_frame(100, [_contact(force=10.0)])],
    )
    return load_board((chunk_6, chunk_7))


def test_ring_plot_has_two_axes_and_expected_lines(tmp_path: Path) -> None:
    ring = _write_ring(
        tmp_path / "0_ring_0.bin",
        [1_000_000.0, 1_005_000.0],
    )

    figure = plot_ring_imu(ring, show=False)

    assert len(figure.axes) == 2
    assert [line.get_label() for line in figure.axes[0].lines] == [
        "acc_x",
        "acc_y",
        "acc_z",
    ]
    assert [line.get_label() for line in figure.axes[1].lines] == [
        "gyr_x",
        "gyr_y",
        "gyr_z",
    ]
    plt.close(figure)


def test_ring_sample_index_and_inferred_time_use_existing_axes(
    tmp_path: Path,
) -> None:
    ring = _write_ring(
        tmp_path / "0_ring_0.bin",
        [1_000_000.0, 1_005_000.0, 1_010_000.0],
    )
    sample_figure = plot_ring_imu(
        ring,
        time_axis="sample_index",
        show=False,
    )
    inferred_figure = plot_ring_imu(ring, show=False)

    np.testing.assert_array_equal(
        sample_figure.axes[0].lines[0].get_xdata(),
        ring.dataframe.index.to_numpy(),
    )
    np.testing.assert_array_equal(
        inferred_figure.axes[0].lines[0].get_xdata(),
        ring.dataframe[RELATIVE_TIME_COLUMN].to_numpy(),
    )
    assert "unconfirmed" in inferred_figure.axes[1].get_xlabel()
    plt.close(sample_figure)
    plt.close(inferred_figure)


def test_ring_inferred_time_fails_when_column_unavailable(
    tmp_path: Path,
) -> None:
    ring = _write_ring(tmp_path / "0_ring_0.bin", [1_000_000.0])
    without_inferred = replace(
        ring,
        dataframe=ring.dataframe.drop(columns=RELATIVE_TIME_COLUMN),
    )

    with pytest.raises(InferredTimeUnavailableError, match="sample_index"):
        plot_ring_imu(without_inferred, show=False)


def test_ring_plot_does_not_modify_input_dataframe(tmp_path: Path) -> None:
    ring = _write_ring(
        tmp_path / "0_ring_0.bin",
        [1_000_000.0, 1_000_000.0, 1_010_000.0],
    )
    before = ring.dataframe.copy(deep=True)

    figure = plot_ring_imu(ring, show=False)

    pd.testing.assert_frame_equal(ring.dataframe, before)
    assert any(
        "duplicate timestamp" in text.get_text() for text in figure.texts
    )
    plt.close(figure)


def test_touch_uses_x_y_display_equal_aspect_and_force_colorbar(
    tmp_path: Path,
) -> None:
    board = _board_data(tmp_path)
    before = board.contacts.copy(deep=True)

    figure = plot_touch_trajectory(board, show=False)

    offsets = figure.axes[0].collections[0].get_offsets()
    np.testing.assert_allclose(offsets[:, 0], board.contacts["x"])
    np.testing.assert_allclose(offsets[:, 1], board.contacts["y_display"])
    assert figure.axes[0].get_aspect() == pytest.approx(1.0)
    assert len(figure.axes) == 2
    assert "force" in figure.axes[1].get_ylabel().lower()
    pd.testing.assert_frame_equal(board.contacts, before)
    plt.close(figure)


def test_empty_contacts_create_explanatory_figures(tmp_path: Path) -> None:
    chunk = _write_chunk(tmp_path / "0_board_0.gz", [_frame(100, [])])
    board = load_board(chunk)

    touch = plot_touch_trajectory(board, show=False)
    force = plot_board_force_over_time(board, show=False)

    assert "No board contacts" in touch.axes[0].texts[0].get_text()
    assert "No board contacts" in force.axes[0].texts[0].get_text()
    plt.close(touch)
    plt.close(force)


def test_board_force_frame_index_mode(tmp_path: Path) -> None:
    board = _board_data(tmp_path)

    figure = plot_board_force_over_time(board, show=False)
    offsets = figure.axes[0].collections[0].get_offsets()

    np.testing.assert_array_equal(
        offsets[:, 0],
        board.contacts["global_frame_index"].to_numpy(),
    )
    np.testing.assert_allclose(offsets[:, 1], board.contacts["force"])
    plt.close(figure)


def test_raw_timestamp_preserves_backward_order_and_warning(
    tmp_path: Path,
) -> None:
    board = _backward_board_data(tmp_path)
    before = board.contacts.copy(deep=True)

    figure = plot_board_force_over_time(
        board,
        time_axis="raw_timestamp",
        show=False,
    )
    offsets = figure.axes[0].collections[0].get_offsets()

    np.testing.assert_array_equal(offsets[:, 0], [200, 100])
    assert any("6->7" in text.get_text() for text in figure.texts)
    pd.testing.assert_frame_equal(board.contacts, before)
    plt.close(figure)


def test_touch_warning_discloses_backward_boundary(tmp_path: Path) -> None:
    board = _backward_board_data(tmp_path)

    figure = plot_touch_trajectory(board, show=False)

    assert any("6->7" in text.get_text() for text in figure.texts)
    plt.close(figure)


def test_output_files_and_parent_directories_are_created(
    tmp_path: Path,
) -> None:
    ring = _write_ring(
        tmp_path / "0_ring_0.bin",
        [1_000_000.0, 1_005_000.0],
    )
    output = tmp_path / "nested" / "ring.png"

    figure = plot_ring_imu(ring, output_path=output, show=False)

    assert output.is_file()
    assert output.stat().st_size > 0
    plt.close(figure)


@pytest.mark.parametrize(
    ("function_name", "time_axis"),
    [
        ("ring", "raw_timestamp"),
        ("board", "inferred_time"),
    ],
)
def test_unsupported_time_modes_raise_clear_error(
    tmp_path: Path,
    function_name: str,
    time_axis: str,
) -> None:
    if function_name == "ring":
        data = _write_ring(tmp_path / "0_ring_0.bin", [1_000_000.0])
        function = plot_ring_imu
    else:
        data = _board_data(tmp_path)
        function = plot_board_force_over_time

    with pytest.raises(UnsupportedTimeAxisError, match="unsupported"):
        function(data, time_axis=time_axis, show=False)


def test_missing_required_columns_raise_plotting_error(tmp_path: Path) -> None:
    board = _board_data(tmp_path)
    malformed = replace(board, contacts=board.contacts.drop(columns="force"))

    with pytest.raises(MissingPlotColumnError, match="force"):
        plot_touch_trajectory(malformed, show=False)


def test_directory_output_path_raises_plotting_error(tmp_path: Path) -> None:
    ring = _write_ring(tmp_path / "0_ring_0.bin", [1_000_000.0])
    directory = tmp_path / "figures"
    directory.mkdir()

    with pytest.raises(PlotOutputError, match="regular file"):
        plot_ring_imu(ring, output_path=directory, show=False)


def test_empty_contact_schema_from_loader_is_supported(tmp_path: Path) -> None:
    path = _write_chunk(tmp_path / "0_board_0.gz", [])
    board = load_board(path)

    assert list(board.contacts.columns) == list(CONTACT_COLUMNS)
    figure = plot_touch_trajectory(board, show=False)

    assert figure.axes
    plt.close(figure)
