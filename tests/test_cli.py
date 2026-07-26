from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import matplotlib

matplotlib.use("Agg")

import compress_pickle
import matplotlib.pyplot as plt
import numpy as np
import pytest

from core.sensel_lib.frame_data import ContactData, FrameData
from scripts import inspect_recording, plot_recording
from writingring.plotting import PlottingError


def _contact(force: float = 10.0) -> ContactData:
    return ContactData(
        id=1,
        state=2,
        x=0.25,
        y=0.75,
        area=4.0,
        force=force,
        major=2.0,
        minor=1.0,
        delta_x=0.0,
        delta_y=0.0,
        delta_force=0.0,
        delta_area=0.0,
        label=0,
        frame_id=1,
    )


def _frame(timestamp: int, force: float = 10.0) -> FrameData:
    frame = FrameData(np.zeros((2, 3), dtype=np.float64), timestamp)
    frame.append_contact(_contact(force))
    return frame


def _write_chunk(path: Path, timestamp: int, force: float) -> Path:
    with path.open("wb") as output:
        compress_pickle.dump([_frame(timestamp, force)], output)
    return path


def _data_root(tmp_path: Path) -> tuple[Path, Path]:
    root = tmp_path / "data"
    action = root / "writer_a" / "letters"
    action.mkdir(parents=True)
    ring_rows = np.asarray(
        [
            [0, 1, 2, 3, 4, 5, 1_000_000],
            [1, 2, 3, 4, 5, 6, 1_000_000],
            [2, 3, 4, 5, 6, 7, 1_010_000],
        ],
        dtype=np.float64,
    )
    ring_rows.tofile(action / "0_ring_0.bin")
    (ring_rows + 100).tofile(action / "0_ring_1.bin")
    (action / "0_timestamp.txt").write_text("1000000 a\n", encoding="utf-8")

    _write_chunk(action / "0_board_10.gz", 300, 40.0)
    _write_chunk(action / "0_board_2.gz", 100, 10.0)
    _write_chunk(action / "0_board_7.gz", 50, 30.0)
    _write_chunk(action / "0_board_6.gz", 200, 20.0)
    return root, action


def _selector_args(data_root: Path) -> list[str]:
    return [
        "--data-root",
        str(data_root),
        "--user",
        "writer_a",
        "--action",
        "letters",
        "--dataset-id",
        "0",
    ]


def test_inspect_cli_text_and_json_output(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    data_root, _ = _data_root(tmp_path)
    json_path = tmp_path / "inspection" / "summary.json"

    result = inspect_recording.main(
        [*_selector_args(data_root), "--json-output", str(json_path)]
    )

    captured = capsys.readouterr()
    assert result == 0
    assert "Recording" in captured.out
    assert "raw DataFrame shape: (3, 7)" in captured.out
    assert "chunk-index range: 2-10" in captured.out
    assert "6->7 (delta=-150.0)" in captured.out
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload["recording"]["dataset_id"] == 0
    assert payload["ring"]["duplicate_timestamp_steps"] == 1
    assert payload["board"]["chunk_indices"] == [2, 6, 7, 10]
    assert payload["board"]["cross_chunk_backward_boundaries"][0][
        "previous_chunk_index"
    ] == 6
    assert "synchronization" in payload["notes"]


def test_inspect_cli_missing_recording_is_nonzero(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    data_root, _ = _data_root(tmp_path)
    args = _selector_args(data_root)
    args[-1] = "99"

    result = inspect_recording.main(args)

    captured = capsys.readouterr()
    assert result != 0
    assert "error: no recording matches" in captured.err
    assert "Traceback" not in captured.err


def test_inspect_cli_nonexistent_root_is_nonzero(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    result = inspect_recording.main(
        _selector_args(tmp_path / "does-not-exist")
    )

    captured = capsys.readouterr()
    assert result != 0
    assert "data root does not exist" in captured.err
    assert "Traceback" not in captured.err


def test_malformed_dataset_selector_is_process_level_nonzero() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "scripts/inspect_recording.py",
            "--data-root",
            "unused",
            "--user",
            "writer_a",
            "--action",
            "letters",
            "--dataset-id",
            "not-an-integer",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode != 0
    assert "invalid int value" in completed.stderr
    assert "Traceback" not in completed.stderr


def test_inspect_cli_ring_failure_is_nonzero(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    data_root, action = _data_root(tmp_path)
    (action / "0_ring_0.bin").write_bytes(b"")

    result = inspect_recording.main(_selector_args(data_root))

    captured = capsys.readouterr()
    assert result != 0
    assert "ring file is empty" in captured.err


def test_inspect_cli_never_opens_ring_1(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data_root, action = _data_root(tmp_path)
    opened: list[Path] = []
    original = np.fromfile

    def tracked(path: str | Path, *args: object, **kwargs: object) -> np.ndarray:
        opened.append(Path(path))
        return original(path, *args, **kwargs)

    monkeypatch.setattr("writingring.ring_loader.np.fromfile", tracked)

    assert inspect_recording.main(_selector_args(data_root)) == 0
    assert opened == [action / "0_ring_0.bin"]


def test_plot_cli_creates_outputs_forwards_modes_and_closes_figures(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    data_root, _ = _data_root(tmp_path)
    output_dir = tmp_path / "plots"
    runtime = plot_recording._load_runtime()
    original_ring_plot = runtime.plot_ring_imu
    original_force_plot = runtime.plot_board_force_over_time
    seen: dict[str, object] = {}
    figure_numbers: list[int] = []

    def tracked_ring(*args: object, **kwargs: object) -> object:
        seen["ring_time_axis"] = kwargs["time_axis"]
        seen["ring_show"] = kwargs["show"]
        figure = original_ring_plot(*args, **kwargs)
        figure_numbers.append(figure.number)
        return figure

    def tracked_force(*args: object, **kwargs: object) -> object:
        seen["board_time_axis"] = kwargs["time_axis"]
        seen["board_show"] = kwargs["show"]
        figure = original_force_plot(*args, **kwargs)
        figure_numbers.append(figure.number)
        return figure

    runtime.plot_ring_imu = tracked_ring
    runtime.plot_board_force_over_time = tracked_force
    monkeypatch.setattr(plot_recording, "_load_runtime", lambda: runtime)

    result = plot_recording.main(
        [
            *_selector_args(data_root),
            "--output-dir",
            str(output_dir),
            "--ring-time-axis",
            "sample_index",
            "--board-time-axis",
            "raw_timestamp",
            "--no-show",
        ]
    )

    assert result == 0
    assert seen == {
        "ring_time_axis": "sample_index",
        "ring_show": False,
        "board_time_axis": "raw_timestamp",
        "board_show": False,
    }
    assert matplotlib.get_backend().lower() == "agg"
    assert all(not plt.fignum_exists(number) for number in figure_numbers)
    expected = {
        "ring_imu.png",
        "touch_trajectory.png",
        "board_force_time.png",
        "summary.json",
    }
    assert {path.name for path in output_dir.iterdir()} == expected
    assert all(path.stat().st_size > 0 for path in output_dir.iterdir())
    captured = capsys.readouterr()
    assert all(name in captured.out for name in expected)

    summary = json.loads(
        (output_dir / "summary.json").read_text(encoding="utf-8")
    )
    assert summary["plotting"]["ring_time_axis"] == "sample_index"
    assert summary["plotting"]["board_time_axis"] == "raw_timestamp"
    assert summary["board"]["chunk_indices"] == [2, 6, 7, 10]
    assert any(
        "6->7" in warning for warning in summary["board"]["warnings"]
    )
    assert "not assumed synchronized" in summary["notes"]["synchronization"]


def test_plot_cli_opens_all_board_chunks_in_numeric_order_and_not_ring_1(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data_root, action = _data_root(tmp_path)
    output_dir = tmp_path / "plots"
    ring_paths: list[Path] = []
    board_paths: list[Path] = []
    original_ring = np.fromfile
    original_board = compress_pickle.load

    def tracked_ring(
        path: str | Path,
        *args: object,
        **kwargs: object,
    ) -> np.ndarray:
        ring_paths.append(Path(path))
        return original_ring(path, *args, **kwargs)

    def tracked_board(
        path: str | Path,
        *args: object,
        **kwargs: object,
    ) -> object:
        board_paths.append(Path(path))
        return original_board(path, *args, **kwargs)

    monkeypatch.setattr("writingring.ring_loader.np.fromfile", tracked_ring)
    monkeypatch.setattr(
        "writingring.board_loader.compress_pickle.load",
        tracked_board,
    )

    result = plot_recording.main(
        [
            *_selector_args(data_root),
            "--output-dir",
            str(output_dir),
            "--no-show",
        ]
    )

    assert result == 0
    assert ring_paths == [action / "0_ring_0.bin"]
    assert board_paths == [
        action / "0_board_2.gz",
        action / "0_board_6.gz",
        action / "0_board_7.gz",
        action / "0_board_10.gz",
    ]


def test_plot_cli_plotting_failure_is_nonzero_and_closes_prior_figure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    data_root, _ = _data_root(tmp_path)
    runtime = plot_recording._load_runtime()
    original_ring_plot = runtime.plot_ring_imu
    figure_number: list[int] = []

    def tracked_ring(*args: object, **kwargs: object) -> object:
        figure = original_ring_plot(*args, **kwargs)
        figure_number.append(figure.number)
        return figure

    def fail_touch(*args: object, **kwargs: object) -> object:
        raise PlottingError("synthetic plotting failure")

    runtime.plot_ring_imu = tracked_ring
    runtime.plot_touch_trajectory = fail_touch
    monkeypatch.setattr(plot_recording, "_load_runtime", lambda: runtime)

    result = plot_recording.main(
        [
            *_selector_args(data_root),
            "--output-dir",
            str(tmp_path / "plots"),
            "--no-show",
        ]
    )

    captured = capsys.readouterr()
    assert result != 0
    assert "error: synthetic plotting failure" in captured.err
    assert all(not plt.fignum_exists(number) for number in figure_number)


def test_plot_cli_output_directory_failure_is_nonzero(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    data_root, _ = _data_root(tmp_path)
    output_path = tmp_path / "not-a-directory"
    output_path.write_text("occupied", encoding="utf-8")

    result = plot_recording.main(
        [
            *_selector_args(data_root),
            "--output-dir",
            str(output_path),
            "--no-show",
        ]
    )

    captured = capsys.readouterr()
    assert result != 0
    assert "output path is not a directory" in captured.err
