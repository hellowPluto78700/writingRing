from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest

from scripts import plot_ring_accel_spectrum


def _data_root(
    tmp_path: Path,
    *,
    sample_count: int = 400,
) -> tuple[Path, Path]:
    root = tmp_path / "data"
    action = root / "writer_a" / "letters"
    action.mkdir(parents=True)
    time = np.arange(sample_count, dtype=np.float64) / 200.0
    rows = np.column_stack(
        (
            np.sin(2 * np.pi * 5 * time),
            np.sin(2 * np.pi * 10 * time),
            np.sin(2 * np.pi * 20 * time),
            np.zeros(sample_count),
            np.zeros(sample_count),
            np.zeros(sample_count),
            1_000_000.0 + np.arange(sample_count) * 5_000.0,
        )
    )
    rows.astype(np.float64).tofile(action / "0_ring_0.bin")
    (rows + 100.0).astype(np.float64).tofile(action / "0_ring_1.bin")
    (action / "0_board_0.gz").write_bytes(b"must not be opened")
    return root, action


def _args(root: Path, output: Path) -> list[str]:
    return [
        "--data-root",
        str(root),
        "--user",
        "writer_a",
        "--action",
        "letters",
        "--dataset-id",
        "0",
        "--output",
        str(output),
        "--no-show",
    ]


def _stationary_data_root(tmp_path: Path) -> Path:
    root = tmp_path / "gravity_data"
    action = root / "writer_a" / "letters"
    action.mkdir(parents=True)
    sample_count = 400
    time = np.arange(sample_count, dtype=np.float64) / 200.0
    acceleration = np.tile([0.0, 0.0, 9.80665], (sample_count, 1))
    acceleration[100:, 0] += np.sin(2 * np.pi * 5 * time[100:])
    acceleration[100:, 1] += np.sin(2 * np.pi * 10 * time[100:])
    acceleration[100:, 2] += np.sin(2 * np.pi * 20 * time[100:])
    rows = np.column_stack(
        (
            acceleration,
            np.zeros((sample_count, 3)),
            1_000_000.0 + np.arange(sample_count) * 5_000.0,
        )
    )
    rows.astype(np.float64).tofile(action / "0_ring_0.bin")
    return root


def test_cli_uses_primary_ring_only_and_never_loads_board(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, action = _data_root(tmp_path)
    output = tmp_path / "plots" / "spectrum.png"
    opened: list[Path] = []
    original = np.fromfile

    def tracked(path: str | Path, *args: object, **kwargs: object) -> np.ndarray:
        opened.append(Path(path))
        return original(path, *args, **kwargs)

    monkeypatch.setattr("writingring.ring_loader.np.fromfile", tracked)

    result = plot_ring_accel_spectrum.main(_args(root, output))

    assert result == 0
    assert opened == [action / "0_ring_0.bin"]
    assert output.is_file() and output.stat().st_size > 0
    assert matplotlib.get_backend().lower() == "agg"


def test_cli_no_show_closes_returned_figure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, _ = _data_root(tmp_path)
    output = tmp_path / "spectrum.png"
    runtime = plot_ring_accel_spectrum._load_runtime()
    original = runtime.plot_ring_acceleration_psd_overlay
    figure_numbers: list[int] = []

    def tracked(*args: object, **kwargs: object) -> object:
        figure = original(*args, **kwargs)
        figure_numbers.append(figure.number)
        return figure

    runtime.plot_ring_acceleration_psd_overlay = tracked
    monkeypatch.setattr(
        plot_ring_accel_spectrum,
        "_load_runtime",
        lambda: runtime,
    )

    assert plot_ring_accel_spectrum.main(_args(root, output)) == 0
    assert figure_numbers
    assert all(not plt.fignum_exists(number) for number in figure_numbers)


def test_existing_output_requires_overwrite(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    root, _ = _data_root(tmp_path)
    output = tmp_path / "spectrum.png"
    output.write_bytes(b"old")

    result = plot_ring_accel_spectrum.main(_args(root, output))

    assert result != 0
    assert output.read_bytes() == b"old"
    assert "--overwrite" in capsys.readouterr().err

    result = plot_ring_accel_spectrum.main(
        [*_args(root, output), "--overwrite"]
    )

    assert result == 0
    assert output.read_bytes() != b"old"


@pytest.mark.parametrize(
    ("extra_args", "message"),
    [
        (["--overlap", "1"], "overlap"),
        (["--frequency-min", "30", "--frequency-max", "20"], "frequency"),
    ],
)
def test_invalid_cli_analysis_is_concise(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    extra_args: list[str],
    message: str,
) -> None:
    root, _ = _data_root(tmp_path)

    result = plot_ring_accel_spectrum.main(
        [*_args(root, tmp_path / "spectrum.png"), *extra_args]
    )

    captured = capsys.readouterr()
    assert result != 0
    assert message in captured.err.lower()
    assert "Traceback" not in captured.err


def test_too_short_cli_input_is_concise(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    root, _ = _data_root(tmp_path, sample_count=100)

    result = plot_ring_accel_spectrum.main(
        _args(root, tmp_path / "spectrum.png")
    )

    captured = capsys.readouterr()
    assert result != 0
    assert "fewer than one complete" in captured.err
    assert "Traceback" not in captured.err


def test_success_summary_uses_calculated_metadata(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    root, _ = _data_root(tmp_path, sample_count=450)
    output = tmp_path / "spectrum.png"

    result = plot_ring_accel_spectrum.main(
        [
            *_args(root, output),
            "--window-seconds",
            "0.5",
            "--overlap",
            "0",
            "--aggregate",
            "median",
            "--frequency-max",
            "40",
        ]
    )

    captured = capsys.readouterr()
    assert result == 0
    assert f"Saved: {output}" in captured.out
    assert "Samples: 450" in captured.out
    assert "Window size: 100 samples" in captured.out
    assert "Hop size: 100 samples" in captured.out
    assert "Windows: 4" in captured.out
    assert "Frequency resolution: 2 Hz" in captured.out
    assert "Displayed range: 0–40 Hz" in captured.out
    assert "Aggregate: median" in captured.out


@pytest.mark.parametrize("plot_mode", ("psd", "support"))
def test_remove_gravity_uses_automatic_stationary_calibration(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    plot_mode: str,
) -> None:
    root = _stationary_data_root(tmp_path)
    output = tmp_path / f"linear-{plot_mode}.png"

    result = plot_ring_accel_spectrum.main(
        [
            *_args(root, output),
            "--remove-gravity",
            "--plot-mode",
            plot_mode,
        ]
    )

    captured = capsys.readouterr()
    assert result == 0
    assert output.is_file() and output.stat().st_size > 0
    assert "Analysis input: linear acceleration" in captured.out
    assert "Automatic stationary search: 0:20" in captured.out
    assert "Calibration: 0:20 (passed)" in captured.out


def test_remove_gravity_manual_interval_overrides_search(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    root = _stationary_data_root(tmp_path)

    result = plot_ring_accel_spectrum.main(
        [
            *_args(root, tmp_path / "manual.png"),
            "--remove-gravity",
            "--calibration-start",
            "20",
            "--calibration-stop",
            "40",
            "--no-auto-calibration",
        ]
    )

    captured = capsys.readouterr()
    assert result == 0
    assert "Calibration selection: manual interval" in captured.out
    assert "Automatic stationary search" not in captured.out


def test_remove_gravity_requires_automatic_or_manual_calibration(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    root = _stationary_data_root(tmp_path)

    result = plot_ring_accel_spectrum.main(
        [
            *_args(root, tmp_path / "disabled.png"),
            "--remove-gravity",
            "--no-auto-calibration",
        ]
    )

    captured = capsys.readouterr()
    assert result == 2
    assert "automatic calibration is disabled" in captured.err
    assert "Traceback" not in captured.err


def test_remove_gravity_upstream_profile_requires_manual_interval(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    root = _stationary_data_root(tmp_path)

    result = plot_ring_accel_spectrum.main(
        [
            *_args(root, tmp_path / "upstream.png"),
            "--remove-gravity",
            "--profile",
            "upstream_suggested",
        ]
    )

    captured = capsys.readouterr()
    assert result == 2
    assert "manual calibration" in captured.err
    assert "Traceback" not in captured.err
