from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest

from scripts import plot_ring_linear_acceleration


def _data_root(tmp_path: Path) -> tuple[Path, Path]:
    root = tmp_path / "data"
    action = root / "writer_a" / "letters"
    action.mkdir(parents=True)
    sample_count = 120
    acceleration = np.tile([0.0, 0.0, 9.8], (sample_count, 1))
    gyroscope = np.zeros((sample_count, 3))
    timestamps = 1_000_000.0 + np.arange(sample_count) * 10_000.0
    rows = np.column_stack((acceleration, gyroscope, timestamps))
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
        "--sampling-rate",
        "100",
        "--calibration-min-samples",
        "10",
        "--no-show",
    ]


def test_cli_uses_primary_ring_only_and_writes_plot(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, action = _data_root(tmp_path)
    output = tmp_path / "plots" / "gravity.png"
    opened: list[Path] = []
    original = np.fromfile

    def tracked(path: str | Path, *args: object, **kwargs: object) -> np.ndarray:
        opened.append(Path(path))
        return original(path, *args, **kwargs)

    monkeypatch.setattr("writingring.ring_loader.np.fromfile", tracked)

    result = plot_ring_linear_acceleration.main(_args(root, output))

    assert result == 0
    assert opened == [action / "0_ring_0.bin"]
    assert output.is_file() and output.stat().st_size > 0
    assert matplotlib.get_backend().lower() == "agg"


def test_cli_summary_reports_assumptions(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    root, _ = _data_root(tmp_path)
    output = tmp_path / "gravity.png"

    result = plot_ring_linear_acceleration.main(_args(root, output))

    captured = capsys.readouterr()
    assert result == 0
    assert f"Saved: {output}" in captured.out
    assert "Calibration: 0:10 (passed)" in captured.out
    assert "Automatic stationary search: 0:10" in captured.out
    assert "Nominal sampling rate: 100 Hz (assumed)" in captured.out
    assert "Profile: explicit" in captured.out
    assert "Gravity removal method: madgwick" in captured.out
    assert "Madgwick beta: 0.1" in captured.out
    assert "offline estimate is noncausal" in captured.out


def test_cli_accepts_low_pass_method_and_custom_cutoff(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    root, _ = _data_root(tmp_path)
    args = [
        *_args(root, tmp_path / "gravity.png"),
        "--gravity-removal-method",
        "low-pass",
        "--low-pass-cutoff-hz",
        "0.5",
    ]

    assert plot_ring_linear_acceleration.main(args) == 0

    captured = capsys.readouterr()
    assert "Gravity removal method: low-pass" in captured.out
    assert "Low-pass cutoff: 0.5 Hz" in captured.out


def test_cli_can_disable_automatic_calibration_only_with_manual_bounds(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    root, _ = _data_root(tmp_path)
    args = [*_args(root, tmp_path / "gravity.png"), "--no-auto-calibration"]

    result = plot_ring_linear_acceleration.main(args)

    captured = capsys.readouterr()
    assert result == 2
    assert "automatic calibration is disabled" in captured.err
    assert "Traceback" not in captured.err


def test_cli_existing_output_requires_overwrite(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    root, _ = _data_root(tmp_path)
    output = tmp_path / "gravity.png"
    output.write_bytes(b"old")

    assert plot_ring_linear_acceleration.main(_args(root, output)) == 2
    assert output.read_bytes() == b"old"
    assert "--overwrite" in capsys.readouterr().err

    assert (
        plot_ring_linear_acceleration.main(
            [*_args(root, output), "--overwrite"]
        )
        == 0
    )
    assert output.read_bytes() != b"old"


def test_cli_no_show_closes_returned_figure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, _ = _data_root(tmp_path)
    runtime = plot_ring_linear_acceleration._load_runtime()
    original = runtime.plot_ring_gravity_removal
    figure_numbers: list[int] = []

    def tracked(*args: object, **kwargs: object) -> object:
        figure = original(*args, **kwargs)
        figure_numbers.append(figure.number)
        return figure

    runtime.plot_ring_gravity_removal = tracked
    monkeypatch.setattr(
        plot_ring_linear_acceleration,
        "_load_runtime",
        lambda: runtime,
    )

    assert (
        plot_ring_linear_acceleration.main(
            _args(root, tmp_path / "gravity.png")
        )
        == 0
    )
    assert figure_numbers
    assert all(not plt.fignum_exists(number) for number in figure_numbers)


def test_upstream_profile_is_available_without_implicit_raw_profile_scale(
    tmp_path: Path,
) -> None:
    root, _ = _data_root(tmp_path)
    args = _args(root, tmp_path / "gravity.png")
    args.extend(
        [
            "--profile",
            "upstream_suggested",
            "--calibration-start",
            "0",
            "--calibration-stop",
            "100",
        ]
    )

    assert plot_ring_linear_acceleration.main(args) == 0
