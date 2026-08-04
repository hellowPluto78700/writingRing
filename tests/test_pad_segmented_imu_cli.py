from __future__ import annotations

from pathlib import Path

import numpy as np

from scripts import analyze_segment_lengths, pad_segmented_imu
from test_segment_padding import _write_package


def test_padding_cli_uses_fresh_pure_padding_report(tmp_path: Path, capsys) -> None:
    root = tmp_path / "segments"
    _write_package(root, user="user_a", action="one", lengths=[3, 5])
    assert analyze_segment_lengths.main(["--input-root", str(root)]) == 0
    report = root / "padding_analysis" / "segment_length_analysis.json"
    output = tmp_path / "fixed"

    assert pad_segmented_imu.main([
        "--input-root", str(root), "--output-root", str(output),
        "--analysis-report", str(report), "--recommendation", "pure-padding",
    ]) == 0
    assert (output / "padding_dataset_summary.json").is_file()
    assert "Published 2 padded segments" in capsys.readouterr().out


def test_padding_cli_skips_overflow_and_publishes_remaining_segments(tmp_path: Path, capsys) -> None:
    root = tmp_path / "segments"
    _write_package(root, user="user_a", action="one", lengths=[3, 5])
    output = tmp_path / "fixed"

    assert pad_segmented_imu.main([
        "--input-root", str(root), "--output-root", str(output), "--target-length", "4",
    ]) == 0
    padded = np.load(
        output / "user_a" / "action_one" / "user_a_action_one_paddedIMU.npy",
        allow_pickle=False,
    )
    assert padded.shape == (1, 4, 6)
    assert "skipped 1 overlong segments" in capsys.readouterr().out
