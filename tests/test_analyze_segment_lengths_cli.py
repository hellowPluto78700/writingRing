from __future__ import annotations

from pathlib import Path

from scripts import analyze_segment_lengths
from test_segment_padding import _write_package


def test_analysis_cli_writes_default_report_directory(tmp_path: Path, capsys) -> None:
    root = tmp_path / "segments"
    _write_package(root, user="user_a", action="one", lengths=[3, 5])

    assert analyze_segment_lengths.main(["--input-root", str(root), "--candidate-lengths", "4", "8"]) == 0
    assert (root / "padding_analysis" / "segment_length_analysis.json").is_file()
    assert "Pure-padding target: 5 samples" in capsys.readouterr().out


def test_analysis_cli_reports_bad_input_without_traceback(tmp_path: Path, capsys) -> None:
    assert analyze_segment_lengths.main(["--input-root", str(tmp_path / "absent")]) == 2
    assert "input root is not a directory" in capsys.readouterr().err
