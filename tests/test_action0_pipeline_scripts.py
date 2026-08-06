from __future__ import annotations

from pathlib import Path
import subprocess


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PIPELINE_ROOT = PROJECT_ROOT / "scripts" / "action0_pipeline"


def test_all_action0_entry_points_use_shared_padding_pipeline() -> None:
    entry_points = sorted(PIPELINE_ROOT.glob("[0-9][0-9]_*.sh"))

    assert len(entry_points) == 8
    for entry_point in entry_points:
        text = entry_point.read_text(encoding="utf-8")
        assert 'source "$SCRIPT_DIR/_common.bash"' in text
        completed = subprocess.run(
            ["bash", "-n", str(entry_point)],
            check=False,
            capture_output=True,
            text=True,
        )
        assert completed.returncode == 0, completed.stderr

    common = (PIPELINE_ROOT / "_common.bash").read_text(encoding="utf-8")
    assert "scripts/analyze_segment_lengths.py" in common
    assert "scripts/pad_segmented_imu.py" in common
    assert '--minimum-coverage "$PADDING_COVERAGE"' in common
    assert 'PADDING_COVERAGE="${PADDING_COVERAGE:-0.99}"' in common
    assert 'PADDING_RECOMMENDATION="${PADDING_RECOMMENDATION:-balanced}"' in common
    assert "pipeline_padding" in common

