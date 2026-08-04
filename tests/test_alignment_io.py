from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from writingring.alignment_io import (
    AlignmentOffsetExportError,
    apply_board_to_ring_offset,
    build_alignment_offset_path,
    extract_alignment_offset,
    read_alignment_offset_txt,
    write_alignment_offset_txt,
)
from writingring.event_alignment import SequenceAlignmentResult


def _result(*, success: bool = True, offset_us: float = -238_451.75) -> SequenceAlignmentResult:
    return SequenceAlignmentResult(
        success=success,
        best_offset_us=offset_us,
        second_best_offset_us=None,
        candidates=pd.DataFrame(),
        event_matches=pd.DataFrame(),
        touch_pair_matches=pd.DataFrame(),
        report={
            "alignment_model": "constant_offset",
            "event_coverage_ratio": 0.875,
            "matched_event_count": 35,
            "total_valid_event_count": 40,
        },
        warnings=(),
    )


def test_offset_export_round_trip_and_mapping_direction(tmp_path: Path) -> None:
    offset = extract_alignment_offset(
        _result(), user="user_0", action="0", dataset_id=0
    )
    path = build_alignment_offset_path(
        tmp_path / "offsets", user="user_0", action="0", dataset_id=0
    )

    written = write_alignment_offset_txt(offset, output_path=path)
    loaded = read_alignment_offset_txt(
        written,
        expected_user="user_0",
        expected_action="0",
        expected_dataset_id=0,
    )

    assert path.name == "0_ring_board_offset.txt"
    assert path.parent == tmp_path / "offsets" / "user_0" / "action_0"
    assert loaded == offset
    assert loaded.offset_ms == pytest.approx(-238.45175)
    assert "ring_timestamp_us = board_timestamp_us + offset_us" in path.read_text()
    np.testing.assert_allclose(
        apply_board_to_ring_offset(np.array([1_000_000.0]), offset_us=250_000.0),
        [1_250_000.0],
    )


def test_offset_export_rejects_failure_nonfinite_identity_mismatch_and_overwrite(
    tmp_path: Path,
) -> None:
    with pytest.raises(AlignmentOffsetExportError, match="did not succeed"):
        extract_alignment_offset(_result(success=False), user="user_0", action="0", dataset_id=0)
    with pytest.raises(AlignmentOffsetExportError, match="finite"):
        extract_alignment_offset(_result(offset_us=float("nan")), user="user_0", action="0", dataset_id=0)
    offset = extract_alignment_offset(_result(), user="user_0", action="0", dataset_id=0)
    path = tmp_path / "offset.txt"
    write_alignment_offset_txt(offset, output_path=path)
    with pytest.raises(AlignmentOffsetExportError, match="already exists"):
        write_alignment_offset_txt(offset, output_path=path)
    write_alignment_offset_txt(offset, output_path=path, overwrite=True)
    with pytest.raises(AlignmentOffsetExportError, match="identity"):
        read_alignment_offset_txt(
            path,
            expected_user="user_9",
            expected_action="0",
            expected_dataset_id=0,
        )
