from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd
import pytest
import matplotlib.pyplot as plt

from writingring.alignment_verification import (
    AlignmentLabel,
    AlignmentVerificationConfig,
    AlignmentVerificationError,
    build_alignment_verification_path,
    create_alignment_verification_figure,
    load_alignment_labels,
    resolve_alignment_label_path,
)
from writingring.event_alignment import SequenceAlignmentResult


def _result(offset_us: float = 500_000.0) -> SequenceAlignmentResult:
    return SequenceAlignmentResult(
        success=True,
        best_offset_us=offset_us,
        second_best_offset_us=None,
        candidates=pd.DataFrame(),
        event_matches=pd.DataFrame({"event_index": [0, 1], "matched": [True, False]}),
        touch_pair_matches=pd.DataFrame(),
        report={},
        warnings=(),
    )


def test_labels_resolution_parser_and_malformed_line(tmp_path: Path) -> None:
    fallback = tmp_path / "0_timestamp.txt"
    fallback.write_text("100 A\n# comment\n200 Long Label\n", encoding="utf-8")
    assert resolve_alignment_label_path(tmp_path, dataset_id=0) == fallback
    labels = load_alignment_labels(fallback)
    assert labels == (
        AlignmentLabel(100.0, "A", 1),
        AlignmentLabel(200.0, "Long Label", 3),
    )
    malformed = tmp_path / "bad.txt"
    malformed.write_text("100\n", encoding="utf-8")
    with pytest.raises(AlignmentVerificationError, match="line 1"):
        load_alignment_labels(malformed)


def test_verification_writes_six_panels_and_applies_offset_once(tmp_path: Path) -> None:
    timestamps = 9_000_000.0 + np.arange(52) * 1_000_000.0
    ring = pd.DataFrame({"value": np.arange(len(timestamps))})
    events = pd.DataFrame(
        {
            "event_index": [0, 1, 2],
            "event_type": ["press", "lift", "press"],
            "frame_timestamp_raw": [10_000_000.0, 20_000_000.0, 80_000_000.0],
        }
    )
    label_path = tmp_path / "labels.txt"
    label_path.write_text("10000000 A\n20000000 B\n", encoding="utf-8")
    output = build_alignment_verification_path(
        tmp_path / "verification", user="user_0", action="0", dataset_id=0
    )

    result = create_alignment_verification_figure(
        ring_dataframe=ring,
        ring_timestamps_us=timestamps,
        transient_score=np.linspace(0.0, 5.0, len(timestamps)),
        board_events=events,
        alignment_result=_result(),
        labels=load_alignment_labels(label_path),
        label_source_path=label_path,
        output_path=output,
        config=AlignmentVerificationConfig(label_time_domain="board"),
        user="user_0",
        action="0",
        dataset_id=0,
    )

    assert output.is_file() and output.stat().st_size > 0
    assert result.displayed_start_s == 0.0
    assert result.displayed_stop_s == 60.0
    assert result.press_count_displayed == 1
    assert result.lift_count_displayed == 1
    assert result.press_count_outside == 1
    assert result.label_count_displayed == 2
    assert result.label_count_outside == 0
    with pytest.raises(AlignmentVerificationError, match="already exists"):
        create_alignment_verification_figure(
            ring_dataframe=ring,
            ring_timestamps_us=timestamps,
            transient_score=np.linspace(0.0, 5.0, len(timestamps)),
            board_events=events,
            alignment_result=_result(),
            labels=load_alignment_labels(label_path),
            label_source_path=label_path,
            output_path=output,
        )


def test_verification_uses_six_shared_y_ten_second_panels(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    timestamps = 1_000_000.0 + np.arange(31) * 1_000_000.0
    label_path = tmp_path / "labels.txt"
    label_path.write_text("11000000 Boundary\n", encoding="utf-8")
    closed: list[object] = []
    original_close = plt.close
    monkeypatch.setattr(plt, "close", lambda figure: closed.append(figure))
    try:
        create_alignment_verification_figure(
            ring_dataframe=pd.DataFrame({"value": np.arange(len(timestamps))}),
            ring_timestamps_us=timestamps,
            transient_score=np.arange(len(timestamps), dtype=float),
            board_events=pd.DataFrame(
                {
                    "event_type": ["press"],
                    "frame_timestamp_raw": [10_000_000.0],
                }
            ),
            alignment_result=_result(),
            labels=load_alignment_labels(label_path),
            label_source_path=label_path,
            output_path=tmp_path / "six-panels.png",
        )
        figure = closed[-1]
        assert len(figure.axes) == 6
        assert figure.axes[0].get_position().y0 > figure.axes[-1].get_position().y0
        assert [axis.get_xlim() for axis in figure.axes] == [
            (0.0, 10.0),
            (10.0, 20.0),
            (20.0, 30.0),
            (30.0, 40.0),
            (40.0, 50.0),
            (50.0, 60.0),
        ]
        assert len({axis.get_ylim() for axis in figure.axes}) == 1
        assert sum(text.get_text() == "Boundary" for axis in figure.axes for text in axis.texts) == 1
    finally:
        for figure in closed:
            original_close(figure)
