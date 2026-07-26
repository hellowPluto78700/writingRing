from pathlib import Path

import pytest

from writingring.discovery import Recording
from writingring.selection import (
    InvalidRecordingSelectorError,
    MultipleRecordingsMatchedError,
    RecordingNotFoundError,
    select_recording,
)


def _recording(
    *,
    user: str = "writer_a",
    action: str = "letters",
    dataset_id: int = 4,
) -> Recording:
    return Recording(
        user=user,
        action=action,
        dataset_id=dataset_id,
        ring_0_path=Path(f"{dataset_id}_ring_0.bin"),
        ring_1_path=None,
        timestamp_path=None,
        board_chunk_paths=(),
        missing_chunk_indices=(),
        warnings=(),
    )


def test_selects_exact_recording_identity() -> None:
    expected = _recording()
    recordings = [
        _recording(user="writer_b"),
        expected,
        _recording(dataset_id=5),
    ]

    selected = select_recording(
        recordings,
        user="writer_a",
        action="letters",
        dataset_id=4,
    )

    assert selected is expected


def test_missing_recording_raises_clear_error() -> None:
    with pytest.raises(RecordingNotFoundError, match="no recording matches"):
        select_recording(
            [_recording()],
            user="writer_a",
            action="letters",
            dataset_id=99,
        )


def test_duplicate_identity_raises_clear_error() -> None:
    recording = _recording()

    with pytest.raises(
        MultipleRecordingsMatchedError,
        match="multiple recordings unexpectedly match",
    ):
        select_recording(
            [recording, recording],
            user="writer_a",
            action="letters",
            dataset_id=4,
        )


@pytest.mark.parametrize(
    ("user", "action", "dataset_id"),
    [
        ("", "letters", 4),
        ("writer_a", "", 4),
        ("writer_a", "letters", -1),
        ("writer_a", "letters", True),
    ],
)
def test_invalid_selectors_raise_clear_error(
    user: str,
    action: str,
    dataset_id: int,
) -> None:
    with pytest.raises(InvalidRecordingSelectorError):
        select_recording(
            [_recording()],
            user=user,
            action=action,
            dataset_id=dataset_id,
        )
