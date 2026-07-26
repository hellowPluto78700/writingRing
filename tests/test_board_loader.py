from pathlib import Path

import compress_pickle
import numpy as np
import pytest

from core.sensel_lib.frame_data import ContactData, FrameData
from writingring.board_loader import (
    BoardDeserializationError,
    BoardPathError,
    EmptyBoardChunkFileError,
    MissingBoardAttributeError,
    MissingBoardChunksError,
    UnexpectedBoardContainerError,
    UnexpectedBoardObjectError,
    load_board,
)
from writingring.discovery import Recording, discover_recordings


def _frame(timestamp: int, contacts: list[ContactData] | None = None) -> FrameData:
    frame = FrameData(np.zeros((2, 3), dtype=np.float64), timestamp)
    for contact in contacts or []:
        frame.append_contact(contact)
    return frame


def _contact(
    *,
    contact_id: int = 3,
    x: float = 0.25,
    y: float = 0.75,
    force: float = 12.5,
    frame_id: int = 9,
) -> ContactData:
    return ContactData(
        id=contact_id,
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
        frame_id=frame_id,
    )


def _write_chunk(path: Path, frames: object) -> Path:
    with path.open("wb") as output:
        compress_pickle.dump(frames, output)
    return path


def _recording(tmp_path: Path, board_paths: tuple[Path, ...]) -> Recording:
    ring_path = tmp_path / "0_ring_0.bin"
    ring_path.touch(exist_ok=True)
    return Recording(
        user="user_0",
        action="0",
        dataset_id=0,
        ring_0_path=ring_path,
        ring_1_path=None,
        timestamp_path=None,
        board_chunk_paths=board_paths,
        missing_chunk_indices=(),
        warnings=(),
    )


def test_loads_valid_frame_lists_in_supplied_numeric_order(
    tmp_path: Path,
) -> None:
    chunk_0 = _write_chunk(tmp_path / "0_board_0.gz", [_frame(100)])
    chunk_1 = _write_chunk(tmp_path / "0_board_1.gz", [_frame(200)])

    result = load_board((chunk_0, chunk_1))

    assert result.validation.chunk_indices == (0, 1)
    assert list(result.frames["chunk_index"]) == [0, 1]
    assert list(result.frames["frame_timestamp_raw"]) == [100, 200]
    assert result.validation.frame_count == 2
    assert len(result.chunk_reports) == 2
    assert all(report.deserialized for report in result.chunk_reports)


def test_frames_without_contacts_are_preserved(tmp_path: Path) -> None:
    path = _write_chunk(
        tmp_path / "0_board_0.gz",
        [_frame(100), _frame(200, [_contact()])],
    )

    result = load_board(path)

    assert len(result.frames) == 2
    assert list(result.frames["contact_count"]) == [0, 1]
    assert len(result.contacts) == 1
    assert result.validation.frames_without_contacts == 1


def test_contact_extraction_preserves_y_and_adds_display_transform(
    tmp_path: Path,
) -> None:
    contact = _contact(contact_id=7, x=0.2, y=0.8, force=33.0)
    path = _write_chunk(
        tmp_path / "0_board_0.gz",
        [_frame(100, [contact])],
    )

    row = load_board(path).contacts.iloc[0]

    assert row["contact_id"] == 7
    assert row["x"] == pytest.approx(0.2)
    assert row["y_raw"] == pytest.approx(0.8)
    assert row["y_display"] == pytest.approx(0.2)
    assert row["force"] == pytest.approx(33.0)


def test_serialized_empty_chunk_is_retained_and_warned(tmp_path: Path) -> None:
    empty = _write_chunk(tmp_path / "0_board_0.gz", [])
    nonempty = _write_chunk(tmp_path / "0_board_1.gz", [_frame(100)])

    result = load_board((empty, nonempty))

    assert result.validation.chunk_count == 2
    assert result.validation.empty_chunk_indices == (0,)
    assert result.validation.empty_leading_chunk_indices == (0,)
    assert len(result.chunk_reports) == 2
    assert result.chunk_reports[0].deserialized
    assert result.chunk_reports[0].empty
    assert result.chunk_reports[0].frame_count == 0
    assert result.validation.frame_count == 1


def test_missing_chunk_indices_are_reported(tmp_path: Path) -> None:
    chunk_0 = _write_chunk(tmp_path / "0_board_0.gz", [_frame(100)])
    chunk_2 = _write_chunk(tmp_path / "0_board_2.gz", [_frame(200)])

    result = load_board((chunk_0, chunk_2))

    assert result.validation.missing_chunk_indices == (1,)
    assert any("missing board chunk indices: 1" in warning for warning in result.warnings)


def test_duplicate_chunk_indices_are_retained_and_reported(
    tmp_path: Path,
) -> None:
    first = _write_chunk(tmp_path / "0_board_1.gz", [_frame(100)])
    second = _write_chunk(tmp_path / "0_board_01.gz", [_frame(200)])

    result = load_board((second, first))

    assert result.validation.chunk_indices == (1, 1)
    assert result.validation.duplicate_chunk_indices == (1,)
    assert result.validation.chunk_count == 2
    assert list(result.frames["frame_timestamp_raw"]) == [200, 100]


def test_zero_byte_chunk_raises_with_failed_report(tmp_path: Path) -> None:
    path = tmp_path / "0_board_0.gz"
    path.touch()

    with pytest.raises(EmptyBoardChunkFileError) as raised:
        load_board(path)

    assert not raised.value.chunk_report.deserialized
    assert raised.value.chunk_report.errors


def test_invalid_gzip_or_pickle_raises_with_failed_report(
    tmp_path: Path,
) -> None:
    path = tmp_path / "0_board_0.gz"
    path.write_bytes(b"not a gzip pickle")

    with pytest.raises(BoardDeserializationError) as raised:
        load_board(path)

    assert not raised.value.chunk_report.deserialized
    assert "failed to deserialize" in raised.value.chunk_report.errors[0]


def test_unexpected_top_level_type_raises(tmp_path: Path) -> None:
    path = _write_chunk(tmp_path / "0_board_0.gz", {"frames": []})

    with pytest.raises(UnexpectedBoardContainerError) as raised:
        load_board(path)

    assert raised.value.chunk_report.deserialized


def test_unexpected_frame_type_raises_without_dropping_chunk(
    tmp_path: Path,
) -> None:
    path = _write_chunk(tmp_path / "0_board_0.gz", [{"timestamp": 100}])

    with pytest.raises(UnexpectedBoardObjectError) as raised:
        load_board(path)

    assert raised.value.chunk_report.chunk_index == 0
    assert raised.value.chunk_report.deserialized


def test_missing_expected_frame_attribute_raises(tmp_path: Path) -> None:
    frame = _frame(100)
    del frame.timestamp
    path = _write_chunk(tmp_path / "0_board_0.gz", [frame])

    with pytest.raises(MissingBoardAttributeError, match="timestamp"):
        load_board(path)


def test_missing_expected_contact_attribute_raises(tmp_path: Path) -> None:
    contact = _contact()
    del contact.force
    path = _write_chunk(
        tmp_path / "0_board_0.gz",
        [_frame(100, [contact])],
    )

    with pytest.raises(MissingBoardAttributeError, match="force"):
        load_board(path)


def test_within_chunk_backward_timestamp_is_reported(tmp_path: Path) -> None:
    path = _write_chunk(
        tmp_path / "0_board_0.gz",
        [_frame(200), _frame(100)],
    )

    result = load_board(path)

    report = result.chunk_reports[0]
    assert not report.timestamps_nondecreasing
    assert report.backward_timestamp_steps == 1
    assert report.backward_timestamp_step_positions == (1,)
    assert result.validation.within_chunk_backward_steps == 1
    assert list(result.frames["frame_timestamp_raw"]) == [200, 100]


def test_cross_chunk_backward_jump_is_reported_without_reordering(
    tmp_path: Path,
) -> None:
    chunk_0 = _write_chunk(tmp_path / "0_board_0.gz", [_frame(200)])
    chunk_1 = _write_chunk(tmp_path / "0_board_1.gz", [_frame(100)])

    result = load_board((chunk_0, chunk_1))

    assert list(result.frames["frame_timestamp_raw"]) == [200, 100]
    assert len(result.validation.cross_chunk_backward_boundaries) == 1
    boundary = result.validation.cross_chunk_backward_boundaries[0]
    assert (boundary.previous_chunk_index, boundary.next_chunk_index) == (0, 1)
    assert boundary.timestamp_delta == -100.0
    assert result.validation.chunk_count == 2


def test_decreasing_numeric_input_order_is_rejected_not_sorted(
    tmp_path: Path,
) -> None:
    chunk_1 = _write_chunk(tmp_path / "0_board_1.gz", [_frame(100)])
    chunk_2 = _write_chunk(tmp_path / "0_board_2.gz", [_frame(200)])

    with pytest.raises(BoardPathError, match="numeric chunk-index order"):
        load_board((chunk_2, chunk_1))


def test_dataset_membership_isolation(tmp_path: Path) -> None:
    chunk_0 = _write_chunk(tmp_path / "0_board_0.gz", [_frame(100)])
    chunk_1 = _write_chunk(tmp_path / "1_board_1.gz", [_frame(200)])

    with pytest.raises(BoardPathError, match="more than one dataset ID"):
        load_board((chunk_0, chunk_1))


def test_recording_uses_only_its_board_paths(tmp_path: Path) -> None:
    action = tmp_path / "user_0" / "0"
    action.mkdir(parents=True)
    (action / "0_ring_0.bin").touch()
    (action / "1_ring_0.bin").touch()
    _write_chunk(action / "0_board_0.gz", [_frame(100)])
    _write_chunk(action / "1_board_0.gz", [_frame(900)])
    recordings = discover_recordings(tmp_path)

    result = load_board(recordings[0])

    assert result.validation.dataset_id == 0
    assert [path.name for path in result.chunk_paths] == ["0_board_0.gz"]
    assert list(result.frames["frame_timestamp_raw"]) == [100]


def test_no_board_paths_raises(tmp_path: Path) -> None:
    recording = _recording(tmp_path, ())

    with pytest.raises(MissingBoardChunksError):
        load_board(recording)


def test_missing_board_file_raises(tmp_path: Path) -> None:
    path = tmp_path / "0_board_0.gz"

    with pytest.raises(BoardPathError, match="does not exist"):
        load_board(path)


def test_nonfinite_coordinate_and_force_values_are_reported(
    tmp_path: Path,
) -> None:
    contact = _contact(x=np.nan, y=np.inf, force=-np.inf)
    path = _write_chunk(
        tmp_path / "0_board_0.gz",
        [_frame(100, [contact])],
    )

    report = load_board(path).validation

    assert report.x_range.nonfinite_count == 1
    assert report.y_raw_range.nonfinite_count == 1
    assert report.force_range.nonfinite_count == 1
    assert any("non-finite x value" in warning for warning in report.warnings)


def test_non_regular_path_raises(tmp_path: Path) -> None:
    path = tmp_path / "0_board_0.gz"
    path.mkdir()

    with pytest.raises(BoardPathError, match="not a regular file"):
        load_board(path)
