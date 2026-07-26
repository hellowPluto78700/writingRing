from pathlib import Path

import pytest

from writingring.discovery import (
    DiscoveryError,
    FileKind,
    discover_recordings,
    parse_recording_filename,
)


def _action_directory(root: Path, user: str = "user_0", action: str = "0") -> Path:
    directory = root / user / action
    directory.mkdir(parents=True)
    return directory


def _touch(directory: Path, *names: str) -> None:
    for name in names:
        (directory / name).touch()


@pytest.mark.parametrize(
    ("name", "dataset_id", "kind", "chunk_index"),
    [
        ("12_ring_0.bin", 12, FileKind.RING_0, None),
        ("12_ring_1.bin", 12, FileKind.RING_1, None),
        ("12_timestamp.txt", 12, FileKind.TIMESTAMP, None),
        ("12_board_10.gz", 12, FileKind.BOARD_CHUNK, 10),
    ],
)
def test_parse_supported_filenames(
    name: str,
    dataset_id: int,
    kind: FileKind,
    chunk_index: int | None,
) -> None:
    parsed = parse_recording_filename(name)

    assert parsed is not None
    assert parsed.dataset_id == dataset_id
    assert parsed.kind is kind
    assert parsed.chunk_index == chunk_index


@pytest.mark.parametrize(
    "name",
    [
        "notes.txt",
        "0_ring.bin",
        "0_ring_2.bin",
        "0_ring_0.csv",
        "0_timestamp.csv",
        "0_board_x.gz",
        "0_board_1.gz.backup",
        "prefix_0_board_1.gz",
    ],
)
def test_malformed_unrelated_filenames_are_not_parsed(name: str) -> None:
    assert parse_recording_filename(name) is None


def test_ring_0_anchors_recording_and_ring_1_is_metadata(tmp_path: Path) -> None:
    action = _action_directory(tmp_path)
    _touch(
        action,
        "0_ring_0.bin",
        "0_ring_1.bin",
        "0_timestamp.txt",
        "0_board_0.gz",
    )

    recordings = discover_recordings(tmp_path)

    assert len(recordings) == 1
    recording = recordings[0]
    assert recording.ring_0_path.name == "0_ring_0.bin"
    assert recording.ring_1_path is not None
    assert recording.ring_1_path.name == "0_ring_1.bin"


def test_ring_1_without_ring_0_does_not_create_recording(tmp_path: Path) -> None:
    action = _action_directory(tmp_path)
    _touch(
        action,
        "7_ring_1.bin",
        "7_timestamp.txt",
        "7_board_0.gz",
    )

    assert discover_recordings(tmp_path) == []


def test_board_chunks_match_dataset_and_sort_numerically(tmp_path: Path) -> None:
    action = _action_directory(tmp_path)
    _touch(
        action,
        "0_ring_0.bin",
        "0_timestamp.txt",
        "0_board_10.gz",
        "0_board_2.gz",
        "0_board_1.gz",
        "0_board_0.gz",
        "1_ring_0.bin",
        "1_timestamp.txt",
        "1_board_3.gz",
    )

    recordings = discover_recordings(tmp_path)

    assert [recording.dataset_id for recording in recordings] == [0, 1]
    assert [path.name for path in recordings[0].board_chunk_paths] == [
        "0_board_0.gz",
        "0_board_1.gz",
        "0_board_2.gz",
        "0_board_10.gz",
    ]
    assert recordings[0].board_chunk_indices == (0, 1, 2, 10)
    assert [path.name for path in recordings[1].board_chunk_paths] == [
        "1_board_3.gz"
    ]


def test_missing_timestamp_is_optional_and_warned(tmp_path: Path) -> None:
    action = _action_directory(tmp_path)
    _touch(action, "0_ring_0.bin", "0_board_0.gz")

    recording = discover_recordings(tmp_path)[0]

    assert recording.timestamp_path is None
    assert "timestamp file is missing" in recording.warnings


def test_missing_board_chunks_are_warned(tmp_path: Path) -> None:
    action = _action_directory(tmp_path)
    _touch(action, "0_ring_0.bin", "0_timestamp.txt")

    recording = discover_recordings(tmp_path)[0]

    assert recording.board_chunk_paths == ()
    assert recording.missing_chunk_indices == ()
    assert "no board chunks found" in recording.warnings


def test_missing_board_indices_include_zero_and_interior_gaps(
    tmp_path: Path,
) -> None:
    action = _action_directory(tmp_path)
    _touch(
        action,
        "0_ring_0.bin",
        "0_timestamp.txt",
        "0_board_1.gz",
        "0_board_3.gz",
    )

    recording = discover_recordings(tmp_path)[0]

    assert recording.missing_chunk_indices == (0, 2)
    assert "missing board chunk indices: 0, 2" in recording.warnings


def test_malformed_unrelated_files_do_not_affect_discovery(tmp_path: Path) -> None:
    action = _action_directory(tmp_path)
    _touch(
        action,
        "0_ring_0.bin",
        "0_timestamp.txt",
        "0_board_0.gz",
        "0_board_old.gz",
        "0_ring_2.bin",
        "README.txt",
    )

    recording = discover_recordings(tmp_path)[0]

    assert [path.name for path in recording.board_chunk_paths] == [
        "0_board_0.gz"
    ]
    assert recording.warnings == ()


def test_missing_data_root_raises_clear_error(tmp_path: Path) -> None:
    missing_root = tmp_path / "missing"

    with pytest.raises(DiscoveryError, match="data root does not exist"):
        discover_recordings(missing_root)

