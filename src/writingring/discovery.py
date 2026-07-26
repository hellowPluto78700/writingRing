"""Discover WritingRing recordings from their filenames.

This module intentionally inspects paths and filenames only. It does not read
ring binary contents or deserialize board chunks.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
import re


class FileKind(str, Enum):
    """Recognized recording-file roles."""

    RING_0 = "ring_0"
    RING_1 = "ring_1"
    TIMESTAMP = "timestamp"
    BOARD_CHUNK = "board_chunk"


@dataclass(frozen=True, slots=True)
class ParsedFilename:
    """Structured fields parsed from one recognized filename."""

    dataset_id: int
    kind: FileKind
    chunk_index: int | None = None


@dataclass(frozen=True, slots=True)
class Recording:
    """Filesystem components belonging to one nominal recording."""

    user: str
    action: str
    dataset_id: int
    ring_0_path: Path
    ring_1_path: Path | None
    timestamp_path: Path | None
    board_chunk_paths: tuple[Path, ...]
    missing_chunk_indices: tuple[int, ...]
    warnings: tuple[str, ...]

    @property
    def board_chunk_indices(self) -> tuple[int, ...]:
        """Return chunk indices in the same order as ``board_chunk_paths``."""

        indices: list[int] = []
        for path in self.board_chunk_paths:
            parsed = parse_recording_filename(path.name)
            if parsed is None or parsed.kind is not FileKind.BOARD_CHUNK:
                raise ValueError(f"not a board-chunk path: {path}")
            if parsed.chunk_index is None:
                raise ValueError(f"board-chunk index is missing: {path}")
            indices.append(parsed.chunk_index)
        return tuple(indices)

    @property
    def chunk_index_range(self) -> tuple[int, int] | None:
        """Return the first and last discovered chunk index, if any."""

        indices = self.board_chunk_indices
        if not indices:
            return None
        return indices[0], indices[-1]


class DiscoveryError(RuntimeError):
    """Raised when discovery cannot safely inspect the requested tree."""


_RING_PATTERN = re.compile(
    r"^(?P<dataset_id>\d+)_ring_(?P<ring_index>[01])\.bin$"
)
_TIMESTAMP_PATTERN = re.compile(r"^(?P<dataset_id>\d+)_timestamp\.txt$")
_BOARD_PATTERN = re.compile(
    r"^(?P<dataset_id>\d+)_board_(?P<chunk_index>\d+)\.gz$"
)


def parse_recording_filename(filename: str | Path) -> ParsedFilename | None:
    """Parse one recording filename without accessing the filesystem.

    Unrecognized and malformed names return ``None`` so unrelated directory
    contents do not break discovery.
    """

    name = Path(filename).name

    ring_match = _RING_PATTERN.fullmatch(name)
    if ring_match is not None:
        ring_index = int(ring_match.group("ring_index"))
        kind = FileKind.RING_0 if ring_index == 0 else FileKind.RING_1
        return ParsedFilename(
            dataset_id=int(ring_match.group("dataset_id")),
            kind=kind,
        )

    timestamp_match = _TIMESTAMP_PATTERN.fullmatch(name)
    if timestamp_match is not None:
        return ParsedFilename(
            dataset_id=int(timestamp_match.group("dataset_id")),
            kind=FileKind.TIMESTAMP,
        )

    board_match = _BOARD_PATTERN.fullmatch(name)
    if board_match is not None:
        return ParsedFilename(
            dataset_id=int(board_match.group("dataset_id")),
            kind=FileKind.BOARD_CHUNK,
            chunk_index=int(board_match.group("chunk_index")),
        )

    return None


def discover_recordings(data_root: str | Path) -> list[Recording]:
    """Discover recordings below ``data_root/user/action``.

    A primary ``*_ring_0.bin`` file anchors every returned recording.
    ``ring_1`` is retained only as optional inspection metadata. No payload
    files are opened.
    """

    root = Path(data_root)
    if not root.exists():
        raise DiscoveryError(f"data root does not exist: {root}")
    if not root.is_dir():
        raise DiscoveryError(f"data root is not a directory: {root}")

    recordings: list[Recording] = []
    user_directories = sorted(
        (path for path in root.iterdir() if path.is_dir()),
        key=lambda path: path.name,
    )
    for user_directory in user_directories:
        action_directories = sorted(
            (path for path in user_directory.iterdir() if path.is_dir()),
            key=lambda path: path.name,
        )
        for action_directory in action_directories:
            recordings.extend(
                _discover_action_directory(
                    action_directory,
                    user=user_directory.name,
                    action=action_directory.name,
                )
            )

    return recordings


def _discover_action_directory(
    action_directory: Path,
    *,
    user: str,
    action: str,
) -> list[Recording]:
    files_by_dataset: dict[int, list[tuple[Path, ParsedFilename]]] = {}
    for path in action_directory.iterdir():
        if not path.is_file():
            continue
        parsed = parse_recording_filename(path.name)
        if parsed is None:
            continue
        files_by_dataset.setdefault(parsed.dataset_id, []).append((path, parsed))

    recordings: list[Recording] = []
    for dataset_id in sorted(files_by_dataset):
        entries = files_by_dataset[dataset_id]
        ring_0_paths = _paths_of_kind(entries, FileKind.RING_0)
        if not ring_0_paths:
            continue
        if len(ring_0_paths) > 1:
            names = ", ".join(path.name for path in ring_0_paths)
            raise DiscoveryError(
                f"multiple primary ring_0 files for "
                f"{user}/{action}/dataset {dataset_id}: {names}"
            )

        warnings: list[str] = []
        ring_1_path = _choose_optional_path(
            _paths_of_kind(entries, FileKind.RING_1),
            role="ring_1",
            warnings=warnings,
        )
        timestamp_path = _choose_optional_path(
            _paths_of_kind(entries, FileKind.TIMESTAMP),
            role="timestamp",
            warnings=warnings,
        )
        if timestamp_path is None:
            warnings.append("timestamp file is missing")

        board_entries = [
            (path, parsed.chunk_index)
            for path, parsed in entries
            if parsed.kind is FileKind.BOARD_CHUNK
        ]
        board_entries.sort(
            key=lambda item: (
                -1 if item[1] is None else item[1],
                item[0].name,
            )
        )
        board_paths = tuple(path for path, _ in board_entries)
        board_indices = tuple(
            index for _, index in board_entries if index is not None
        )

        if not board_indices:
            missing_indices: tuple[int, ...] = ()
            warnings.append("no board chunks found")
        else:
            discovered_indices = set(board_indices)
            missing_indices = tuple(
                index
                for index in range(0, max(discovered_indices) + 1)
                if index not in discovered_indices
            )
            if missing_indices:
                formatted = ", ".join(str(index) for index in missing_indices)
                warnings.append(f"missing board chunk indices: {formatted}")

            duplicate_indices = sorted(
                {
                    index
                    for index in discovered_indices
                    if board_indices.count(index) > 1
                }
            )
            if duplicate_indices:
                formatted = ", ".join(str(index) for index in duplicate_indices)
                warnings.append(f"duplicate board chunk indices: {formatted}")

        recordings.append(
            Recording(
                user=user,
                action=action,
                dataset_id=dataset_id,
                ring_0_path=ring_0_paths[0],
                ring_1_path=ring_1_path,
                timestamp_path=timestamp_path,
                board_chunk_paths=board_paths,
                missing_chunk_indices=missing_indices,
                warnings=tuple(warnings),
            )
        )

    return recordings


def _paths_of_kind(
    entries: list[tuple[Path, ParsedFilename]],
    kind: FileKind,
) -> list[Path]:
    return sorted(
        (path for path, parsed in entries if parsed.kind is kind),
        key=lambda path: path.name,
    )


def _choose_optional_path(
    paths: list[Path],
    *,
    role: str,
    warnings: list[str],
) -> Path | None:
    if not paths:
        return None
    if len(paths) > 1:
        names = ", ".join(path.name for path in paths)
        warnings.append(f"multiple {role} files found; using {paths[0].name}: {names}")
    return paths[0]

