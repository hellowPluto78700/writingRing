#!/usr/bin/env python3
"""Extract downloaded WritingRing ``data.zip`` archives into one data root.

The extractor retains the archive's ``data/user_<id>/<action>/`` hierarchy,
ignores macOS metadata, rejects unsafe archive paths, and stages files before
publishing the requested output directory. It never writes under
``data_sample/``.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
import shutil
import sys
import tempfile
from typing import Iterable, Sequence
import zipfile


DATA_DIRECTORY_NAME = "data"
MACOS_METADATA_DIRECTORY = "__MACOSX"
COPY_BUFFER_SIZE = 1024 * 1024


class ExtractionError(Exception):
    """Base error for downloaded-archive extraction failures."""


@dataclass(frozen=True, slots=True)
class ArchiveMember:
    """One verified file to copy from an archive into the staged data root."""

    archive_path: Path
    member_name: str
    relative_destination: PurePosixPath
    size_bytes: int


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--archive",
        type=Path,
        action="append",
        help="archive to extract; repeatable, defaults to downloads/**/data.zip",
    )
    parser.add_argument(
        "--downloads-root",
        type=Path,
        default=Path("downloads"),
        help="root searched for data.zip when --archive is omitted",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path(DATA_DIRECTORY_NAME),
        help="new output root that receives user_<id>/<action>/ files",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Extract archives and return a conventional CLI status code."""

    args = build_parser().parse_args(argv)
    try:
        archives = _resolve_archives(
            explicit_archives=args.archive,
            downloads_root=args.downloads_root,
        )
        result = extract_archives(archives, output_root=args.output_root)
    except (ExtractionError, OSError, zipfile.BadZipFile) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2

    print(f"Archives: {len(archives)}")
    print(f"Files extracted: {result.file_count}")
    print(f"Bytes extracted: {result.byte_count}")
    print(f"Data root: {result.output_root}")
    return 0


@dataclass(frozen=True, slots=True)
class ExtractionResult:
    """Summary of a successfully published extraction."""

    output_root: Path
    file_count: int
    byte_count: int


def extract_archives(
    archive_paths: Iterable[Path],
    *,
    output_root: Path,
) -> ExtractionResult:
    """Stage verified archive content and atomically publish ``output_root``.

    Each archive must include paths under a top-level ``data/`` directory.
    Destination collisions and an existing output root are rejected rather than
    overwritten.
    """

    archives = tuple(Path(path) for path in archive_paths)
    if not archives:
        raise ExtractionError("no data.zip archive was supplied or discovered")
    if output_root.exists():
        raise ExtractionError(
            f"output root already exists and will not be overwritten: {output_root}"
        )
    if output_root.name != DATA_DIRECTORY_NAME:
        raise ExtractionError(
            f"output root must be named {DATA_DIRECTORY_NAME!r} to preserve "
            f"the documented hierarchy, got {output_root.name!r}"
        )
    if _is_within_data_sample(output_root):
        raise ExtractionError("output root must not be under protected data_sample/")

    members = _collect_members(archives)
    if not members:
        raise ExtractionError("archives contain no files under data/")

    output_parent = output_root.parent
    output_parent.mkdir(parents=True, exist_ok=True)
    staging_parent = Path(
        tempfile.mkdtemp(
            prefix=f".{output_root.name}-extract-",
            dir=output_parent,
        )
    )
    staged_root = staging_parent / DATA_DIRECTORY_NAME
    try:
        _extract_members(members, staged_root=staged_root)
        staged_root.replace(output_root)
    except BaseException:
        shutil.rmtree(staging_parent, ignore_errors=True)
        raise
    finally:
        if staging_parent.exists():
            staging_parent.rmdir()

    return ExtractionResult(
        output_root=output_root,
        file_count=len(members),
        byte_count=sum(member.size_bytes for member in members),
    )


def _resolve_archives(
    *,
    explicit_archives: Sequence[Path] | None,
    downloads_root: Path,
) -> tuple[Path, ...]:
    if explicit_archives:
        archives = tuple(Path(path) for path in explicit_archives)
    else:
        if not downloads_root.is_dir():
            raise ExtractionError(
                f"downloads root is not a directory: {downloads_root}"
            )
        archives = tuple(sorted(downloads_root.rglob("data.zip")))
    if not archives:
        raise ExtractionError("no data.zip archive was supplied or discovered")
    for archive in archives:
        if not archive.is_file():
            raise ExtractionError(f"archive is not a regular file: {archive}")
    return archives


def _collect_members(archives: Sequence[Path]) -> tuple[ArchiveMember, ...]:
    members: list[ArchiveMember] = []
    destinations: set[PurePosixPath] = set()
    for archive_path in archives:
        with zipfile.ZipFile(archive_path) as archive:
            for info in archive.infolist():
                relative_destination = _member_destination(info)
                if relative_destination is None:
                    continue
                if relative_destination in destinations:
                    raise ExtractionError(
                        "multiple archive members target the same destination: "
                        f"{relative_destination}"
                    )
                destinations.add(relative_destination)
                members.append(
                    ArchiveMember(
                        archive_path=archive_path,
                        member_name=info.filename,
                        relative_destination=relative_destination,
                        size_bytes=info.file_size,
                    )
                )
    return tuple(members)


def _member_destination(info: zipfile.ZipInfo) -> PurePosixPath | None:
    path = PurePosixPath(info.filename)
    if path.is_absolute() or ".." in path.parts:
        raise ExtractionError(f"unsafe archive member path: {info.filename!r}")
    if not path.parts or path.parts[0] == MACOS_METADATA_DIRECTORY:
        return None
    if path.parts[0] != DATA_DIRECTORY_NAME:
        return None
    if info.is_dir():
        return None
    if _is_zip_symlink(info):
        raise ExtractionError(f"symbolic-link archive member is not allowed: {info.filename!r}")
    relative_destination = PurePosixPath(*path.parts[1:])
    if not relative_destination.parts:
        return None
    return relative_destination


def _is_zip_symlink(info: zipfile.ZipInfo) -> bool:
    return (info.external_attr >> 16) & 0o170000 == 0o120000


def _extract_members(
    members: Sequence[ArchiveMember],
    *,
    staged_root: Path,
) -> None:
    archives: dict[Path, zipfile.ZipFile] = {}
    try:
        for member in members:
            archive = archives.get(member.archive_path)
            if archive is None:
                archive = zipfile.ZipFile(member.archive_path)
                archives[member.archive_path] = archive
            target = staged_root.joinpath(*member.relative_destination.parts)
            target.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(member.member_name) as source, target.open("xb") as output:
                shutil.copyfileobj(source, output, length=COPY_BUFFER_SIZE)
            extracted_size = target.stat().st_size
            if extracted_size != member.size_bytes:
                raise ExtractionError(
                    "extracted size does not match archive member for "
                    f"{member.member_name!r}: expected {member.size_bytes}, "
                    f"got {extracted_size}"
                )
    finally:
        for archive in archives.values():
            archive.close()


def _is_within_data_sample(path: Path) -> bool:
    resolved_path = path.resolve(strict=False)
    protected_root = (Path.cwd() / "data_sample").resolve(strict=False)
    try:
        resolved_path.relative_to(protected_root)
    except ValueError:
        return False
    return True


if __name__ == "__main__":
    raise SystemExit(main())
