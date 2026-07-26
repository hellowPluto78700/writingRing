#!/usr/bin/env python3
"""List WritingRing recordings discovered below a data root."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PROJECT_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from writingring.discovery import DiscoveryError, Recording, discover_recordings


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root",
        type=Path,
        required=True,
        help="directory containing user/action recording directories",
    )
    return parser


def format_recordings(recordings: Sequence[Recording]) -> str:
    """Format discovered recordings as a compact text table."""

    headers = (
        "user",
        "action",
        "dataset",
        "primary ring",
        "timestamp",
        "board chunks",
        "chunk range",
        "missing indices",
        "warnings",
    )
    rows: list[tuple[str, ...]] = []
    for recording in recordings:
        chunk_range = recording.chunk_index_range
        rows.append(
            (
                recording.user,
                recording.action,
                str(recording.dataset_id),
                recording.ring_0_path.name,
                (
                    f"present ({recording.timestamp_path.name})"
                    if recording.timestamp_path is not None
                    else "missing"
                ),
                str(len(recording.board_chunk_paths)),
                (
                    f"{chunk_range[0]}-{chunk_range[1]}"
                    if chunk_range is not None
                    else "-"
                ),
                (
                    ", ".join(
                        str(index)
                        for index in recording.missing_chunk_indices
                    )
                    or "-"
                ),
                "; ".join(recording.warnings) or "-",
            )
        )

    if not rows:
        return "No recordings found."

    widths = [
        max(len(headers[index]), *(len(row[index]) for row in rows))
        for index in range(len(headers))
    ]
    output = [
        "  ".join(
            value.ljust(widths[index])
            for index, value in enumerate(headers)
        ),
        "  ".join("-" * width for width in widths),
    ]
    output.extend(
        "  ".join(
            value.ljust(widths[index])
            for index, value in enumerate(row)
        )
        for row in rows
    )
    return "\n".join(output)


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        recordings = discover_recordings(args.data_root)
    except DiscoveryError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2

    print(format_recordings(recordings))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

