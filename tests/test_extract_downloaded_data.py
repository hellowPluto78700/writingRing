from __future__ import annotations

from pathlib import Path
import zipfile

import pytest

from scripts import extract_downloaded_data


def _archive(path: Path, members: dict[str, bytes]) -> Path:
    with zipfile.ZipFile(path, "w") as archive:
        for name, content in members.items():
            archive.writestr(name, content)
    return path


def test_extracts_data_tree_and_ignores_macos_metadata(tmp_path: Path) -> None:
    archive = _archive(
        tmp_path / "data.zip",
        {
            "data/user_2/0/0_ring_0.bin": b"ring",
            "data/user_2/0/0_timestamp.txt": b"marker",
            "__MACOSX/data/user_2/0/._0_ring_0.bin": b"metadata",
            "notes.txt": b"outside data tree",
        },
    )
    output = tmp_path / "data"

    result = extract_downloaded_data.extract_archives(
        (archive,),
        output_root=output,
    )

    assert result.output_root == output
    assert result.file_count == 2
    assert result.byte_count == len(b"ring") + len(b"marker")
    assert (output / "user_2" / "0" / "0_ring_0.bin").read_bytes() == b"ring"
    assert (output / "user_2" / "0" / "0_ring_0.bin").stat().st_size == 4
    assert not (output / "__MACOSX").exists()
    assert not (tmp_path / "notes.txt").exists()


def test_existing_output_root_is_not_overwritten(tmp_path: Path) -> None:
    archive = _archive(
        tmp_path / "data.zip",
        {"data/user_2/0/0_ring_0.bin": b"new"},
    )
    output = tmp_path / "data"
    output.mkdir()
    sentinel = output / "keep.txt"
    sentinel.write_text("old")

    with pytest.raises(extract_downloaded_data.ExtractionError, match="exists"):
        extract_downloaded_data.extract_archives((archive,), output_root=output)

    assert sentinel.read_text() == "old"


def test_unsafe_member_is_rejected_without_publishing_data(tmp_path: Path) -> None:
    archive = _archive(
        tmp_path / "data.zip",
        {
            "data/user_2/0/0_ring_0.bin": b"ring",
            "../outside.txt": b"unsafe",
        },
    )
    output = tmp_path / "data"

    with pytest.raises(extract_downloaded_data.ExtractionError, match="unsafe"):
        extract_downloaded_data.extract_archives((archive,), output_root=output)

    assert not output.exists()
    assert not (tmp_path / "outside.txt").exists()


def test_cli_discovers_download_archives(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    downloads = tmp_path / "downloads" / "WritingRing"
    downloads.mkdir(parents=True)
    _archive(
        downloads / "data.zip",
        {"data/user_3/1/0_timestamp.txt": b"marker"},
    )
    output = tmp_path / "data"

    status = extract_downloaded_data.main(
        [
            "--downloads-root",
            str(tmp_path / "downloads"),
            "--output-root",
            str(output),
        ]
    )

    assert status == 0
    assert (output / "user_3" / "1" / "0_timestamp.txt").is_file()
    assert f"Data root: {output}" in capsys.readouterr().out


def test_output_must_be_named_data_and_not_protected_sample(tmp_path: Path) -> None:
    archive = _archive(
        tmp_path / "data.zip",
        {"data/user_2/0/0_ring_0.bin": b"ring"},
    )

    with pytest.raises(extract_downloaded_data.ExtractionError, match="named"):
        extract_downloaded_data.extract_archives(
            (archive,),
            output_root=tmp_path / "other",
        )
