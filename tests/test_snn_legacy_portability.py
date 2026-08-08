from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from snn.har_snn import (
    build_checkpoint_path,
    build_data_path,
    build_saved_model_path,
)
from snn.utils_parser import getArgsParser


def test_legacy_roots_are_required_and_path_typed(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    data_root = tmp_path / "dataset-root"
    model_root = tmp_path / "model-root"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "har_snn.py",
            "--data-root",
            str(data_root),
            "--model-root",
            str(model_root),
        ],
    )

    args = getArgsParser()

    assert args.data_root == data_root
    assert args.model_root == model_root
    assert isinstance(args.data_root, Path)
    assert isinstance(args.model_root, Path)


def test_legacy_roots_cannot_be_omitted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(sys, "argv", ["har_snn.py"])

    with pytest.raises(SystemExit):
        getArgsParser()


@pytest.mark.parametrize(
    ("flags", "expected_suffix"),
    [
        ({}, Path("eventsNIMU/windows")),
        ({"raw_data_wg": True}, Path("capture24_wg/windows")),
        ({"raw_data_gr": True}, Path("capture24_gr/windows")),
        ({"use_xylo": True}, Path("spikesXylo/windows")),
    ],
)
def test_data_root_composition_preserves_legacy_suffixes(
    tmp_path: Path,
    flags: dict[str, bool],
    expected_suffix: Path,
) -> None:
    data_root = tmp_path / "dataset-root"

    actual = build_data_path(data_root, "Capture24", **flags)

    assert actual == data_root / "capture24" / "data" / expected_suffix


def test_data_root_composition_preserves_historical_branch_priority(
    tmp_path: Path,
) -> None:
    actual = build_data_path(
        tmp_path,
        "Capture24",
        raw_data_wg=True,
        raw_data_gr=True,
        use_xylo=True,
    )

    assert actual == tmp_path / "capture24" / "data" / "capture24_wg" / "windows"


def test_checkpoint_root_composition_preserves_filenames(tmp_path: Path) -> None:
    model_root = tmp_path / "models"

    assert build_checkpoint_path(model_root, "pretrained.pth") == (
        model_root / "pretrained.pth"
    )
    assert build_saved_model_path(model_root, "2026-01-01T00:00:00+00:00", "run42") == (
        model_root / "2026-01-01T00:00:00+00:00_run42.pth"
    )


def test_package_and_direct_script_help_do_not_require_training_dependencies() -> None:
    repository_root = Path(__file__).resolve().parents[1]
    commands = [
        [sys.executable, "-m", "snn.har_snn", "--help"],
        [sys.executable, "snn/har_snn.py", "--help"],
    ]

    for command in commands:
        result = subprocess.run(
            command,
            cwd=repository_root,
            capture_output=True,
            text=True,
            check=False,
        )
        assert result.returncode == 0, result.stderr
        assert "--data-root" in result.stdout
        assert "--model-root" in result.stdout


def test_scoped_legacy_files_have_no_old_machine_literals() -> None:
    scoped_files = (
        Path("snn/har_snn.py"),
        Path("snn/utils_run.py"),
        Path("snn/utils_parser.py"),
    )
    old_machine_literals = ("/" + "home", "/" + "work")

    for path in scoped_files:
        source = path.read_text(encoding="utf-8")
        assert not any(literal in source for literal in old_machine_literals), path
