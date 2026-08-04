from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from writingring.alignment_io import AlignmentOffset, write_alignment_offset_txt
from scripts import segment_ring_imu


def _data_root(tmp_path: Path) -> Path:
    root = tmp_path / "data"
    action_dir = root / "writer_a" / "letters"
    action_dir.mkdir(parents=True)
    timestamps = 1_000_000 + np.arange(700, dtype=np.float64) * 1_000
    imu = np.column_stack(
        (
            np.zeros((len(timestamps), 2)),
            np.full(len(timestamps), 9.8),
            np.zeros((len(timestamps), 3)),
        )
    )
    np.column_stack((imu, timestamps)).astype(np.float64).tofile(
        action_dir / "0_ring_0.bin"
    )
    (action_dir / "0_ring_1.bin").write_bytes(b"must not be opened")
    (action_dir / "0_timestamp.txt").write_text(
        "1000000 a\n1350000 B\n", encoding="utf-8"
    )
    return root


def test_cli_exports_primary_ring_variable_segments_and_honors_overwrite(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    root = _data_root(tmp_path)
    output = tmp_path / "outputs"
    opened: list[Path] = []
    original = np.fromfile

    def tracked(path: object, *args: object, **kwargs: object) -> np.ndarray:
        if isinstance(path, (str, Path)):
            opened.append(Path(path))
        return original(path, *args, **kwargs)

    monkeypatch.setattr("writingring.ring_loader.np.fromfile", tracked)
    args = [
        "--data-root", str(root), "--user", "writer_a", "--action", "letters",
        "--output-root", str(output),
    ]
    assert segment_ring_imu.main(args) == 0
    captured = capsys.readouterr()
    assert "Exported 2 variable-length segments with 700 total IMU samples." in captured.out
    assert "Gravity removal: low-pass" in captured.out
    assert opened == [root / "writer_a" / "letters" / "0_ring_0.bin"]
    base = output / "writer_a" / "action_letters"
    raw = base / "writer_a_action_letters_rawIMU.npy"
    np.testing.assert_equal(np.load(raw, allow_pickle=False).shape, (700, 6))
    np.testing.assert_array_equal(
        np.load(base / "writer_a_action_letters_segment_offsets.npy", allow_pickle=False),
        [0, 350, 700],
    )

    assert segment_ring_imu.main(args) == 2
    assert "already exists" in capsys.readouterr().err
    assert segment_ring_imu.main([*args, "--overwrite"]) == 0


def test_default_output_root_is_gravity_method_specific() -> None:
    assert segment_ring_imu._default_output_root("low-pass") == Path(
        "outputs/segmentedIMU_LowPassFiltering"
    )
    assert segment_ring_imu._default_output_root("madgwick") == Path(
        "outputs/segmentedIMU_Madgwick"
    )
    assert segment_ring_imu._default_output_root("raw") == Path(
        "outputs/segmentedIMU_RawIMU"
    )
    assert segment_ring_imu._default_output_root(
        "low-pass", boundary_mode="aligned-board-events"
    ) == Path("outputs/boardAssistSegmentedIMU_LowPassFilterin")
    assert segment_ring_imu._default_output_root(
        "madgwick", boundary_mode="aligned-board-events"
    ) == Path("outputs/boardAssistSegmentedIMU_Madgwick")
    assert segment_ring_imu._default_output_root(
        "raw", boundary_mode="aligned-board-events"
    ) == Path("outputs/boardAssistSegmentedIMU_RawIMU")


def test_cli_routes_explicit_aligned_mode_with_only_aligned_options(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    captured: dict[str, object] = {}
    output_root = tmp_path / "outputs"
    paths = SimpleNamespace(
        raw_imu_path=output_root / "aligned_board_events/user_a/action_a/raw.npy",
        labels_path=output_root / "labels.npy",
        segment_offsets_path=output_root / "offsets.npy",
        segment_lengths_path=output_root / "lengths.npy",
        segments_csv_path=output_root / "segments.csv",
        summary_json_path=output_root / "summary.json",
        board_event_targets_path=output_root / "targets.npy",
        board_events_csv_path=output_root / "board_events.csv",
    )

    def fake_aligned(**kwargs: object) -> SimpleNamespace:
        captured.update(kwargs)
        return SimpleNamespace(
            raw_imu=np.empty((12, 6), dtype=np.float32),
            summary={
                "exported_segment_count": 2,
                "verification_image_count": 1,
                "recording_count": 1,
            },
            output_paths=paths,
        )

    monkeypatch.setattr(
        "writingring.board_event_segmentation.segment_user_action_by_aligned_board_events",
        fake_aligned,
    )
    assert segment_ring_imu.main(
        [
            "--data-root", str(tmp_path / "data"),
            "--user", "user_a", "--action", "a",
            "--output-root", str(output_root),
            "--boundary-mode", "aligned-board-events",
            "--alignment-offset-root", str(tmp_path / "offsets"),
            "--gravity-removal-method", "raw",
            "--pre-press-context-seconds", "0.3",
            "--post-lift-context-seconds", "0.4",
            "--verification-panel-seconds", "7",
            "--verification-dpi", "123",
        ]
    ) == 0

    config = captured["config"]
    verification = captured["verification_config"]
    assert config.pre_press_context_us == 300_000.0
    assert config.post_lift_context_us == 400_000.0
    assert captured["gravity_config"].gravity_removal_method == "raw"
    assert verification.panel_duration_s == 7.0
    assert verification.output_dpi == 123
    assert "Board-guided" in capsys.readouterr().out


def test_cli_rejects_aligned_options_in_label_mode(tmp_path: Path, capsys) -> None:
    assert segment_ring_imu.main(
        [
            "--data-root", str(tmp_path / "data"),
            "--user", "user_a", "--action", "a",
            "--alignment-offset-root", str(tmp_path / "offsets"),
        ]
    ) == 2
    assert "requires --overlay-aligned-board-events" in capsys.readouterr().err


def test_cli_exports_raw_imu_in_label_mode(tmp_path: Path, capsys) -> None:
    root = _data_root(tmp_path)
    output = tmp_path / "outputs"
    assert segment_ring_imu.main(
        [
            "--data-root", str(root),
            "--user", "writer_a", "--action", "letters",
            "--output-root", str(output),
            "--gravity-removal-method", "raw",
        ]
    ) == 0
    exported = np.load(
        output / "writer_a" / "action_letters" / "writer_a_action_letters_rawIMU.npy",
        allow_pickle=False,
    )
    assert exported.shape == (700, 6)
    np.testing.assert_allclose(exported[:, 2], np.full(700, 9.8))
    assert "Gravity removal: raw" in capsys.readouterr().out


def test_cli_rejects_unimplemented_fallback_to_label_during_argument_parsing(
    tmp_path: Path,
    capsys,
) -> None:
    with pytest.raises(SystemExit) as error:
        segment_ring_imu.main(
            [
                "--data-root", str(tmp_path / "data"),
                "--user", "user_a", "--action", "a",
                "--boundary-mode", "aligned-board-events",
                "--alignment-offset-root", str(tmp_path / "offsets"),
                "--missing-event-policy", "fallback-to-label",
            ]
        )
    assert error.value.code == 2
    assert "invalid choice" in capsys.readouterr().err


def test_cli_requires_offset_root_in_aligned_mode(tmp_path: Path, capsys) -> None:
    assert segment_ring_imu.main(
        [
            "--data-root", str(tmp_path / "data"),
            "--user", "user_a", "--action", "a",
            "--boundary-mode", "aligned-board-events",
        ]
    ) == 2
    assert "--alignment-offset-root is required" in capsys.readouterr().err


def test_label_mode_can_write_verification_without_board_or_offset(
    tmp_path: Path,
    capsys,
) -> None:
    root = _data_root(tmp_path)
    output = tmp_path / "outputs"
    assert segment_ring_imu.main(
        [
            "--data-root", str(root), "--user", "writer_a", "--action", "letters",
            "--output-root", str(output),
            "--write-label-verification",
            "--verification-panel-seconds", "7",
        ]
    ) == 0
    base = output / "writer_a" / "action_letters"
    assert (base / "0_ring_0_segmentation_verification.png").is_file()
    assert "Label verification images: 1" in capsys.readouterr().out


def test_label_overlay_requires_label_verification(tmp_path: Path, capsys) -> None:
    assert segment_ring_imu.main(
        [
            "--data-root", str(tmp_path / "data"),
            "--user", "writer_a", "--action", "letters",
            "--overlay-aligned-board-events",
        ]
    ) == 2
    assert "requires --write-label-verification" in capsys.readouterr().err


def test_label_overlay_missing_board_returns_code_2_without_traceback(
    tmp_path: Path,
    capsys,
) -> None:
    root = _data_root(tmp_path)
    offset_root = tmp_path / "offsets"
    write_alignment_offset_txt(
        AlignmentOffset(
            user="writer_a",
            action="letters",
            dataset_id=0,
            ring_stream="ring_0",
            offset_us=0.0,
            alignment_model="constant_offset",
            alignment_success=True,
            event_coverage_ratio=1.0,
            matched_event_count=2,
            total_valid_event_count=2,
        ),
        output_path=(
            offset_root
            / "writer_a"
            / "action_letters"
            / "0_ring_board_offset.txt"
        ),
    )

    assert segment_ring_imu.main(
        [
            "--data-root", str(root), "--user", "writer_a", "--action", "letters",
            "--output-root", str(tmp_path / "outputs"),
            "--write-label-verification",
            "--overlay-aligned-board-events",
            "--alignment-offset-root", str(offset_root),
        ]
    ) == 2
    stderr = capsys.readouterr().err
    assert "no board chunk paths" in stderr
    assert "Traceback" not in stderr
