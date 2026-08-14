from pathlib import Path
import sys
from types import SimpleNamespace

import scripts.plot_board_segment_trajectories as trajectory_plotter

from scripts.plot_board_segment_trajectories import (
    action_output_root,
    build_parser,
    default_segmentation_root,
)


def test_default_segmentation_root_is_action_specific() -> None:
    assert default_segmentation_root("0") == Path(
        "outputs/action0_rectified/low-pass/aligned-board-events/segmentation"
    )
    assert default_segmentation_root("1") == Path(
        "outputs/action1_rectified/low-pass/aligned-board-events/segmentation"
    )


def test_parser_uses_dynamic_segmentation_root_and_action_output_root() -> None:
    args = build_parser().parse_args(
        ["--data-root", "data", "--action", "1"]
    )

    assert args.segmentation_root is None
    assert default_segmentation_root(args.action) == Path(
        "outputs/action1_rectified/low-pass/aligned-board-events/segmentation"
    )
    assert action_output_root(args.output_root, args.action) == Path(
        "outputs/plotting_verification/action1"
    )


def test_explicit_output_root_remains_action_isolated() -> None:
    args = build_parser().parse_args(
        [
            "--data-root",
            "data",
            "--action",
            "0",
            "--segmentation-root",
            "custom/segmentation",
            "--output-root",
            "custom/plots",
        ]
    )

    assert args.segmentation_root == Path("custom/segmentation")
    assert action_output_root(args.output_root, args.action) == Path(
        "custom/plots/action0"
    )


def test_main_reads_action_manifest_and_passes_action_output_root(
    tmp_path: Path,
    monkeypatch,
) -> None:
    segmentation_root = tmp_path / "segmentation"
    manifest_path = (
        segmentation_root
        / "user_0"
        / "action_1"
        / "user_0_action_1_segments.csv"
    )
    manifest_path.parent.mkdir(parents=True)
    manifest_path.write_text(
        "dataset_id,segment_index,exported,label,start_sample_index,"
        "stop_sample_index_exclusive,final_start_timestamp_us,"
        "final_end_timestamp_us,alignment_offset_us,work_axis_offset_us,"
        "alignment_offset_domain,input_kind\n"
        "0,0,true,i,0,1,1000,2000,0,0,canonical_timestamp,spike-imu\n",
        encoding="utf-8",
    )

    recording = SimpleNamespace(user="user_0", action="1", dataset_id=0)
    captured: dict[str, object] = {}

    monkeypatch.setattr(
        trajectory_plotter,
        "resolve_recordings",
        lambda *args, **kwargs: [recording],
    )

    def fake_process_recording(**kwargs: object) -> dict[str, int]:
        captured.update(kwargs)
        return {
            "segments": 1,
            "written": 1,
            "skipped_existing": 0,
            "contact_samples": 0,
            "tracks": 0,
        }

    monkeypatch.setattr(
        trajectory_plotter,
        "process_recording",
        fake_process_recording,
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "plot_board_segment_trajectories.py",
            "--data-root",
            str(tmp_path / "data"),
            "--user",
            "user_0",
            "--action",
            "1",
            "--segmentation-root",
            str(segmentation_root),
            "--output-root",
            str(tmp_path / "plots"),
        ],
    )

    assert trajectory_plotter.main() == 0
    assert captured["manifest_path"] == manifest_path
    assert captured["output_root"] == tmp_path / "plots" / "action1"
