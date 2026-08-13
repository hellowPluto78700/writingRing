from __future__ import annotations

import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

import numpy as np
import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PIPELINE_ROOT = PROJECT_ROOT / "scripts" / "bash_script" / "action0_pipeline"
COMMON_PATH = PIPELINE_ROOT / "_common.bash"


def test_all_action0_entry_points_use_shared_padding_pipeline() -> None:
    entry_points = sorted(PIPELINE_ROOT.glob("[0-9][0-9]_*.sh"))

    assert len(entry_points) == 8
    for entry_point in entry_points:
        text = entry_point.read_text(encoding="utf-8")
        assert 'source "$SCRIPT_DIR/_common.bash"' in text
        completed = subprocess.run(
            ["bash", "-n", str(entry_point)],
            check=False,
            capture_output=True,
            text=True,
        )
        assert completed.returncode == 0, completed.stderr

    common = (PIPELINE_ROOT / "_common.bash").read_text(encoding="utf-8")
    assert "scripts/analyze_segment_lengths.py" in common
    assert "scripts/pad_segmented_imu.py" in common
    assert '--minimum-coverage "$PADDING_COVERAGE"' in common
    assert 'PADDING_COVERAGE="${PADDING_COVERAGE:-0.99}"' in common
    assert 'PADDING_RECOMMENDATION="${PADDING_RECOMMENDATION:-balanced}"' in common
    assert "pipeline_padding" in common


def test_action0_transform_control_is_validated_and_wired(tmp_path: Path) -> None:
    common = COMMON_PATH.read_text(encoding="utf-8")
    assert 'POST_ENCODE_TRANSFORM="${POST_ENCODE_TRANSFORM:-none}"' in common
    assert 'none|AbsRectify)' in common
    assert '"POST_ENCODE_TRANSFORM must be none or AbsRectify"' in common
    assert '--post-encode-transform "$POST_ENCODE_TRANSFORM"' in common

    environment = {
        "POST_ENCODE_TRANSFORM": "invalid",
        "OUTPUT_BASE": str(tmp_path / "outputs"),
    }
    completed = subprocess.run(
        [
            "bash",
            "-c",
            f"source {COMMON_PATH}; pipeline_init raw label",
        ],
        cwd=PROJECT_ROOT,
        env={**os.environ, **environment},
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode != 0
    assert "POST_ENCODE_TRANSFORM must be none or AbsRectify" in completed.stderr


def test_requested_transform_reaches_continue_and_qa_validation_sites() -> None:
    common = COMMON_PATH.read_text(encoding="utf-8")
    encode_start = common.index("pipeline_encode_outputs_valid()")
    encode_end = common.index("pipeline_alignment_outputs_valid()")
    qa_start = common.index("pipeline_qa()")
    encode_block = common[encode_start:encode_end]
    qa_block = common[qa_start:]

    encode_validator = encode_block[encode_block.index("pipeline_validate_spike_artifact") :]
    qa_validator = qa_block[qa_block.index("pipeline_validate_spike_artifact") :]
    assert '"$POST_ENCODE_TRANSFORM" || return 1' in encode_validator
    assert '"$POST_ENCODE_TRANSFORM"' in qa_validator


def _write_spike_artifact(
    root: Path,
    *,
    settings: object = "missing",
) -> tuple[Path, Path, Path]:
    values_path = root / "spikeIMU.npy"
    timestamps_path = root / "timestamps_us.npy"
    metadata_path = root / "metadata.json"
    np.save(values_path, np.zeros((2, 21), dtype=np.float32), allow_pickle=False)
    np.save(timestamps_path, np.arange(2, dtype=np.float64), allow_pickle=False)
    metadata: dict[str, object] = {"spike_imu": {"channel_count": 21}}
    if settings != "missing":
        metadata["settings"] = settings
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    return values_path, metadata_path, timestamps_path


def _run_spike_validator(
    values_path: Path,
    metadata_path: Path,
    timestamps_path: Path,
    requested_transform: str,
    qa_log: Path,
) -> subprocess.CompletedProcess[str]:
    command = "\n".join(
        [
            f"source {shlex.quote(str(COMMON_PATH))}",
            f"QA_LOG={shlex.quote(str(qa_log))}",
            f"PYTHON_CMD=({shlex.quote(sys.executable)})",
            "pipeline_validate_spike_artifact "
            f"{shlex.quote(str(values_path))} "
            f"{shlex.quote(str(metadata_path))} "
            f"{shlex.quote(str(timestamps_path))} "
            f"{shlex.quote(requested_transform)}",
        ]
    )
    return subprocess.run(
        ["bash", "-c", command],
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )


@pytest.mark.parametrize(
    ("settings", "requested_transform", "valid"),
    [
        ("missing", "none", True),
        (None, "none", True),
        ({}, "none", True),
        ({"post_encode_transform": None}, "none", True),
        ({"post_encode_transform": "AbsRectify"}, "AbsRectify", True),
        ("missing", "AbsRectify", False),
        (None, "AbsRectify", False),
        ({"post_encode_transform": None}, "AbsRectify", False),
        ({"post_encode_transform": "AbsRectify"}, "none", False),
    ],
)
def test_spike_validator_enforces_transform_provenance(
    tmp_path: Path,
    settings: object,
    requested_transform: str,
    valid: bool,
) -> None:
    values_path, metadata_path, timestamps_path = _write_spike_artifact(
        tmp_path,
        settings=settings,
    )
    completed = _run_spike_validator(
        values_path,
        metadata_path,
        timestamps_path,
        requested_transform,
        tmp_path / "qa.log",
    )

    assert (completed.returncode == 0) is valid
    if not valid:
        output = completed.stdout + completed.stderr
        assert "SpikeIMU post_encode_transform mismatch" in output
        assert "expected=" in output
        assert "actual=" in output


def test_continue_validation_reuses_matching_transform_and_rebuilds_on_mismatch(
    tmp_path: Path,
) -> None:
    preprocess_root = tmp_path / "preprocessedIMU" / "user" / "0" / "dataset"
    spike_root = tmp_path / "spikeEncoding" / "custom-wavelet" / "user" / "0" / "dataset"
    preprocess_root.mkdir(parents=True)
    spike_root.mkdir(parents=True)
    np.save(
        preprocess_root / "dataset_preprocessedIMU.npy",
        np.zeros((2, 3), dtype=np.float32),
        allow_pickle=False,
    )
    np.save(
        preprocess_root / "dataset_timestamps_us.npy",
        np.arange(2, dtype=np.float64),
        allow_pickle=False,
    )
    (preprocess_root / "dataset_preprocessing.json").write_text("{}", encoding="utf-8")
    np.save(spike_root / "spikes.npy", np.zeros((2, 15), dtype=np.float32), allow_pickle=False)
    np.save(spike_root / "spikeIMU.npy", np.zeros((2, 21), dtype=np.float32), allow_pickle=False)
    np.save(spike_root / "recording_offsets.npy", np.array([0, 2], dtype=np.int64), allow_pickle=False)

    metadata = {"spike_imu": {"channel_count": 21}, "settings": {"post_encode_transform": "AbsRectify"}}
    (spike_root / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")

    script = "\n".join(
        [
            f"source {shlex.quote(str(COMMON_PATH))}",
            f"QA_LOG={shlex.quote(str(tmp_path / 'qa.log'))}",
            f"PYTHON_CMD=({shlex.quote(sys.executable)})",
            "POST_ENCODE_TRANSFORM=AbsRectify",
            "RECORD_USERS=(user)",
            "RECORD_ACTIONS=(0)",
            "RECORD_DATASET_IDS=(dataset)",
            "PIPELINE_USERS=(user)",
            "ACTION=0",
            "BOUNDARY_MODE=label",
            f"PREPROCESS_ROOT={shlex.quote(str(tmp_path / 'preprocessedIMU'))}",
            f"SPIKE_ROOT={shlex.quote(str(tmp_path / 'spikeEncoding/custom-wavelet'))}",
            f"SEGMENT_ROOT={shlex.quote(str(tmp_path / 'segmentation'))}",
            f"PADDING_ANALYSIS_DIR={shlex.quote(str(tmp_path / 'padding-analysis'))}",
            f"PADDING_OUTPUT_ROOT={shlex.quote(str(tmp_path / 'segmentation-padded'))}",
            "OVERWRITE=0",
            "pipeline_plan_continue",
            'printf "stage=%s force=%s overwrite=%s\\n" "$PIPELINE_RESUME_STAGE" "$PIPELINE_FORCE_REBUILD" "$OVERWRITE"',
        ]
    )
    matching = subprocess.run(
        ["bash", "-c", script],
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert matching.returncode == 0, matching.stdout + matching.stderr
    assert "stage=segment force=0 overwrite=0" in matching.stdout

    mismatch_script = script.replace("POST_ENCODE_TRANSFORM=AbsRectify", "POST_ENCODE_TRANSFORM=none")
    mismatching = subprocess.run(
        ["bash", "-c", mismatch_script],
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert mismatching.returncode == 0, mismatching.stdout + mismatching.stderr
    assert "stage=preprocess force=1 overwrite=1" in mismatching.stdout
    assert "rebuilding from preprocess with overwrite enabled" in mismatching.stdout


@pytest.mark.parametrize(
    ("old_dependency", "current_dependency"),
    [
        (
            {
                "source_recording_ids": [{"user": "u", "action": "0", "dataset_id": 0}],
                "outcomes_by_status": {
                    "SUCCESS": [
                        {
                            "identity": {"user": "u", "action": "0", "dataset_id": 0},
                            "report_sha256": "a" * 64,
                        }
                    ],
                    "SKIPPED": [],
                },
            },
            {
                "source_recording_ids": [{"user": "u", "action": "0", "dataset_id": 0}],
                "outcomes_by_status": {
                    "SUCCESS": [],
                    "SKIPPED": [
                        {
                            "identity": {"user": "u", "action": "0", "dataset_id": 0},
                            "reason": "alignment_initial_interval_skipped",
                            "report_sha256": "b" * 64,
                        }
                    ],
                },
            },
        ),
        (
            {
                "source_recording_ids": [{"user": "u", "action": "0", "dataset_id": 0}],
                "outcomes_by_status": {
                    "SUCCESS": [],
                    "SKIPPED": [
                        {
                            "identity": {"user": "u", "action": "0", "dataset_id": 0},
                            "reason": "alignment_initial_interval_skipped",
                            "report_sha256": "a" * 64,
                        }
                    ],
                },
            },
            {
                "source_recording_ids": [{"user": "u", "action": "0", "dataset_id": 0}],
                "outcomes_by_status": {
                    "SUCCESS": [
                        {
                            "identity": {"user": "u", "action": "0", "dataset_id": 0},
                            "report_sha256": "b" * 64,
                        }
                    ],
                    "SKIPPED": [],
                },
            },
        ),
        (
            {
                "source_recording_ids": [{"user": "u", "action": "0", "dataset_id": 0}],
                "outcomes_by_status": {
                    "SUCCESS": [
                        {
                            "identity": {"user": "u", "action": "0", "dataset_id": 0},
                            "report_sha256": "a" * 64,
                        }
                    ],
                    "SKIPPED": [],
                },
            },
            {
                "source_recording_ids": [{"user": "u", "action": "0", "dataset_id": 0}],
                "outcomes_by_status": {
                    "SUCCESS": [
                        {
                            "identity": {"user": "u", "action": "0", "dataset_id": 0},
                            "report_sha256": "b" * 64,
                        }
                    ],
                    "SKIPPED": [],
                },
            },
        ),
    ],
)
def test_aligned_dependency_transition_selects_segment_and_only_downstream_overwrite(
    tmp_path: Path,
    old_dependency: dict[str, object],
    current_dependency: dict[str, object],
) -> None:
    summary_path = (
        tmp_path
        / "segmentation"
        / "u"
        / "action_0"
        / "u_action_0_segmentation_summary.json"
    )
    summary_path.parent.mkdir(parents=True)
    summary_path.write_text(
        json.dumps({"alignment_outcome_dependency": old_dependency}),
        encoding="utf-8",
    )
    script = "\n".join(
        [
            f"source {shlex.quote(str(COMMON_PATH))}",
            f"QA_LOG={shlex.quote(str(tmp_path / 'qa.log'))}",
            f"PREPROCESS_ROOT={shlex.quote(str(tmp_path / 'preprocessed'))}",
            f"SPIKE_ROOT={shlex.quote(str(tmp_path / 'spikes'))}",
            f"SEGMENT_ROOT={shlex.quote(str(tmp_path / 'segmentation'))}",
            f"PADDING_OUTPUT_ROOT={shlex.quote(str(tmp_path / 'padded'))}",
            "PYTHON_CMD=(python3)",
            "DATA_ROOT=/unused",
            "OFFSET_ROOT=/unused",
            "ALIGNMENT_VERIFICATION_ROOT=/unused",
            "ALIGNMENT_REPORT_ROOT=/unused",
            "RECORD_USERS=(u)",
            "RECORD_ACTIONS=(0)",
            "RECORD_DATASET_IDS=(0)",
            "PIPELINE_USERS=(u)",
            "ACTION=0",
            "BOUNDARY_MODE=aligned-board-events",
            "OVERWRITE=0",
            "pipeline_preprocess_has_any_output() { return 0; }",
            "pipeline_encode_has_any_output() { return 0; }",
            "pipeline_alignment_has_any_output() { return 0; }",
            "pipeline_segment_has_any_output() { return 0; }",
            "pipeline_padding_has_any_output() { return 0; }",
            "pipeline_preprocess_outputs_valid() { return 0; }",
            "pipeline_encode_outputs_valid() { return 0; }",
            "pipeline_alignment_outputs_valid() { return 0; }",
            "pipeline_segment_outputs_valid() { return 0; }",
            "pipeline_padding_outputs_valid() { return 0; }",
            "pipeline_compute_alignment_outcome_dependency() {",
            f"  printf '%s' {shlex.quote(json.dumps(current_dependency))}",
            "}",
            "pipeline_plan_continue",
            'printf "stage=%s force=%s overwrite=%s segment_args=%s padding_args=%s\\n" '
            '"$PIPELINE_RESUME_STAGE" "$PIPELINE_FORCE_REBUILD" "$OVERWRITE" '
            '"${PIPELINE_DOWNSTREAM_SEGMENT_OVERWRITE_ARGS[*]}" '
            '"${PIPELINE_DOWNSTREAM_PADDING_OVERWRITE_ARGS[*]}"',
        ]
    )
    completed = subprocess.run(
        ["bash", "-c", script],
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert "stage=segment force=0 overwrite=0" in completed.stdout
    assert "segment_args=--overwrite" in completed.stdout
    assert "padding_args=--overwrite" in completed.stdout


@pytest.mark.parametrize(
    "readable_summary",
    [
        {"source_recording_count": 1},
        {"alignment_outcome_dependency": {"source_recording_ids": "malformed"}},
    ],
)
def test_readable_legacy_dependency_is_downstream_stale_but_structural_summary_rebuilds(
    tmp_path: Path,
    readable_summary: dict[str, object],
) -> None:
    common_setup = [
        f"source {shlex.quote(str(COMMON_PATH))}",
        f"QA_LOG={shlex.quote(str(tmp_path / 'qa.log'))}",
        f"SEGMENT_ROOT={shlex.quote(str(tmp_path / 'segmentation'))}",
        "RECORD_USERS=(u)",
        "RECORD_ACTIONS=(0)",
        "RECORD_DATASET_IDS=(0)",
        "PIPELINE_USERS=(u)",
        "ACTION=0",
        "BOUNDARY_MODE=aligned-board-events",
        "OVERWRITE=0",
        "pipeline_preprocess_has_any_output() { return 0; }",
        "pipeline_encode_has_any_output() { return 0; }",
        "pipeline_alignment_has_any_output() { return 0; }",
        "pipeline_segment_has_any_output() { return 0; }",
        "pipeline_padding_has_any_output() { return 0; }",
        "pipeline_preprocess_outputs_valid() { return 0; }",
        "pipeline_encode_outputs_valid() { return 0; }",
        "pipeline_alignment_outputs_valid() { return 0; }",
        "pipeline_padding_outputs_valid() { return 0; }",
        "pipeline_compute_alignment_outcome_dependency() { printf '{}'; }",
    ]
    readable = tmp_path / "segmentation" / "u" / "action_0"
    readable.mkdir(parents=True)
    (readable / "u_action_0_segmentation_summary.json").write_text(
        json.dumps(readable_summary),
        encoding="utf-8",
    )
    readable_script = "\n".join(
        common_setup
        + [
            "pipeline_segment_outputs_valid() { return 0; }",
            "pipeline_plan_continue",
            'printf "stage=%s force=%s\\n" "$PIPELINE_RESUME_STAGE" "$PIPELINE_FORCE_REBUILD"',
        ]
    )
    readable_result = subprocess.run(
        ["bash", "-c", readable_script],
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert readable_result.returncode == 0, readable_result.stdout + readable_result.stderr
    assert "stage=segment force=0" in readable_result.stdout

    structural_script = "\n".join(
        common_setup
        + [
            "pipeline_segment_outputs_valid() { return 1; }",
            "pipeline_plan_continue",
            'printf "stage=%s force=%s overwrite=%s\\n" "$PIPELINE_RESUME_STAGE" "$PIPELINE_FORCE_REBUILD" "$OVERWRITE"',
        ]
    )
    structural_result = subprocess.run(
        ["bash", "-c", structural_script],
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert structural_result.returncode == 0, structural_result.stdout + structural_result.stderr
    assert "stage=preprocess force=1 overwrite=1" in structural_result.stdout


def test_final_qa_reconciles_dependency_after_planning(tmp_path: Path) -> None:
    summary_path = (
        tmp_path
        / "segmentation"
        / "u"
        / "action_0"
        / "u_action_0_segmentation_summary.json"
    )
    summary_path.parent.mkdir(parents=True)
    summary_path.write_text(
        json.dumps({"alignment_outcome_dependency": {"version": "old"}}),
        encoding="utf-8",
    )
    script = "\n".join(
        [
            f"source {shlex.quote(str(COMMON_PATH))}",
            f"QA_LOG={shlex.quote(str(tmp_path / 'qa.log'))}",
            f"PREPROCESS_ROOT={shlex.quote(str(tmp_path / 'preprocessed'))}",
            f"SPIKE_ROOT={shlex.quote(str(tmp_path / 'spikes'))}",
            f"SEGMENT_ROOT={shlex.quote(str(tmp_path / 'segmentation'))}",
            f"PADDING_OUTPUT_ROOT={shlex.quote(str(tmp_path / 'padded'))}",
            "PYTHON_CMD=(python3)",
            "RING_FILES=(ring)",
            "RECORD_USERS=(u)",
            "RECORD_ACTIONS=(0)",
            "RECORD_DATASET_IDS=(0)",
            "PIPELINE_USERS=(u)",
            "ACTION=0",
            "BOUNDARY_MODE=aligned-board-events",
            "pipeline_count_files() {",
            "  case \"$2\" in",
            "    *_preprocessing.json|spikeIMU.npy|*_segmentation_summary.json|padding_dataset_summary.json) printf '1\\n';;",
            "    *) printf '0\\n';;",
            "  esac",
            "}",
            "pipeline_require_file() { :; }",
            "pipeline_validate_spike_artifact() { :; }",
            "pipeline_validate_segment_artifact() { :; }",
            "pipeline_validate_padding_artifact() { :; }",
            "pipeline_validate_alignment_outcome() { PIPELINE_LAST_ALIGNMENT_STATUS=SUCCESS; return 0; }",
            "pipeline_compute_alignment_outcome_dependency() { printf '{\"version\":\"new\"}'; }",
            "pipeline_qa",
        ]
    )
    completed = subprocess.run(
        ["bash", "-c", script],
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode != 0
    assert "alignment outcome dependency is missing, malformed, or stale" in completed.stderr


def test_downstream_segment_stage_overwrites_only_segment_and_padding_publishers(
    tmp_path: Path,
) -> None:
    calls_path = tmp_path / "calls.log"
    script = "\n".join(
        [
            f"source {shlex.quote(str(COMMON_PATH))}",
            f"LOG_ROOT={shlex.quote(str(tmp_path))}",
            f"PADDING_LOG={shlex.quote(str(tmp_path / 'padding.log'))}",
            f"SEGMENT_ROOT={shlex.quote(str(tmp_path / 'segment'))}",
            f"PADDING_ANALYSIS_DIR={shlex.quote(str(tmp_path / 'analysis'))}",
            f"PADDING_OUTPUT_ROOT={shlex.quote(str(tmp_path / 'padded'))}",
            f"CALLS={shlex.quote(str(calls_path))}",
            "PYTHON_CMD=(python3)",
            "PIPELINE_USERS=(u)",
            "ACTION=0",
            "BOUNDARY_MODE=aligned-board-events",
            "DATA_ROOT=/unused",
            "SPIKE_ROOT=/unused",
            "OFFSET_ROOT=/unused",
            "SAMPLING_RATE=200",
            "PADDING_COVERAGE=0.99",
            "PADDING_ROUND_TO=1",
            "PADDING_RECOMMENDATION=balanced",
            "PADDING_VALUE=0.0",
            "OVERWRITE=0",
            "pipeline_run_logged() { printf '%s\\n' \"$*\" >>\"$CALLS\"; }",
            "pipeline_require_file() { :; }",
            "pipeline_write_segmentation_error_report() { :; }",
            "pipeline_successful_segmentation_user_count() { printf '1\\n'; }",
            "pipeline_prepare_downstream_overwrite",
            "pipeline_execute_from_stage segment",
        ]
    )
    completed = subprocess.run(
        ["bash", "-c", script],
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    calls = calls_path.read_text(encoding="utf-8").splitlines()
    assert len(calls) == 3
    assert "scripts/segment_ring_imu.py" in calls[0]
    assert "scripts/analyze_segment_lengths.py" in calls[1]
    assert "scripts/pad_segmented_imu.py" in calls[2]
    assert all("--overwrite" in call for call in calls)


def test_report_only_segmentation_state_is_resume_valid_and_excluded_from_padding(
    tmp_path: Path,
) -> None:
    segment_root = tmp_path / "segmentation"
    report_path = (
        segment_root
        / "recording_errors"
        / "user"
        / "action_0"
        / "user_action_0_segmentation_recording_errors.json"
    )
    report_path.parent.mkdir(parents=True)
    identity = {"user": "user", "action": "0", "dataset_id": 1}
    report_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "terminal_state": "all_recordings_error",
                "input_kind": "spike-imu",
                "boundary_mode": "aligned_board_events",
                "source_recording_count": 1,
                "processed_recording_count": 0,
                "skipped_recording_count": 0,
                "alignment_skipped_recording_count": 0,
                "segmentation_error_recording_count": 1,
                "segmentation_errors": [
                    {
                        "identity": identity,
                        "alignment_status": "SUCCESS",
                        "stage": "board_event_segmentation",
                        "error_type": "BoardEventSegmentationError",
                        "message": "fixture local error",
                    }
                ],
                "alignment_outcome_dependency": {
                    "source_recording_ids": [identity],
                    "outcomes_by_status": {
                        "SUCCESS": [{"identity": identity, "report_sha256": "a" * 64}],
                        "SKIPPED": [],
                    },
                },
            }
        ),
        encoding="utf-8",
    )
    script = "\n".join(
        [
            f"source {shlex.quote(str(COMMON_PATH))}",
            f"QA_LOG={shlex.quote(str(tmp_path / 'qa.log'))}",
            f"SEGMENT_ROOT={shlex.quote(str(segment_root))}",
            f"PADDING_ANALYSIS_DIR={shlex.quote(str(tmp_path / 'analysis'))}",
            f"PADDING_OUTPUT_ROOT={shlex.quote(str(tmp_path / 'padded'))}",
            f"PYTHON_CMD=({shlex.quote(sys.executable)})",
            "PIPELINE_USERS=(user)",
            "ACTION=0",
            "BOUNDARY_MODE=aligned-board-events",
            "pipeline_validate_user_segmentation_state user 0",
            'printf "state=%s errors=%s\\n" "$PIPELINE_LAST_SEGMENTATION_STATE" "$PIPELINE_LAST_SEGMENTATION_ERROR_COUNT"',
            "pipeline_segment_outputs_valid",
                "pipeline_padding_outputs_valid",
                "pipeline_write_segmentation_error_report",
                "pipeline_preprocess_has_any_output() { return 0; }",
                "pipeline_encode_has_any_output() { return 0; }",
                "pipeline_alignment_has_any_output() { return 0; }",
                "pipeline_preprocess_outputs_valid() { return 0; }",
                "pipeline_encode_outputs_valid() { return 0; }",
                "pipeline_alignment_outputs_valid() { return 0; }",
                "pipeline_alignment_outcome_dependencies_valid() { return 0; }",
                "pipeline_plan_continue",
                'printf "resume=%s force=%s\\n" "$PIPELINE_RESUME_STAGE" "$PIPELINE_FORCE_REBUILD"',
        ]
    )
    completed = subprocess.run(
        ["bash", "-c", script],
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert "state=ALL_RECORDINGS_ERROR errors=1" in completed.stdout
    assert "resume=complete force=0" in completed.stdout
    root_report = json.loads(
        (segment_root / "segmentation_recording_error_report.json").read_text(
            encoding="utf-8"
        )
    )
    assert root_report["recording_errors"] == [
        {
            "user": "user",
            "action": "0",
            "dataset_id": 1,
            "stage": "board_event_segmentation",
            "error_type": "BoardEventSegmentationError",
            "message": "fixture local error",
        }
    ]
    assert (segment_root / "segmentation_recording_error_report.csv").is_file()
    assert not (segment_root / "user").exists()


def test_aligned_board_cli_requests_skip_and_authorized_outcome_overwrite() -> None:
    common = COMMON_PATH.read_text(encoding="utf-8")
    align_start = common.index("pipeline_align()")
    align_end = common.index("pipeline_segment()")
    align_block = common[align_start:align_end]
    assert "--initial-interval-policy skip" in align_block
    assert "--unalignable-recording-policy skip" in align_block
    assert "--recording-error-policy skip" in common
    assert "pipeline_validate_alignment_outcome" in align_block
    assert "pipeline_require_file" not in align_block

    overwrite_start = common.index("pipeline_configure_overwrite_args()")
    overwrite_end = common.index("pipeline_init()")
    overwrite_block = common[overwrite_start:overwrite_end]
    assert "--overwrite-outcome" in overwrite_block


def test_segment_command_uses_board_only_flags_for_aligned_mode(tmp_path: Path) -> None:
    calls_path = tmp_path / "calls.log"
    common_setup = [
        f"source {shlex.quote(str(COMMON_PATH))}",
        f"CALLS={shlex.quote(str(calls_path))}",
        f"LOG_ROOT={shlex.quote(str(tmp_path))}",
        "PYTHON_CMD=(python3)",
        "DATA_ROOT=/unused",
        "SPIKE_ROOT=/unused",
        f"SEGMENT_ROOT={shlex.quote(str(tmp_path / 'segmentation'))}",
        f"OFFSET_ROOT={shlex.quote(str(tmp_path / 'offsets'))}",
        "SAMPLING_RATE=200",
        "PIPELINE_USERS=(user)",
        "ACTION=0",
        "SEGMENT_OVERWRITE_ARGS=()",
        "PIPELINE_DOWNSTREAM_SEGMENT_OVERWRITE_ARGS=()",
        "pipeline_run_logged() { printf '%s\\n' \"$*\" >>\"$CALLS\"; }",
        "pipeline_write_segmentation_error_report() { :; }",
    ]

    aligned = subprocess.run(
        [
            "bash",
            "-c",
            "\n".join(
                [*common_setup, "BOUNDARY_MODE=aligned-board-events", "pipeline_segment"]
            ),
        ],
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert aligned.returncode == 0, aligned.stdout + aligned.stderr
    aligned_command = calls_path.read_text(encoding="utf-8").splitlines()[-1]
    assert "--pre-press-context-seconds 0.2" in aligned_command
    assert "--post-lift-context-seconds 0.2" in aligned_command
    assert "--maximum-segment-duration-seconds 5.0" in aligned_command
    assert "--carry-in-press-lookback-seconds 0.2" in aligned_command

    label = subprocess.run(
        [
            "bash",
            "-c",
            "\n".join([*common_setup, "BOUNDARY_MODE=label", "pipeline_segment"]),
        ],
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert label.returncode == 0, label.stdout + label.stderr
    label_command = calls_path.read_text(encoding="utf-8").splitlines()[-1]
    assert "--maximum-segment-duration-seconds" not in label_command
    assert "--carry-in-press-lookback-seconds" not in label_command


def _run_alignment_status_stub(
    tmp_path: Path,
    *,
    status: str = "SUCCESS",
    error: bool = False,
) -> subprocess.CompletedProcess[str]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    stub = tmp_path / "python_status_stub.py"
    stub.write_text(
        "import os\n"
        "import sys\n"
        "if os.environ.get('ALIGNMENT_STATUS_ERROR') == '1':\n"
        "    raise SystemExit(17)\n"
        "sys.stdout.write(os.environ['ALIGNMENT_STATUS'])\n",
        encoding="utf-8",
    )
    environment = {
        "ALIGNMENT_STATUS": status,
        "ALIGNMENT_STATUS_ERROR": "1" if error else "0",
    }
    script = "\n".join(
        [
            f"source {shlex.quote(str(COMMON_PATH))}",
            f"QA_LOG={shlex.quote(str(tmp_path / 'qa.log'))}",
            f"PYTHON_CMD=({shlex.quote(sys.executable)} {shlex.quote(str(stub))})",
            f"DATA_ROOT={shlex.quote(str(tmp_path / 'data'))}",
            f"SPIKE_ROOT={shlex.quote(str(tmp_path / 'spike'))}",
            f"OFFSET_ROOT={shlex.quote(str(tmp_path / 'offsets'))}",
            f"ALIGNMENT_VERIFICATION_ROOT={shlex.quote(str(tmp_path / 'verification'))}",
            f"ALIGNMENT_REPORT_ROOT={shlex.quote(str(tmp_path / 'reports'))}",
            "pipeline_validate_alignment_outcome user 0 4 || true",
            'printf "status=%s\\n" "$PIPELINE_LAST_ALIGNMENT_STATUS"',
        ]
    )
    return subprocess.run(
        ["bash", "-c", script],
        cwd=PROJECT_ROOT,
        env={**os.environ, **environment},
        check=False,
        capture_output=True,
        text=True,
    )


@pytest.mark.parametrize("status", ["SUCCESS", "SKIPPED"])
def test_alignment_validator_accepts_only_typed_completed_statuses(
    tmp_path: Path,
    status: str,
) -> None:
    completed = _run_alignment_status_stub(tmp_path, status=status)
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert f"status={status}" in completed.stdout


@pytest.mark.parametrize("status", ["FAILED", "MALFORMED", "", "success"])
def test_alignment_validator_rejects_invalid_statuses_and_validator_errors(
    tmp_path: Path,
    status: str,
) -> None:
    completed = _run_alignment_status_stub(tmp_path, status=status)
    assert completed.returncode == 0
    assert "status=INVALID" in completed.stdout

    errored = _run_alignment_status_stub(
        tmp_path / f"error_{status or 'empty'}",
        error=True,
    )
    assert errored.returncode == 0
    assert "status=INVALID" in errored.stdout


def test_alignment_validator_constructs_current_provenance_from_repository_loaders() -> None:
    common = COMMON_PATH.read_text(encoding="utf-8")
    validator_start = common.index("pipeline_validate_alignment_outcome()")
    validator_end = common.index("pipeline_validate_segment_package()")
    validator = common[validator_start:validator_end]
    for required_loader in (
        "discover_recordings",
        "select_recording",
        "load_recording_features",
        "load_board",
        "build_alignment_input_provenance",
        "build_board_chunk_provenance",
    ):
        assert required_loader in validator
    assert "expected_input_provenance=input_provenance" in validator
    assert "expected_board_provenance=board_provenance" in validator
    assert '"$OFFSET_ROOT"' in validator
    assert '"$ALIGNMENT_VERIFICATION_ROOT"' in validator
    assert '"$ALIGNMENT_REPORT_ROOT"' in validator
    assert "read_text" not in validator
    assert "json.loads" not in validator


def test_alignment_stage_validity_accepts_success_and_skip_but_rejects_invalid(
    tmp_path: Path,
) -> None:
    script = "\n".join(
        [
            f"source {shlex.quote(str(COMMON_PATH))}",
            "RECORD_USERS=(u u u)",
            "RECORD_ACTIONS=(0 0 0)",
            "RECORD_DATASET_IDS=(0 1 2)",
            "calls=0",
            "pipeline_validate_alignment_outcome() {",
            "  calls=$((calls + 1))",
            "  case \"$3\" in",
            "    0) PIPELINE_LAST_ALIGNMENT_STATUS=SUCCESS; return 0;;",
            "    1) PIPELINE_LAST_ALIGNMENT_STATUS=SKIPPED; return 0;;",
            "    2) PIPELINE_LAST_ALIGNMENT_STATUS=SUCCESS; return 0;;",
            "    *) PIPELINE_LAST_ALIGNMENT_STATUS=INVALID; return 1;;",
            "  esac",
            "}",
            "pipeline_alignment_outputs_valid; printf 'valid calls=%s\\n' \"$calls\"",
        ]
    )
    valid = subprocess.run(
        ["bash", "-c", script],
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert valid.returncode == 0, valid.stdout + valid.stderr
    assert "valid calls=3" in valid.stdout

    invalid = subprocess.run(
        ["bash", "-c", script.replace("RECORD_DATASET_IDS=(0 1 2)", "RECORD_DATASET_IDS=(0 2 3)")],
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert invalid.returncode != 0


def test_mixed_alignment_outcomes_continue_through_alignment_stage(
    tmp_path: Path,
) -> None:
    calls_path = tmp_path / "calls.log"
    script = "\n".join(
        [
            f"source {shlex.quote(str(COMMON_PATH))}",
            f"CALLS={shlex.quote(str(calls_path))}",
            "LOG_ROOT=/unused",
            "PYTHON_CMD=(python3)",
            "DATA_ROOT=/unused",
            "SPIKE_ROOT=/unused",
            "OFFSET_ROOT=/unused",
            "ALIGNMENT_REPORT_ROOT=/unused",
            "ALIGNMENT_VERIFICATION_ROOT=/unused",
            "BOUNDARY_MODE=aligned-board-events",
            "RECORD_USERS=(u u)",
            "RECORD_ACTIONS=(0 0)",
            "RECORD_DATASET_IDS=(0 1)",
            "ALIGN_OVERWRITE_ARGS=()",
            "pipeline_run_logged() { printf 'run:%s\\n' \"$*\" >>\"$CALLS\"; }",
            "pipeline_validate_alignment_outcome() {",
            "  case \"$3\" in",
            "    0) PIPELINE_LAST_ALIGNMENT_STATUS=SUCCESS; return 0;;",
            "    1) PIPELINE_LAST_ALIGNMENT_STATUS=SKIPPED; return 0;;",
            "    *) PIPELINE_LAST_ALIGNMENT_STATUS=INVALID; return 1;;",
            "  esac",
            "}",
            "pipeline_segment() { printf 'segment\\n' >>\"$CALLS\"; }",
            "pipeline_padding() { printf 'padding\\n' >>\"$CALLS\"; }",
            "pipeline_execute_from_stage align",
        ]
    )
    completed = subprocess.run(
        ["bash", "-c", script],
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    calls = calls_path.read_text(encoding="utf-8").splitlines()
    assert len(calls) == 4
    assert "--initial-interval-policy skip" in calls[0]
    assert "--unalignable-recording-policy skip" in calls[0]
    assert "--initial-interval-policy skip" in calls[1]
    assert "--unalignable-recording-policy skip" in calls[1]
    assert calls[2:] == ["segment", "padding"]


def test_qa_accounts_success_skipped_and_invalid_recordings_after_one_contract_call(
    tmp_path: Path,
) -> None:
    # Stub the non-alignment artifact validators so this test isolates QA's
    # recording-level outcome accounting and invalid-outcome failure boundary.
    for relative in (
        "preprocessedIMU/a_preprocessing.json",
        "preprocessedIMU/b_preprocessing.json",
        "preprocessedIMU/c_preprocessing.json",
        "spikeEncoding/a/spikeIMU.npy",
        "spikeEncoding/b/spikeIMU.npy",
        "spikeEncoding/c/spikeIMU.npy",
        "segmentation/user/user_action_0_spikeIMU.npy",
        "segmentation/user/user_action_0_labels.npy",
        "segmentation/user/user_action_0_segment_offsets.npy",
        "segmentation/user/user_action_0_segment_lengths.npy",
        "segmentation/user/user_action_0_segments.csv",
        "segmentation/user/user_action_0_segmentation_summary.json",
        "segmentation/user/user_action_0_board_event_targets.npy",
        "segmentation/user/user_action_0_board_events.csv",
        "segmentation-padded/padding_dataset_summary.json",
    ):
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"artifact")
    qa_log = tmp_path / "qa.log"
    script = "\n".join(
        [
            f"source {shlex.quote(str(COMMON_PATH))}",
            f"QA_LOG={shlex.quote(str(qa_log))}",
            f"RING_FILES=(a b c)",
            f"RECORD_USERS=(u u u)",
            f"RECORD_ACTIONS=(0 0 0)",
            f"RECORD_DATASET_IDS=(0 1 2)",
            f"PIPELINE_USERS=(user)",
            "ACTION=0",
            "BOUNDARY_MODE=aligned-board-events",
            f"PREPROCESS_ROOT={shlex.quote(str(tmp_path / 'preprocessedIMU'))}",
            f"SPIKE_ROOT={shlex.quote(str(tmp_path / 'spikeEncoding'))}",
            f"SEGMENT_ROOT={shlex.quote(str(tmp_path / 'segmentation'))}",
            f"PADDING_OUTPUT_ROOT={shlex.quote(str(tmp_path / 'segmentation-padded'))}",
            "pipeline_validate_spike_artifact() { :; }",
            "pipeline_validate_segment_artifact() { :; }",
            "pipeline_validate_padding_artifact() { :; }",
            "pipeline_validate_alignment_outcome() {",
            "  case \"$3\" in",
            "    0) PIPELINE_LAST_ALIGNMENT_STATUS=SUCCESS; return 0;;",
            "    1) PIPELINE_LAST_ALIGNMENT_STATUS=SKIPPED; return 0;;",
            "    *) PIPELINE_LAST_ALIGNMENT_STATUS=INVALID; return 1;;",
            "  esac",
            "}",
            "pipeline_qa",
        ]
    )
    completed = subprocess.run(
        ["bash", "-c", script],
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode != 0
    assert "Alignment outcomes: total=3 success=1 skipped=1 invalid=1" in completed.stdout
    log = qa_log.read_text(encoding="utf-8")
    assert "alignment_outcomes_total=3" in log
    assert "alignment_outcomes_success=1" in log
    assert "alignment_outcomes_skipped=1" in log
    assert "alignment_outcomes_invalid=1" in log


def test_qa_continues_after_mixed_success_and_skipped_alignment_outcomes(
    tmp_path: Path,
) -> None:
    for relative in (
        "preprocessedIMU/a_preprocessing.json",
        "preprocessedIMU/b_preprocessing.json",
        "spikeEncoding/a/spikeIMU.npy",
        "spikeEncoding/b/spikeIMU.npy",
        "segmentation/user/user_action_0_segmentation_summary.json",
        "segmentation/user/user_action_0_spikeIMU.npy",
        "segmentation-padded/padding_dataset_summary.json",
    ):
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"artifact")

    qa_log = tmp_path / "qa.log"
    script = "\n".join(
        [
            f"source {shlex.quote(str(COMMON_PATH))}",
            f"QA_LOG={shlex.quote(str(qa_log))}",
            "POST_ENCODE_TRANSFORM=none",
            "COMBINATION_ROOT=/unused",
            "LOG_ROOT=/unused",
            "RING_FILES=(a b)",
            "RECORD_USERS=(u u)",
            "RECORD_ACTIONS=(0 0)",
            "RECORD_DATASET_IDS=(0 1)",
            "PIPELINE_USERS=(user)",
            "ACTION=0",
            "BOUNDARY_MODE=aligned-board-events",
            f"PREPROCESS_ROOT={shlex.quote(str(tmp_path / 'preprocessedIMU'))}",
            f"SPIKE_ROOT={shlex.quote(str(tmp_path / 'spikeEncoding'))}",
            f"SEGMENT_ROOT={shlex.quote(str(tmp_path / 'segmentation'))}",
            f"PADDING_OUTPUT_ROOT={shlex.quote(str(tmp_path / 'segmentation-padded'))}",
            "pipeline_require_file() { :; }",
            "pipeline_validate_spike_artifact() { :; }",
            "pipeline_validate_segment_artifact() { :; }",
            "pipeline_validate_padding_artifact() { :; }",
            "pipeline_validate_alignment_outcome() {",
            "  case \"$3\" in",
            "    0) PIPELINE_LAST_ALIGNMENT_STATUS=SUCCESS; return 0;;",
            "    1) PIPELINE_LAST_ALIGNMENT_STATUS=SKIPPED; return 0;;",
            "    *) PIPELINE_LAST_ALIGNMENT_STATUS=INVALID; return 1;;",
            "  esac",
            "}",
            "pipeline_alignment_outcome_dependencies_valid() { return 0; }",
            "pipeline_validate_user_segmentation_state() { PIPELINE_LAST_SEGMENTATION_STATE=PACKAGE_NO_ERRORS; PIPELINE_LAST_SEGMENTATION_ERROR_COUNT=0; return 0; }",
            "pipeline_write_segmentation_error_report() { :; }",
            "pipeline_qa",
        ]
    )
    completed = subprocess.run(
        ["bash", "-c", script],
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert "Alignment outcomes: total=2 success=1 skipped=1 invalid=0" in completed.stdout
    assert "QA passed for 2 ring_0 recording(s), 1 user(s), 1 successful packages, and 0 segmentation recording error(s)." in completed.stdout
    log = qa_log.read_text(encoding="utf-8")
    assert "alignment_outcomes_total=2" in log
    assert "alignment_outcomes_success=1" in log
    assert "alignment_outcomes_skipped=1" in log
    assert "alignment_outcomes_invalid=0" in log
