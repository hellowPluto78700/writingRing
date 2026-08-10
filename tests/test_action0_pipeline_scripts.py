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
PIPELINE_ROOT = PROJECT_ROOT / "scripts" / "action0_pipeline"
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


def test_aligned_board_cli_requests_skip_and_authorized_outcome_overwrite() -> None:
    common = COMMON_PATH.read_text(encoding="utf-8")
    align_start = common.index("pipeline_align()")
    align_end = common.index("pipeline_segment()")
    align_block = common[align_start:align_end]
    assert "--initial-interval-policy skip" in align_block
    assert "pipeline_validate_alignment_outcome" in align_block
    assert "pipeline_require_file" not in align_block

    overwrite_start = common.index("pipeline_configure_overwrite_args()")
    overwrite_end = common.index("pipeline_init()")
    overwrite_block = common[overwrite_start:overwrite_end]
    assert "--overwrite-outcome" in overwrite_block


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
