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
