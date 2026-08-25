from __future__ import annotations

import json
from pathlib import Path
import shlex
import subprocess
import sys

import numpy as np
import pytest

from scripts import encode_spikes
from writingring.imu_preprocessing import PREPROCESSED_IMU_COLUMNS, STANDARD_GRAVITY_M_S2
from writingring.preprocessing_io import PREPROCESSED_IMU_UNITS, sha256_file
from writingring.resampling import ResamplingError, resample_recording


PROJECT_ROOT = Path(__file__).resolve().parents[1]
COMMON_PATH = PROJECT_ROOT / "scripts" / "bash_script" / "preprocessing_pipeline" / "_common.bash"


def _source_artifact(tmp_path: Path) -> tuple[Path, Path, Path]:
    root = tmp_path / "preprocessedIMU" / "user" / "0" / "3"
    root.mkdir(parents=True)
    timestamps = np.arange(401, dtype=np.float64) * 5_000.0
    timestamps[200] = timestamps[199]  # Canonical provenance is allowed to duplicate.
    phase = np.arange(len(timestamps), dtype=np.float64) / 200.0
    acceleration_g = np.column_stack((np.sin(2 * np.pi * 3 * phase), phase, -phase))
    values = np.column_stack((
        acceleration_g,
        acceleration_g * STANDARD_GRAVITY_M_S2,
        np.column_stack((np.cos(phase), phase * 2, phase * -3)),
    )).astype(np.float32)
    imu_path = root / "3_preprocessedIMU.npy"
    timestamp_path = root / "3_timestamps_us.npy"
    summary_path = root / "3_preprocessing.json"
    np.save(imu_path, values, allow_pickle=False)
    np.save(timestamp_path, timestamps, allow_pickle=False)
    summary_path.write_text(json.dumps({
        "schema_version": 1,
        "artifact_type": "preprocessed_imu",
        "recording": {"user": "user", "action": "0", "data_id": 3},
        "sample_count": len(values), "channel_count": 9,
        "channel_names": list(PREPROCESSED_IMU_COLUMNS), "sampling_rate_hz": 200.0,
        "standard_gravity_m_s2": STANDARD_GRAVITY_M_S2,
        "acceleration_semantics": "gravity_removed_linear_acceleration",
        "gravity_removal_method": "low-pass", "gravity_removed": True,
        "units": list(PREPROCESSED_IMU_UNITS), "timestamp_unit": "microseconds",
        "source_file": str(imu_path.resolve()), "source_file_sha256": sha256_file(imu_path),
        "timestamps_path": str(timestamp_path.resolve()), "timestamps_sha256": sha256_file(timestamp_path),
        "timestamp_source_path": str(timestamp_path.resolve()), "timestamp_sha256": sha256_file(timestamp_path),
    }, indent=2), encoding="utf-8")
    return imu_path, summary_path, timestamp_path


def test_resampling_preserves_provenance_and_recomputes_gravity_columns(tmp_path: Path) -> None:
    imu, summary, timestamps = _source_artifact(tmp_path)
    result = resample_recording(
        input_imu_path=imu, input_summary_path=summary, input_timestamps_path=timestamps,
        output_root=tmp_path / "resampledIMU", relative_recording=Path("user/0/3"), target_rate_hz=64.0,
    )
    assert result.summary["sampling_rate_hz"] == 64.0
    resampling = result.summary["resampling"]
    assert resampling["source_imu_sha256"] == sha256_file(imu)  # type: ignore[index]
    assert resampling["resampling_method"] == "scipy.signal.resample_poly"  # type: ignore[index]
    assert resampling["rate_conversion"] == {"up": 8, "down": 25}  # type: ignore[index]
    assert resampling["anti_alias_filter"] == {  # type: ignore[index]
        "family": "polyphase_fir",
        "implementation": "scipy.signal.resample_poly",
        "window": "kaiser",
        "kaiser_beta": 5.0,
        "padtype": "line",
    }
    assert result.timestamps[0] == 0.0
    assert result.timestamps[-1] <= np.load(timestamps, allow_pickle=False)[-1]
    assert np.all(np.diff(result.timestamps) > 0)
    assert len(result.imu) == len(result.timestamps) == 129
    np.testing.assert_allclose(
        result.imu[:, :3] * STANDARD_GRAVITY_M_S2,
        result.imu[:, 3:6],
        rtol=1e-6,
        atol=1e-7,
    )
    assert result.paths.summary_path.name == "3_resampling.json"


def test_polyphase_resampling_preserves_low_frequency_amplitude(tmp_path: Path) -> None:
    imu, summary, timestamps = _source_artifact(tmp_path)
    source = np.load(imu, allow_pickle=False)
    result = resample_recording(
        input_imu_path=imu, input_summary_path=summary, input_timestamps_path=timestamps,
        output_root=tmp_path / "resampledIMU", relative_recording=Path("user/0/3"), target_rate_hz=64.0,
    )

    # Compare the 3 Hz acceleration channel away from the recording boundaries.
    # The resampler should preserve passband amplitude rather than introducing a
    # sample-rate-dependent gain change.
    source_margin = 40
    target_margin = 13
    source_rms = float(np.sqrt(np.mean(source[source_margin:-source_margin, 0] ** 2)))
    target_rms = float(np.sqrt(np.mean(result.imu[target_margin:-target_margin, 0] ** 2)))
    assert target_rms / source_rms == pytest.approx(1.0, rel=0.02)


def test_resampling_rejects_upsampling(tmp_path: Path) -> None:
    imu, summary, timestamps = _source_artifact(tmp_path)
    with pytest.raises(ResamplingError, match="upsampling"):
        resample_recording(
            input_imu_path=imu, input_summary_path=summary, input_timestamps_path=timestamps,
            output_root=tmp_path / "resampledIMU", relative_recording=Path("user/0/3"), target_rate_hz=200.0,
        )


def test_encoder_uses_resampled_rate_and_carries_resampling_provenance(tmp_path: Path) -> None:
    imu, summary, timestamps = _source_artifact(tmp_path)
    result = resample_recording(
        input_imu_path=imu, input_summary_path=summary, input_timestamps_path=timestamps,
        output_root=tmp_path / "resampledIMU", relative_recording=Path("user/0/3"), target_rate_hz=64.0,
    )
    settings = tmp_path / "encoder.json"
    settings.write_text(json.dumps({"sampling_rate_hz": 200.0}), encoding="utf-8")
    assert encode_spikes.main([
        "--input-imu", str(result.paths.imu_path), "--input-summary", str(result.paths.summary_path),
        "--encoder", "custom-wavelet", "--encoder-settings", str(settings),
        "--effective-sampling-rate-hz", "64", "--output-root", str(tmp_path / "spikes"),
    ]) == 0
    published = json.loads((tmp_path / "spikes" / "custom-wavelet" / "3_resampledIMU" / "3_resampledIMU_spike_encoding_summary.json").read_text(encoding="utf-8"))
    assert published["settings"]["sampling_rate_hz"] == 64.0
    assert published["spike_encoder"]["wavelet_widths_samples"] == [128, 64, 32, 16, 8]
    assert published["resampling"]["target_sampling_rate_hz"] == 64.0
    assert published["resampling"]["rate_conversion"] == {"up": 8, "down": 25}


def test_continue_rebuilds_resampling_and_downstream_when_rate_changes(tmp_path: Path) -> None:
    imu, summary, timestamps = _source_artifact(tmp_path)
    resample_recording(
        input_imu_path=imu, input_summary_path=summary, input_timestamps_path=timestamps,
        output_root=tmp_path / "resampledIMU", relative_recording=Path("user/0/3"), target_rate_hz=64.0,
    )
    script = "\n".join((
        f"source {shlex.quote(str(COMMON_PATH))}",
        f"PREPROCESS_ROOT={shlex.quote(str(tmp_path / 'preprocessedIMU'))}",
        f"RESAMPLE_ROOT={shlex.quote(str(tmp_path / 'resampledIMU'))}",
        "RESAMPLE_RATE_HZ=128", f"PYTHON_CMD=({shlex.quote(sys.executable)})",
        "RECORD_USERS=(user)", "RECORD_ACTIONS=(0)", "RECORD_DATASET_IDS=(3)",
        "BOUNDARY_MODE=label", "OVERWRITE=0", "pipeline_preprocess_has_any_output() { return 0; }",
        "pipeline_preprocess_outputs_valid() { return 0; }", "pipeline_encode_has_any_output() { return 0; }",
        "pipeline_segment_has_any_output() { return 1; }", "pipeline_padding_has_any_output() { return 1; }",
        "pipeline_plan_continue", 'printf "stage=%s overwrite=%s\\n" "$PIPELINE_RESUME_STAGE" "${RESAMPLE_OVERWRITE_ARGS[*]}"',
    ))
    completed = subprocess.run(["bash", "-c", script], cwd=PROJECT_ROOT, text=True, capture_output=True, check=False)
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert "stage=resample overwrite=--overwrite" in completed.stdout
