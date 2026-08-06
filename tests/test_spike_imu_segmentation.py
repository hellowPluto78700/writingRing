from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from scripts import encode_spikes, segment_ring_imu
from writingring.alignment_io import (
    AlignmentOffset,
    AlignmentOffsetExportError,
    write_alignment_offset_txt,
)
from writingring.discovery import discover_recordings
from writingring.imu_preprocessing import PREPROCESSED_IMU_COLUMNS
from writingring.preprocessing_io import PREPROCESSED_IMU_UNITS, sha256_file
from writingring.recording_features import (
    SPIKE_IMU_FEATURE_SCHEMA,
    SPIKE_IMU_TRANSIENT_CHANNEL_NAMES,
    RecordingFeatureInput,
    load_spike_imu_features,
)
from writingring.segmentation import SegmentationConfig, segment_user_action


def _data_root(tmp_path: Path, *, sample_count: int = 400) -> Path:
    root = tmp_path / "data"
    action_dir = root / "writer_a" / "letters"
    action_dir.mkdir(parents=True)
    timestamps = 1_000_000.0 + np.arange(sample_count, dtype=np.float64) * 1_000.0
    rows = np.column_stack(
        (
            np.zeros((sample_count, 2)),
            np.full(sample_count, 9.80665),
            np.zeros((sample_count, 3)),
            timestamps,
        )
    )
    rows.tofile(action_dir / "0_ring_0.bin")
    (action_dir / "0_ring_1.bin").write_bytes(b"must not be opened")
    (action_dir / "0_timestamp.txt").write_text(
        "1000000 first\n1200000 last\n",
        encoding="utf-8",
    )
    return root


def _write_spike_artifact(
    tmp_path: Path,
    data_root: Path,
    *,
    sample_count: int = 400,
) -> tuple[Path, np.ndarray]:
    recording = discover_recordings(data_root)[0]
    directory = tmp_path / "spike" / recording.user / recording.action / str(recording.dataset_id)
    directory.mkdir(parents=True)
    values = np.arange(sample_count * 21, dtype=np.float32).reshape(sample_count, 21)
    timestamps = 1_000_000.0 + np.arange(sample_count, dtype=np.float64) * 1_000.0
    values_path = directory / "spikeIMU.npy"
    timestamps_path = directory / "0_timestamps_us.npy"
    metadata_path = directory / "metadata.json"
    np.save(values_path, values, allow_pickle=False)
    np.save(timestamps_path, timestamps, allow_pickle=False)
    payload = {
        "schema_version": 3,
        "metadata_schema": "spike_encoding_v3",
        "recording": {
            "user": recording.user,
            "action": recording.action,
            "data_id": recording.dataset_id,
        },
        "sampling_rate_hz": 1_000.0,
        "timestamps_path": str(timestamps_path.resolve()),
        "timestamps_sha256": sha256_file(timestamps_path),
        "timestamp_unit": "microseconds",
        "input": {
            "sample_count": sample_count,
            "channel_count": len(PREPROCESSED_IMU_COLUMNS),
            "sampling_rate_hz": 1_000.0,
        },
        "spike_imu": {
            "schema": SPIKE_IMU_FEATURE_SCHEMA,
            "sample_count": sample_count,
            "channel_count": 21,
            "channel_names": [f"event_{i}" for i in range(15)]
            + list(SPIKE_IMU_TRANSIENT_CHANNEL_NAMES),
            "units": ["event"] * 15 + list(PREPROCESSED_IMU_UNITS[3:]),
            "sha256": sha256_file(values_path),
        },
    }
    metadata_path.write_text(json.dumps(payload), encoding="utf-8")
    return tmp_path / "spike", values


def _matching_spike_offset(feature_input: RecordingFeatureInput) -> AlignmentOffset:
    """Build the provenance-bearing offset used by SpikeIMU overlay tests."""

    return AlignmentOffset(
        user=feature_input.user,
        action=feature_input.action,
        dataset_id=feature_input.dataset_id,
        ring_stream="ring_0",
        offset_us=0.0,
        alignment_model="constant_offset",
        alignment_success=True,
        event_coverage_ratio=1.0,
        matched_event_count=1,
        total_valid_event_count=1,
        alignment_signal_source=feature_input.input_kind,
        feature_schema=feature_input.feature_schema,
        feature_values_sha256=feature_input.values_sha256,
        feature_metadata_sha256=feature_input.metadata_sha256,
        timestamp_sha256=feature_input.timestamps_sha256,
        transient_channel_indices=feature_input.transient_channel_indices,
        spike_event_channels_used=False,
    )


def test_spike_label_segmentation_publishes_21_channel_slices(
    tmp_path: Path,
) -> None:
    data_root = _data_root(tmp_path)
    spike_root, source_values = _write_spike_artifact(tmp_path, data_root)

    result = segment_user_action(
        data_root=data_root,
        user="writer_a",
        action="letters",
        output_root=tmp_path / "segmented",
        input_kind="spike-imu",
        spike_root=spike_root,
        expected_sampling_rate_hz=1_000.0,
    )

    assert result.raw_imu.shape == (400, 21)
    assert result.summary["input_kind"] == "spike-imu"
    assert result.summary["feature_schema"] == SPIKE_IMU_FEATURE_SCHEMA
    assert result.summary["channel_count"] == 21
    assert result.summary["boundary_source"] == "timestamp_labels"
    assert result.summary["alignment_required"] is False
    assert result.summary["event_channel_slice"] == [0, 15]
    assert result.summary["acceleration_m_s2_channel_slice"] == [15, 18]
    assert result.summary["gyroscope_channel_slice"] == [18, 21]
    assert result.output_paths.raw_imu_path.name.endswith("_spikeIMU.npy")
    np.testing.assert_array_equal(result.raw_imu, source_values)
    np.testing.assert_array_equal(result.segment_lengths, [200, 200])
    np.testing.assert_array_equal(result.segment_offsets, [0, 200, 400])
    assert result.manifest["input_kind"].tolist() == ["spike-imu", "spike-imu"]
    assert result.output_paths.raw_imu_path.is_file()


def test_canonical_batch_encoding_metadata_is_consumable_by_spike_loader(
    tmp_path: Path,
    capsys,
) -> None:
    data_root = _data_root(tmp_path, sample_count=80)
    recording = discover_recordings(data_root)[0]
    preprocessed_directory = (
        tmp_path / "preprocessed" / recording.user / recording.action / "0"
    )
    preprocessed_directory.mkdir(parents=True)
    acceleration_g = np.column_stack(
        (
            np.sin(np.arange(80) / 5.0),
            np.cos(np.arange(80) / 7.0),
            np.sin(np.arange(80) / 9.0),
        )
    )
    preprocessed = np.column_stack(
        (
            acceleration_g,
            acceleration_g * 9.80665,
            np.zeros((80, 3), dtype=np.float64),
        )
    )
    imu_path = preprocessed_directory / "0_preprocessedIMU.npy"
    timestamps_path = preprocessed_directory / "0_timestamps_us.npy"
    summary_path = preprocessed_directory / "0_preprocessing.json"
    timestamps = 1_000_000.0 + np.arange(80, dtype=np.float64) * 5_000.0
    np.save(imu_path, preprocessed, allow_pickle=False)
    np.save(timestamps_path, timestamps, allow_pickle=False)
    summary_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "artifact_type": "preprocessed_imu",
                "recording": {
                    "user": recording.user,
                    "action": recording.action,
                    "data_id": recording.dataset_id,
                },
                "sample_count": 80,
                "channel_count": 9,
                "channel_names": list(PREPROCESSED_IMU_COLUMNS),
                "sampling_rate_hz": 200.0,
                "standard_gravity_m_s2": 9.80665,
                "acceleration_semantics": "gravity_removed_linear_acceleration",
                "gravity_removal_method": "low-pass",
                "gravity_removed": True,
                "units": list(PREPROCESSED_IMU_UNITS),
                "source_file": str(imu_path.resolve()),
                "source_file_sha256": sha256_file(imu_path),
                "timestamps_path": str(timestamps_path.resolve()),
                "timestamps_sha256": sha256_file(timestamps_path),
                "timestamp_unit": "microseconds",
            }
        ),
        encoding="utf-8",
    )
    settings_path = tmp_path / "settings.json"
    settings_path.write_text(json.dumps({"sampling_rate_hz": 200.0}), encoding="utf-8")

    assert encode_spikes.main(
        [
            "--input-root", str(tmp_path / "preprocessed"),
            "--output-root", str(tmp_path / "encoded"),
            "--encoder", "custom-wavelet",
            "--encoder-settings", str(settings_path),
        ]
    ) == 0

    spike_root = tmp_path / "encoded" / "custom-wavelet"
    loaded = load_spike_imu_features(recording, spike_root=spike_root)
    assert loaded.values.shape == (80, 21)
    np.testing.assert_array_equal(loaded.timestamps_us, timestamps)
    metadata = json.loads(
        (
            spike_root / recording.user / recording.action / "0" / "metadata.json"
        ).read_text(encoding="utf-8")
    )
    assert metadata["source"]["timestamp_source_path"] == str(timestamps_path.resolve())
    assert metadata["source"]["source_hash_verified"] is True
    assert metadata["timestamp_source_hash_verified"] is True
    assert metadata["spike_imu"]["units"] == ["event"] * 15 + list(
        PREPROCESSED_IMU_UNITS[3:]
    )

    np.save(timestamps_path, timestamps + 1_000_000.0, allow_pickle=False)
    assert encode_spikes.main(
        [
            "--input-root", str(tmp_path / "preprocessed"),
            "--output-root", str(tmp_path / "encoded"),
            "--encoder", "custom-wavelet",
            "--encoder-settings", str(settings_path),
            "--overwrite",
        ]
    ) == 2
    assert "timestamp source SHA-256" in capsys.readouterr().err


def test_spike_label_cli_never_reapplies_gravity_and_rejects_invalid_modes(
    tmp_path: Path,
    capsys,
) -> None:
    data_root = _data_root(tmp_path)
    spike_root, _ = _write_spike_artifact(tmp_path, data_root)
    common = [
        "--data-root", str(data_root),
        "--user", "writer_a", "--action", "letters",
        "--input-kind", "spike-imu", "--spike-root", str(spike_root),
        "--output-root", str(tmp_path / "cli-output"),
    ]

    assert segment_ring_imu.main(common) == 0
    output = (
        tmp_path / "cli-output" / "writer_a" / "action_letters"
        / "writer_a_action_letters_spikeIMU.npy"
    )
    assert np.load(output, allow_pickle=False).shape == (400, 21)
    assert "SpikeIMU:" in capsys.readouterr().out

    verification_root = tmp_path / "cli-verification"
    assert segment_ring_imu.main(
        [
            *common[:-2],
            "--output-root", str(verification_root),
            "--write-label-verification",
        ]
    ) == 0
    assert (
        verification_root / "writer_a" / "action_letters"
        / "0_segmentation_verification.png"
    ).is_file()
    assert "Board" not in capsys.readouterr().out

    assert segment_ring_imu.main(
        [
            "--data-root", str(data_root),
            "--user", "writer_a", "--action", "letters",
            "--input-kind", "spike-imu",
        ]
    ) == 2
    assert "--spike-root is required" in capsys.readouterr().err

    assert segment_ring_imu.main(
        [*common, "--gravity-removal-method", "low-pass"]
    ) == 2
    assert "not valid for --input-kind spike-imu" in capsys.readouterr().err

    assert segment_ring_imu.main(
        [*common, "--boundary-mode", "aligned-board-events",
         "--alignment-offset-root", str(tmp_path / "offsets")]
    ) == 2
    assert "PR 4" in capsys.readouterr().err


def test_spike_label_verification_overlay_does_not_change_boundaries(
    tmp_path: Path,
    monkeypatch,
) -> None:
    data_root = _data_root(tmp_path)
    spike_root, _ = _write_spike_artifact(tmp_path, data_root)
    recording = discover_recordings(data_root)[0]
    feature_input = load_spike_imu_features(
        recording,
        spike_root=spike_root,
        expected_sampling_rate_hz=1_000.0,
    )
    offset_root = tmp_path / "offsets"
    write_alignment_offset_txt(
        _matching_spike_offset(feature_input),
        output_path=(
            offset_root / "writer_a" / "action_letters" / "0_ring_board_offset.txt"
        ),
    )
    captured: list[tuple[tuple[int, int, str], ...]] = []
    monkeypatch.setattr(
        "writingring.segmentation_verification.create_segmentation_verification_figure",
        lambda **kwargs: captured.append(
            tuple(
                (
                    sample.start_sample_index,
                    sample.stop_sample_index_exclusive,
                    sample.label,
                )
                for sample in kwargs["segmented_samples"]
            )
        ),
    )
    empty_events = pd.DataFrame()
    monkeypatch.setattr(
        "writingring.board_loader.load_board",
        lambda _recording: SimpleNamespace(frames=pd.DataFrame(), contacts=pd.DataFrame()),
    )
    monkeypatch.setattr(
        "writingring.board_event_segmentation.prepare_complete_board_events",
        lambda _frames, _contacts: SimpleNamespace(
            events=empty_events,
            touch_pairs=empty_events,
        ),
    )
    monkeypatch.setattr(
        "writingring.board_event_segmentation.align_board_event_tables",
        lambda events, pairs, *, offset: (events, pairs),
    )
    common = dict(
        data_root=data_root,
        user="writer_a",
        action="letters",
        output_directory=tmp_path / "verification",
        gravity_config=None,
        segmentation_config=SegmentationConfig(),
        panel_duration_s=10.0,
        output_dpi=80,
        overwrite=True,
        input_kind="spike-imu",
        spike_root=spike_root,
        expected_sampling_rate_hz=1_000.0,
        alignment_offset_root=offset_root,
    )

    without_overlay = segment_user_action(
        data_root=data_root,
        user="writer_a",
        action="letters",
        output_root=tmp_path / "segmented-without-overlay",
        input_kind="spike-imu",
        spike_root=spike_root,
        expected_sampling_rate_hz=1_000.0,
        label_overlay_requested=False,
    )
    with_overlay = segment_user_action(
        data_root=data_root,
        user="writer_a",
        action="letters",
        output_root=tmp_path / "segmented-with-overlay",
        input_kind="spike-imu",
        spike_root=spike_root,
        expected_sampling_rate_hz=1_000.0,
        label_overlay_requested=True,
    )
    np.testing.assert_array_equal(without_overlay.raw_imu, with_overlay.raw_imu)
    np.testing.assert_array_equal(
        without_overlay.segment_offsets,
        with_overlay.segment_offsets,
    )
    np.testing.assert_array_equal(
        without_overlay.segment_lengths,
        with_overlay.segment_lengths,
    )
    pd.testing.assert_frame_equal(without_overlay.manifest, with_overlay.manifest)

    segment_ring_imu._write_label_verifications(
        **common,
        overlay_aligned_board_events=False,
    )
    segment_ring_imu._write_label_verifications(
        **common,
        overlay_aligned_board_events=True,
    )

    assert len(captured) == 2
    assert captured[0] == captured[1]


@pytest.mark.parametrize(
    ("field", "replacement", "message"),
    [
        ("alignment_signal_source", "raw-ring", "signal source"),
        ("feature_values_sha256", "0" * 64, "feature values SHA-256"),
        ("timestamp_sha256", "1" * 64, "timestamp SHA-256"),
    ],
)
def test_spike_label_overlay_rejects_mismatched_provenance(
    tmp_path: Path,
    field: str,
    replacement: str,
    message: str,
) -> None:
    data_root = _data_root(tmp_path)
    spike_root, _ = _write_spike_artifact(tmp_path, data_root)
    recording = discover_recordings(data_root)[0]
    feature_input = load_spike_imu_features(
        recording,
        spike_root=spike_root,
        expected_sampling_rate_hz=1_000.0,
    )
    offset_root = tmp_path / "offsets"
    offset = replace(
        _matching_spike_offset(feature_input),
        **{field: replacement},
    )
    write_alignment_offset_txt(
        offset,
        output_path=(
            offset_root / "writer_a" / "action_letters" / "0_ring_board_offset.txt"
        ),
    )

    with pytest.raises(AlignmentOffsetExportError, match=message):
        segment_ring_imu._write_label_verifications(
            data_root=data_root,
            user="writer_a",
            action="letters",
            output_directory=tmp_path / "verification",
            gravity_config=None,
            segmentation_config=SegmentationConfig(),
            panel_duration_s=10.0,
            output_dpi=80,
            overwrite=True,
            overlay_aligned_board_events=True,
            alignment_offset_root=offset_root,
            input_kind="spike-imu",
            spike_root=spike_root,
            expected_sampling_rate_hz=1_000.0,
        )


def test_spike_label_overlay_rejects_legacy_raw_ring_offset(
    tmp_path: Path,
) -> None:
    data_root = _data_root(tmp_path)
    spike_root, _ = _write_spike_artifact(tmp_path, data_root)
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
            matched_event_count=1,
            total_valid_event_count=1,
        ),
        output_path=(
            offset_root / "writer_a" / "action_letters" / "0_ring_board_offset.txt"
        ),
    )

    with pytest.raises(AlignmentOffsetExportError, match="signal source"):
        segment_ring_imu._write_label_verifications(
            data_root=data_root,
            user="writer_a",
            action="letters",
            output_directory=tmp_path / "verification",
            gravity_config=None,
            segmentation_config=SegmentationConfig(),
            panel_duration_s=10.0,
            output_dpi=80,
            overwrite=True,
            overlay_aligned_board_events=True,
            alignment_offset_root=offset_root,
            input_kind="spike-imu",
            spike_root=spike_root,
            expected_sampling_rate_hz=1_000.0,
        )
