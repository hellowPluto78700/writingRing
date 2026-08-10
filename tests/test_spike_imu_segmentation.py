from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from scripts import align_ring_board, encode_spikes, segment_ring_imu
from writingring.alignment_io import (
    AlignmentOffset,
    AlignmentOffsetExportError,
    build_alignment_input_provenance,
    build_alignment_outcome_paths,
    build_board_chunk_provenance,
    make_alignment_skip_artifact,
    publish_alignment_skip,
    publish_alignment_success,
    build_alignment_time_axes,
    read_alignment_offset_txt,
    write_alignment_offset_txt,
)
from writingring.board_event_segmentation import (
    BoardEventSegmentationConfig,
    BoardEventSegmentationError,
    segment_user_action_by_aligned_board_events,
)
from writingring.discovery import discover_recordings
from writingring.gravity import GravityRemovalConfig
from writingring.imu_preprocessing import PREPROCESSED_IMU_COLUMNS
from writingring.preprocessing_io import PREPROCESSED_IMU_UNITS, sha256_file
from writingring.recording_features import (
    RecordingFeatureError,
    SPIKE_IMU_FEATURE_SCHEMA,
    SPIKE_IMU_TRANSIENT_CHANNEL_NAMES,
    RecordingFeatureInput,
    load_raw_ring_features,
    load_spike_imu_features,
    validate_common_feature_sampling_rate,
)
from writingring.segmentation import (
    SegmentationConfig,
    SegmentationError,
    segment_user_action,
)
from writingring.event_alignment import (
    InitialIntervalNoUsablePairError,
    SequenceAlignmentResult,
)


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


def _add_data_recording(
    data_root: Path,
    *,
    dataset_id: int,
    sample_count: int = 400,
) -> None:
    action_dir = data_root / "writer_a" / "letters"
    timestamps = 1_000_000.0 + np.arange(sample_count, dtype=np.float64) * 1_000.0
    rows = np.column_stack(
        (
            np.zeros((sample_count, 2)),
            np.full(sample_count, 9.80665),
            np.zeros((sample_count, 3)),
            timestamps,
        )
    )
    rows.tofile(action_dir / f"{dataset_id}_ring_0.bin")
    (action_dir / f"{dataset_id}_timestamp.txt").write_text(
        "1000000 first\n1200000 last\n",
        encoding="utf-8",
    )


def _write_spike_artifact(
    tmp_path: Path,
    data_root: Path,
    *,
    sample_count: int = 400,
    dataset_id: int = 0,
    sampling_rate_hz: float = 1_000.0,
) -> tuple[Path, np.ndarray]:
    recording = next(
        recording
        for recording in discover_recordings(data_root)
        if recording.dataset_id == dataset_id
    )
    directory = tmp_path / "spike" / recording.user / recording.action / str(recording.dataset_id)
    directory.mkdir(parents=True)
    values = np.arange(sample_count * 21, dtype=np.float32).reshape(sample_count, 21)
    timestamps = 1_000_000.0 + (
        np.arange(sample_count, dtype=np.float64) * 1_000_000.0 / sampling_rate_hz
    )
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
        "sampling_rate_hz": sampling_rate_hz,
        "timestamps_path": str(timestamps_path.resolve()),
        "timestamps_sha256": sha256_file(timestamps_path),
        "timestamp_unit": "microseconds",
        "input": {
            "sample_count": sample_count,
            "channel_count": len(PREPROCESSED_IMU_COLUMNS),
            "sampling_rate_hz": sampling_rate_hz,
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


def _publish_spike_success_outcome(
    offset_root: Path,
    feature_input: RecordingFeatureInput,
    board_path: Path,
    *,
    offset: AlignmentOffset | None = None,
) -> None:
    """Publish a complete provenance-bearing SUCCESS outcome for a fixture."""

    input_provenance = build_alignment_input_provenance(feature_input)
    board_provenance = build_board_chunk_provenance((board_path,))
    paths = build_alignment_outcome_paths(
        offset_root,
        offset_root.parent / "verification",
        offset_root.parent / "reports",
        user=feature_input.user,
        action=feature_input.action,
        dataset_id=feature_input.dataset_id,
    )
    verification_source = offset_root.parent / f"{feature_input.dataset_id}-alignment-source.png"
    verification_source.write_bytes(b"alignment verification")
    publish_alignment_success(
        paths,
        offset=offset or _matching_spike_offset(feature_input),
        report={
            "recording": {
                "user": feature_input.user,
                "action": feature_input.action,
                "dataset_id": feature_input.dataset_id,
            },
            "input_provenance": input_provenance,
            "board_provenance": board_provenance,
        },
        verification_path=verification_source,
    )


def _publish_spike_skipped_outcome(
    offset_root: Path,
    feature_input: RecordingFeatureInput,
    board_path: Path,
) -> None:
    """Publish one complete provenance-bearing SKIPPED outcome for a fixture."""

    input_provenance = build_alignment_input_provenance(feature_input)
    board_provenance = build_board_chunk_provenance((board_path,))
    paths = build_alignment_outcome_paths(
        offset_root,
        offset_root.parent / "verification",
        offset_root.parent / "reports",
        user=feature_input.user,
        action=feature_input.action,
        dataset_id=feature_input.dataset_id,
    )
    error = InitialIntervalNoUsablePairError(
        previous_global_frame_index=0,
        next_global_frame_index=1,
        previous_timestamp_raw=1_000_000,
        next_timestamp_raw=999_000,
        prefix_boundary_position=1,
        last_pre_jump_global_frame_index=0,
        total_global_valid_pair_count=1,
        usable_prefix_valid_pair_count=0,
    )
    skip = make_alignment_skip_artifact(
        error,
        user=feature_input.user,
        action=feature_input.action,
        dataset_id=feature_input.dataset_id,
        input_provenance=input_provenance,
        board_provenance=board_provenance,
    )
    publish_alignment_skip(
        paths,
        skip_artifact=skip,
        report={
            "recording": {
                "user": feature_input.user,
                "action": feature_input.action,
                "dataset_id": feature_input.dataset_id,
            },
            "input_provenance": input_provenance,
            "board_provenance": board_provenance,
        },
    )


def _prepare_spike_aligned_fixture(
    tmp_path: Path,
    *,
    sampling_rates_hz: dict[int, float],
    skipped_dataset_ids: set[int] | frozenset[int] = frozenset(),
) -> tuple[Path, Path, Path, dict[int, Path]]:
    """Create provenance-complete SpikeIMU outcomes for aligned tests."""

    data_root = _data_root(tmp_path)
    for dataset_id in sorted(sampling_rates_hz):
        if dataset_id != 0:
            _add_data_recording(data_root, dataset_id=dataset_id)
    spike_root = tmp_path / "spike"
    for dataset_id, sampling_rate_hz in sorted(sampling_rates_hz.items()):
        _write_spike_artifact(
            tmp_path,
            data_root,
            dataset_id=dataset_id,
            sampling_rate_hz=sampling_rate_hz,
        )
    features = {
        recording.dataset_id: load_spike_imu_features(
            recording,
            spike_root=spike_root,
        )
        for recording in discover_recordings(data_root)
    }
    offset_root = tmp_path / "offsets"
    board_paths = {
        dataset_id: tmp_path / f"{dataset_id}_board_0.gz"
        for dataset_id in sorted(sampling_rates_hz)
    }
    for dataset_id, board_path in board_paths.items():
        board_path.write_bytes(b"board provenance")
        if dataset_id in skipped_dataset_ids:
            _publish_spike_skipped_outcome(
                offset_root,
                features[dataset_id],
                board_path,
            )
        else:
            _publish_spike_success_outcome(
                offset_root,
                features[dataset_id],
                board_path,
            )
    return data_root, spike_root, offset_root, board_paths


def _patch_spike_aligned_board_loader(
    monkeypatch: pytest.MonkeyPatch,
    board_paths: dict[int, Path],
) -> None:
    """Use a deterministic Board table while validating fixture provenance."""

    frames = pd.DataFrame(
        {
            "global_frame_index": np.arange(6),
            "frame_timestamp_raw": [
                1_000_000.0,
                1_020_000.0,
                1_030_000.0,
                1_040_000.0,
                1_050_000.0,
                1_060_000.0,
            ],
            "chunk_index": 0,
            "contact_count": [0, 1, 1, 1, 0, 0],
        }
    )
    monkeypatch.setattr(
        "writingring.board_event_segmentation.load_board",
        lambda recording: SimpleNamespace(
            frames=frames,
            contacts=pd.DataFrame({"global_frame_index": np.arange(6)}),
            chunk_paths=(board_paths[recording.dataset_id],),
        ),
    )


def test_common_spike_sampling_rate_reports_reference_and_conflict(
    tmp_path: Path,
) -> None:
    data_root = _data_root(tmp_path)
    spike_root, _ = _write_spike_artifact(tmp_path, data_root)
    recording = discover_recordings(data_root)[0]
    reference = load_spike_imu_features(recording, spike_root=spike_root)
    conflict = replace(reference, dataset_id=1, sampling_rate_hz=100.0)

    with pytest.raises(
        RecordingFeatureError,
        match=r"dataset 0 uses 1000 Hz, but dataset 1 uses 100 Hz",
    ):
        validate_common_feature_sampling_rate((reference, conflict))


def test_spike_aligned_skipped_rate_is_excluded_from_requested_check(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data_root, spike_root, offset_root, board_paths = _prepare_spike_aligned_fixture(
        tmp_path,
        sampling_rates_hz={0: 200.0, 1: 100.0},
        skipped_dataset_ids={1},
    )
    _patch_spike_aligned_board_loader(monkeypatch, board_paths)

    result = segment_user_action_by_aligned_board_events(
        data_root=data_root,
        user="writer_a",
        action="letters",
        output_root=tmp_path / "segmented",
        alignment_offset_root=offset_root,
        input_kind="spike-imu",
        spike_root=spike_root,
        expected_sampling_rate_hz=200.0,
        config=BoardEventSegmentationConfig(
            pre_press_context_us=0.0,
            post_lift_context_us=0.0,
        ),
    )

    assert result.summary["processed_recording_count"] == 1
    assert result.summary["skipped_recording_count"] == 1
    assert result.summary["sampling_rate_hz"] == 200.0
    assert result.summary["alignment_outcome_dependency"]["outcomes_by_status"][
        "SKIPPED"
    ][0]["identity"]["dataset_id"] == 1


def test_spike_aligned_success_rate_must_match_requested_rate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data_root, spike_root, offset_root, board_paths = _prepare_spike_aligned_fixture(
        tmp_path,
        sampling_rates_hz={0: 100.0},
    )
    _patch_spike_aligned_board_loader(monkeypatch, board_paths)
    output_root = tmp_path / "segmented"

    with pytest.raises(
        BoardEventSegmentationError,
        match=r"SpikeIMU metadata sampling_rate_hz does not match the requested rate",
    ):
        segment_user_action_by_aligned_board_events(
            data_root=data_root,
            user="writer_a",
            action="letters",
            output_root=output_root,
            alignment_offset_root=offset_root,
            input_kind="spike-imu",
            spike_root=spike_root,
            expected_sampling_rate_hz=200.0,
        )
    assert not output_root.exists()


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


def test_spike_label_aggregation_rejects_mixed_sampling_rates(
    tmp_path: Path,
) -> None:
    data_root = _data_root(tmp_path)
    _add_data_recording(data_root, dataset_id=1)
    spike_root, _ = _write_spike_artifact(
        tmp_path,
        data_root,
        dataset_id=0,
        sampling_rate_hz=200.0,
    )
    _write_spike_artifact(
        tmp_path,
        data_root,
        dataset_id=1,
        sampling_rate_hz=100.0,
    )
    output_root = tmp_path / "mixed-label-output"

    with pytest.raises(
        SegmentationError,
        match=r"dataset 0 uses 200 Hz, but dataset 1 uses 100 Hz",
    ):
        segment_user_action(
            data_root=data_root,
            user="writer_a",
            action="letters",
            output_root=output_root,
            input_kind="spike-imu",
            spike_root=spike_root,
        )
    assert not output_root.exists()


def test_spike_aligned_board_aggregation_rejects_mixed_sampling_rates(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data_root = _data_root(tmp_path)
    _add_data_recording(data_root, dataset_id=1)
    spike_root, _ = _write_spike_artifact(
        tmp_path,
        data_root,
        dataset_id=0,
        sampling_rate_hz=200.0,
    )
    _write_spike_artifact(
        tmp_path,
        data_root,
        dataset_id=1,
        sampling_rate_hz=100.0,
    )
    output_root = tmp_path / "mixed-aligned-output"
    offset_root = tmp_path / "offsets"
    board_paths = {
        dataset_id: tmp_path / f"{dataset_id}_board_0.gz"
        for dataset_id in (0, 1)
    }
    for path in board_paths.values():
        path.write_bytes(b"board provenance")
    features = {
        dataset_id: load_spike_imu_features(
            next(
                recording
                for recording in discover_recordings(data_root)
                if recording.dataset_id == dataset_id
            ),
            spike_root=spike_root,
        )
        for dataset_id in (0, 1)
    }
    for dataset_id in (0, 1):
        _publish_spike_success_outcome(
            offset_root,
            features[dataset_id],
            board_paths[dataset_id],
        )
    frames = pd.DataFrame(
        {
            "global_frame_index": np.arange(4),
            "frame_timestamp_raw": [1_000_000.0, 1_001_000.0, 1_002_000.0, 1_003_000.0],
            "chunk_index": 0,
            "contact_count": [0, 0, 0, 0],
        }
    )
    monkeypatch.setattr(
        "writingring.board_event_segmentation.load_board",
        lambda recording: SimpleNamespace(
            frames=frames,
            contacts=pd.DataFrame({"global_frame_index": np.arange(4)}),
            chunk_paths=(board_paths[recording.dataset_id],),
        ),
    )

    with pytest.raises(
        BoardEventSegmentationError,
        match=r"dataset 0 uses 200 Hz, but dataset 1 uses 100 Hz",
    ):
        segment_user_action_by_aligned_board_events(
            data_root=data_root,
            user="writer_a",
            action="letters",
            output_root=output_root,
            alignment_offset_root=offset_root,
            input_kind="spike-imu",
            spike_root=spike_root,
        )
    assert not output_root.exists()
    assert not any(tmp_path.glob("mixed-aligned-output*"))


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
    monkeypatch: pytest.MonkeyPatch,
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

    board_path = tmp_path / "0_board_0.gz"
    board_path.write_bytes(b"board provenance")
    frames = pd.DataFrame(
        {
            "global_frame_index": np.arange(4),
            "frame_timestamp_raw": [1_000_000.0, 1_001_000.0, 1_002_000.0, 1_003_000.0],
            "chunk_index": 0,
            "contact_count": [0, 0, 0, 0],
        }
    )
    monkeypatch.setattr(
        "writingring.board_event_segmentation.load_board",
        lambda recording: SimpleNamespace(
            frames=frames,
            contacts=pd.DataFrame({"global_frame_index": np.arange(4)}),
            chunk_paths=(board_path,),
        ),
    )

    assert segment_ring_imu.main(
        [*common[:-2], "--output-root", str(tmp_path / "aligned-cli-output"),
         "--boundary-mode", "aligned-board-events",
         "--alignment-offset-root", str(tmp_path / "offsets")]
    ) == 2
    assert "alignment outcome validation failed" in capsys.readouterr().err


def test_spike_board_assist_publishes_21_channels_and_board_targets(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data_root = _data_root(tmp_path)
    spike_root, source_values = _write_spike_artifact(tmp_path, data_root)
    recording = discover_recordings(data_root)[0]
    feature_input = load_spike_imu_features(
        recording,
        spike_root=spike_root,
        expected_sampling_rate_hz=1_000.0,
    )
    offset_root = tmp_path / "offsets"
    board_path = tmp_path / "0_board_0.gz"
    board_path.write_bytes(b"board provenance")
    frames = pd.DataFrame(
        {
            "global_frame_index": np.arange(6),
            "frame_timestamp_raw": [
                1_000_000.0,
                1_020_000.0,
                1_030_000.0,
                1_040_000.0,
                1_050_000.0,
                1_060_000.0,
            ],
            "chunk_index": 0,
            "contact_count": [0, 1, 1, 1, 0, 0],
        }
    )
    monkeypatch.setattr(
        "writingring.board_event_segmentation.load_board",
        lambda _recording: SimpleNamespace(
            frames=frames,
            contacts=pd.DataFrame({"global_frame_index": np.arange(6)}),
            chunk_paths=(board_path,),
        ),
    )
    _publish_spike_success_outcome(offset_root, feature_input, board_path)

    result = segment_user_action_by_aligned_board_events(
        data_root=data_root,
        user="writer_a",
        action="letters",
        output_root=tmp_path / "segmented",
        alignment_offset_root=offset_root,
        input_kind="spike-imu",
        spike_root=spike_root,
        expected_sampling_rate_hz=1_000.0,
        config=BoardEventSegmentationConfig(
            pre_press_context_us=0.0,
            post_lift_context_us=0.0,
        ),
    )

    assert result.raw_imu.shape == (30, 21)
    assert result.board_event_targets.shape == (30, 4)
    np.testing.assert_array_equal(result.raw_imu, source_values[20:50])
    np.testing.assert_array_equal(result.raw_imu[:, 15:], source_values[20:50, 15:])
    assert result.output_paths.raw_imu_path.name.endswith("_spikeIMU.npy")
    assert result.summary["input_kind"] == "spike-imu"
    assert result.summary["feature_schema"] == SPIKE_IMU_FEATURE_SCHEMA
    assert result.summary["channel_count"] == 21
    assert result.summary["alignment_input_hash_match_verified"] is True
    assert result.summary["board_event_targets_present"] is True
    assert result.summary["source_recording_count"] == 1
    assert result.summary["processed_recording_count"] == 1
    assert result.summary["skipped_recording_count"] == 0
    assert result.summary["feature_values_sha256"] == [feature_input.values_sha256]
    assert result.summary["timestamps_sha256"] == [feature_input.timestamps_sha256]
    assert result.output_paths.board_event_targets_path.is_file()

    cli_output_root = tmp_path / "cli-segmented"
    assert segment_ring_imu.main(
        [
            "--data-root", str(data_root),
            "--user", "writer_a", "--action", "letters",
            "--input-kind", "spike-imu", "--spike-root", str(spike_root),
            "--sampling-rate", "1000",
            "--boundary-mode", "aligned-board-events",
            "--alignment-offset-root", str(offset_root),
            "--pre-press-context-seconds", "0",
            "--post-lift-context-seconds", "0",
            "--output-root", str(cli_output_root),
        ]
    ) == 0
    cli_directory = cli_output_root / "writer_a" / "action_letters"
    np.testing.assert_array_equal(
        np.load(cli_directory / "writer_a_action_letters_spikeIMU.npy", allow_pickle=False),
        source_values[20:50],
    )
    assert np.load(
        cli_directory / "writer_a_action_letters_board_event_targets.npy",
        allow_pickle=False,
    ).shape == (30, 4)
    cli_summary = json.loads(
        (cli_directory / "writer_a_action_letters_segmentation_summary.json")
        .read_text(encoding="utf-8")
    )
    assert cli_summary["alignment_input_hash_match_verified"] is True


def test_spike_board_assist_publishes_alignment_outcome_dependency_for_mixed_states(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data_root = _data_root(tmp_path)
    _add_data_recording(data_root, dataset_id=1)
    spike_root, _ = _write_spike_artifact(tmp_path, data_root, dataset_id=0)
    _write_spike_artifact(tmp_path, data_root, dataset_id=1)
    recordings = discover_recordings(data_root)
    features = {
        recording.dataset_id: load_spike_imu_features(
            recording,
            spike_root=spike_root,
            expected_sampling_rate_hz=1_000.0,
        )
        for recording in recordings
    }
    offset_root = tmp_path / "offsets"
    board_paths = {
        dataset_id: tmp_path / f"{dataset_id}_board_0.gz"
        for dataset_id in (0, 1)
    }
    for path in board_paths.values():
        path.write_bytes(b"board provenance")
    _publish_spike_success_outcome(
        offset_root,
        features[0],
        board_paths[0],
    )
    _publish_spike_skipped_outcome(
        offset_root,
        features[1],
        board_paths[1],
    )
    frames = pd.DataFrame(
        {
            "global_frame_index": np.arange(6),
            "frame_timestamp_raw": [
                1_000_000.0,
                1_020_000.0,
                1_030_000.0,
                1_040_000.0,
                1_050_000.0,
                1_060_000.0,
            ],
            "chunk_index": 0,
            "contact_count": [0, 1, 1, 1, 0, 0],
        }
    )
    monkeypatch.setattr(
        "writingring.board_event_segmentation.load_board",
        lambda recording: SimpleNamespace(
            frames=frames,
            contacts=pd.DataFrame({"global_frame_index": np.arange(6)}),
            chunk_paths=(board_paths[recording.dataset_id],),
        ),
    )

    result = segment_user_action_by_aligned_board_events(
        data_root=data_root,
        user="writer_a",
        action="letters",
        output_root=tmp_path / "segmented",
        alignment_offset_root=offset_root,
        input_kind="spike-imu",
        spike_root=spike_root,
        expected_sampling_rate_hz=1_000.0,
        config=BoardEventSegmentationConfig(
            pre_press_context_us=0.0,
            post_lift_context_us=0.0,
        ),
    )

    dependency = result.summary["alignment_outcome_dependency"]
    assert dependency["source_recording_ids"] == [
        {"user": "writer_a", "action": "letters", "dataset_id": 0},
        {"user": "writer_a", "action": "letters", "dataset_id": 1},
    ]
    assert [
        item["identity"]["dataset_id"]
        for item in dependency["outcomes_by_status"]["SUCCESS"]
    ] == [0]
    skipped = dependency["outcomes_by_status"]["SKIPPED"]
    assert [item["identity"]["dataset_id"] for item in skipped] == [1]
    assert skipped[0]["reason"] == "initial_interval_no_usable_pair"
    for item in dependency["outcomes_by_status"]["SUCCESS"] + skipped:
        identity = item["identity"]
        report_path = build_alignment_outcome_paths(
            offset_root,
            offset_root.parent / "verification",
            offset_root.parent / "reports",
            user=identity["user"],
            action=identity["action"],
            dataset_id=identity["dataset_id"],
        ).report_path
        assert item["report_sha256"] == sha256_file(report_path)
        assert len(item["report_sha256"]) == 64
        assert all(character in "0123456789abcdef" for character in item["report_sha256"])
    assert result.summary["source_recording_count"] == 2
    assert result.summary["processed_recording_count"] == 1
    assert result.summary["skipped_recording_count"] == len(skipped) == 1
    assert result.summary["skipped_segment_count"] == 1


def test_spike_alignment_uses_canonical_timestamps_and_imu_channels(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data_root = _data_root(tmp_path)
    spike_root, _ = _write_spike_artifact(tmp_path, data_root)
    recording = discover_recordings(data_root)[0]
    timestamps_path = (
        spike_root
        / recording.user
        / recording.action
        / str(recording.dataset_id)
        / "0_timestamps_us.npy"
    )
    timestamps = 1_000_000.0 + np.cumsum(
        np.asarray([1_000, 0, 2_000, 1_500] + [1_000] * 396, dtype=np.float64)
    )
    np.save(timestamps_path, timestamps, allow_pickle=False)
    metadata_path = timestamps_path.with_name("metadata.json")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["timestamps_sha256"] = sha256_file(timestamps_path)
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

    captured: dict[str, np.ndarray] = {}
    board_chunk = tmp_path / "0_board_0.gz"
    board_chunk.write_bytes(b"board provenance")
    board = SimpleNamespace(
        frames=pd.DataFrame(),
        contacts=pd.DataFrame(),
        chunk_paths=(board_chunk,),
        chunk_reports=(),
    )
    monkeypatch.setattr("writingring.board_loader.load_board", lambda _recording: board)
    empty_events = pd.DataFrame()
    interval = SimpleNamespace(events=empty_events, touch_pairs=empty_events)
    monkeypatch.setattr(
        "writingring.event_alignment.detect_board_events",
        lambda _frames: SimpleNamespace(events=empty_events, touch_pairs=empty_events),
    )
    monkeypatch.setattr(
        "writingring.event_alignment.select_board_interval_from_presses",
        lambda _frames, _contacts, _detection: interval,
    )
    original_score = __import__(
        "writingring.event_alignment", fromlist=["compute_transient_score_array"]
    ).compute_transient_score_array

    def capture_score(values: np.ndarray) -> np.ndarray:
        captured["transient_values"] = np.asarray(values, dtype=np.float64).copy()
        return original_score(values)

    monkeypatch.setattr(
        "writingring.event_alignment.compute_transient_score_array", capture_score
    )
    monkeypatch.setattr(
        "writingring.event_alignment.detect_transient_peak_regions",
        lambda _score, _timestamps: pd.DataFrame(),
    )
    alignment_result = SequenceAlignmentResult(
        success=True,
        best_offset_us=0.0,
        second_best_offset_us=None,
        candidates=pd.DataFrame(),
        event_matches=pd.DataFrame(
            {
                "matched": [True],
                "matched_peak_index": [0],
            }
        ),
        touch_pair_matches=pd.DataFrame(),
        report={
            "alignment_model": "constant_offset",
            "event_coverage_ratio": 1.0,
            "matched_event_count": 1,
            "total_valid_event_count": 1,
        },
        warnings=(),
    )
    monkeypatch.setattr(
        "writingring.event_alignment.align_events_to_transient_peaks",
        lambda *_args, **_kwargs: alignment_result,
    )
    verification_path = tmp_path / "verification.png"
    monkeypatch.setattr(
        "writingring.alignment_verification.create_alignment_verification_figure",
        lambda **_kwargs: SimpleNamespace(
            output_path=verification_path,
            displayed_start_s=0.0,
            displayed_stop_s=60.0,
            press_count_displayed=0,
            lift_count_displayed=0,
            label_count_displayed=0,
            warnings=(),
        ),
    )

    assert align_ring_board.main(
        [
            "--data-root", str(data_root),
            "--user", "writer_a", "--action", "letters", "--dataset-id", "0",
            "--input-kind", "spike-imu", "--spike-root", str(spike_root),
            "--offset-output-root", str(tmp_path / "offsets"),
            "--verification-output-root", str(tmp_path / "verification"),
            "--report-output-root", str(tmp_path / "reports"),
            "--overwrite-offset", "--overwrite-verification", "--overwrite-report",
        ]
    ) == 0

    feature_input = load_spike_imu_features(recording, spike_root=spike_root)
    np.testing.assert_array_equal(
        captured["transient_values"], feature_input.values[:, 15:21]
    )
    offset = read_alignment_offset_txt(
        tmp_path / "offsets" / "writer_a" / "action_letters" / "0_ring_board_offset.txt",
        expected_user="writer_a",
        expected_action="letters",
        expected_dataset_id=0,
    )
    assert offset.alignment_signal_source == "spike-imu"
    assert offset.feature_values_sha256 == feature_input.values_sha256
    assert offset.feature_metadata_sha256 == feature_input.metadata_sha256
    assert offset.timestamp_sha256 == feature_input.timestamps_sha256
    assert offset.timestamp_source_sha256 == feature_input.timestamps_sha256
    assert offset.timestamp_source_duplicate_step_count == 1
    assert offset.alignment_time_axis_strategy == "endpoint_reconstruction"
    assert offset.feature_sampling_rate_hz == pytest.approx(
        feature_input.sampling_rate_hz
    )
    assert offset.alignment_time_axis_sample_count == len(timestamps)
    assert offset.canonical_timestamps_modified is False
    assert offset.work_axis_offset_us == pytest.approx(0.0)
    assert offset.projection_delta_us == pytest.approx(0.0)
    assert offset.projection_contributing_match_count == 1
    report = json.loads(
        (
            tmp_path / "reports" / "writer_a" / "action_letters"
            / "0_alignment_report.json"
        ).read_text(encoding="utf-8")
    )
    assert report["timestamp_sha256"] == feature_input.timestamps_sha256
    assert report["source_hash_verified"] is True
    assert report["timestamp_source"]["sha256"] == feature_input.timestamps_sha256
    assert report["timestamp_source"]["duplicate_step_count"] == 1
    assert report["feature_sampling_rate_hz"] == pytest.approx(
        feature_input.sampling_rate_hz
    )
    assert report["alignment_time_axis"]["strategy"] == "endpoint_reconstruction"
    assert report["alignment_time_axis"]["canonical_start_us"] == pytest.approx(
        timestamps[0]
    )
    assert report["alignment_time_axis"]["canonical_stop_us"] == pytest.approx(
        timestamps[-1]
    )
    assert report["alignment_time_axis"]["canonical_timestamps_modified"] is False
    assert report["canonical_offset_projection"]["success"] is True
    assert report["canonical_offset_projection"]["exported_offset_us"] == pytest.approx(0.0)


def test_alignment_cli_initial_interval_error_vs_explicit_skip(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data_root = _data_root(tmp_path)
    board_chunk = tmp_path / "0_board_0.gz"
    board_chunk.write_bytes(b"real ordered Board chunk")
    board = SimpleNamespace(
        frames=pd.DataFrame(),
        contacts=pd.DataFrame(),
        chunk_paths=(board_chunk,),
        chunk_reports=(),
    )
    error = InitialIntervalNoUsablePairError(
        previous_global_frame_index=10,
        next_global_frame_index=11,
        previous_timestamp_raw=1_000,
        next_timestamp_raw=900,
        prefix_boundary_position=11,
        last_pre_jump_global_frame_index=10,
        total_global_valid_pair_count=1,
        usable_prefix_valid_pair_count=0,
    )
    empty_events = pd.DataFrame()
    monkeypatch.setattr("writingring.board_loader.load_board", lambda _recording: board)
    monkeypatch.setattr(
        "writingring.event_alignment.detect_board_events",
        lambda _frames: SimpleNamespace(events=empty_events, touch_pairs=empty_events),
    )
    monkeypatch.setattr(
        "writingring.event_alignment.select_board_interval_from_presses",
        lambda _frames, _contacts, _detection: (_ for _ in ()).throw(error),
    )
    common = [
        "--data-root", str(data_root),
        "--user", "writer_a", "--action", "letters", "--dataset-id", "0",
        "--offset-output-root", str(tmp_path / "offsets"),
        "--verification-output-root", str(tmp_path / "verification"),
        "--report-output-root", str(tmp_path / "reports"),
    ]
    assert align_ring_board.main([*common, "--initial-interval-policy", "error"]) == 2
    skip_path = (
        tmp_path / "offsets" / "writer_a" / "action_letters"
        / "0_ring_board_skip.json"
    )
    assert not skip_path.exists()

    assert align_ring_board.main([*common, "--initial-interval-policy", "skip"]) == 0
    assert skip_path.is_file()
    assert (
        tmp_path / "reports" / "writer_a" / "action_letters"
        / "0_alignment_report.json"
    ).is_file()


def _patch_failed_alignment_cli(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    failed_confidence_checks: object,
) -> tuple[list[str], Path, Path, Path, Path]:
    """Patch the CLI up to one structured, unsuccessful alignment result."""

    data_root = _data_root(tmp_path)
    board_chunk = tmp_path / "0_board_0.gz"
    board_chunk.write_bytes(b"real ordered Board chunk")
    board = SimpleNamespace(
        frames=pd.DataFrame(),
        contacts=pd.DataFrame(),
        chunk_paths=(board_chunk,),
        chunk_reports=(),
    )
    empty_events = pd.DataFrame()
    monkeypatch.setattr("writingring.board_loader.load_board", lambda _recording: board)
    monkeypatch.setattr(
        "writingring.event_alignment.detect_board_events",
        lambda _frames: SimpleNamespace(events=empty_events, touch_pairs=empty_events),
    )
    monkeypatch.setattr(
        "writingring.event_alignment.select_board_interval_from_presses",
        lambda _frames, _contacts, _detection: SimpleNamespace(
            events=empty_events,
            touch_pairs=empty_events,
        ),
    )
    monkeypatch.setattr(
        "writingring.event_alignment.detect_transient_peak_regions",
        lambda _score, _timestamps: pd.DataFrame(),
    )
    total_valid_touch_pair_count = (
        3
        if isinstance(failed_confidence_checks, list)
        and "minimum_valid_touch_pairs" in failed_confidence_checks
        else 5
    )
    monkeypatch.setattr(
        "writingring.event_alignment.align_events_to_transient_peaks",
        lambda *_args, **_kwargs: SequenceAlignmentResult(
            success=False,
            best_offset_us=123.0,
            second_best_offset_us=456.0,
            candidates=pd.DataFrame(),
            event_matches=pd.DataFrame(),
            touch_pair_matches=pd.DataFrame(),
            report={
                "alignment_model": "constant_offset",
                "failed_confidence_checks": failed_confidence_checks,
                "confidence_checks": {
                    "minimum_valid_touch_pairs": {
                        "passed": total_valid_touch_pair_count >= 5,
                        "actual": total_valid_touch_pair_count,
                        "minimum": 5,
                    },
                    "minimum_event_coverage_ratio": {
                        "passed": False,
                        "actual": 0.1,
                        "minimum": 0.4,
                    },
                },
                "matched_event_count": 1,
                "total_valid_event_count": 10,
                "event_coverage_ratio": 0.1,
                "matched_press_count": 1,
                "total_valid_press_count": 5,
                "press_coverage_ratio": 0.2,
                "matched_lift_count": 0,
                "total_valid_lift_count": 5,
                "lift_coverage_ratio": 0.0,
                "fully_matched_touch_pair_count": 0,
                "total_valid_touch_pair_count": total_valid_touch_pair_count,
                "best_offset_us": 123.0,
                "best_vs_second_best_nearly_tied": False,
            },
            warnings=("low confidence",),
        ),
    )
    offset_root = tmp_path / "offsets"
    verification_root = tmp_path / "verification"
    report_root = tmp_path / "reports"
    common = [
        "--data-root", str(data_root),
        "--user", "writer_a", "--action", "letters", "--dataset-id", "0",
        "--offset-output-root", str(offset_root),
        "--verification-output-root", str(verification_root),
        "--report-output-root", str(report_root),
    ]
    skip_path = offset_root / "writer_a" / "action_letters" / "0_ring_board_skip.json"
    offset_path = offset_root / "writer_a" / "action_letters" / "0_ring_board_offset.txt"
    verification_path = (
        verification_root / "writer_a" / "action_letters"
        / "0_alignment_verification.png"
    )
    report_path = report_root / "writer_a" / "action_letters" / "0_alignment_report.json"
    return common, skip_path, offset_path, verification_path, report_path


@pytest.mark.parametrize(
    ("failed_confidence_checks", "expected_reason"),
    [
        (["minimum_event_coverage_ratio"], "insufficient_event_coverage"),
        (["minimum_valid_touch_pairs"], "insufficient_valid_touch_pairs"),
    ],
)
def test_alignment_cli_unalignable_policy_default_and_explicit_skip(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failed_confidence_checks: list[str],
    expected_reason: str,
) -> None:
    common, skip_path, offset_path, verification_path, report_path = (
        _patch_failed_alignment_cli(
            tmp_path,
            monkeypatch,
            failed_confidence_checks=failed_confidence_checks,
        )
    )

    # Strict-by-default keeps an unsuccessful result FAILED and report-only.
    assert align_ring_board.main(common) == 2
    failed_report = json.loads(report_path.read_text(encoding="utf-8"))
    assert failed_report["alignment_status"] == "failed"
    assert not skip_path.exists()
    assert not offset_path.exists()
    assert not verification_path.exists()

    # A new explicit opt-in publishes only the validated SKIPPED artifacts.
    common.append("--unalignable-recording-policy")
    common.append("skip")
    common.extend(["--overwrite-report", "--overwrite-outcome"])
    assert align_ring_board.main(common) == 0
    skip = json.loads(skip_path.read_text(encoding="utf-8"))
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert skip["reason"] == expected_reason
    assert skip["diagnostics"]["minimum_valid_touch_pairs"] == 5
    assert skip["diagnostics"]["minimum_event_coverage_ratio"] == pytest.approx(0.4)
    assert report["alignment_status"] == "skipped"
    assert not offset_path.exists()
    assert not verification_path.exists()


@pytest.mark.parametrize(
    "failed_confidence_checks",
    [
        None,
        ["unexpected_check"],
        ["minimum_event_coverage_ratio", "minimum_event_coverage_ratio"],
        ["minimum_event_coverage_ratio", "minimum_valid_touch_pairs"],
    ],
)
def test_alignment_cli_unalignable_policy_rejects_malformed_or_unexpected_checks(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failed_confidence_checks: object,
) -> None:
    common, skip_path, offset_path, verification_path, report_path = (
        _patch_failed_alignment_cli(
            tmp_path,
            monkeypatch,
            failed_confidence_checks=failed_confidence_checks,
        )
    )
    common.extend(["--unalignable-recording-policy", "skip"])

    assert align_ring_board.main(common) == 2
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["alignment_status"] == "failed"
    assert not skip_path.exists()
    assert not offset_path.exists()
    assert not verification_path.exists()


def test_alignment_cli_new_policy_also_controls_legacy_initial_skip(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data_root = _data_root(tmp_path)
    board_chunk = tmp_path / "0_board_0.gz"
    board_chunk.write_bytes(b"real ordered Board chunk")
    error = InitialIntervalNoUsablePairError(
        previous_global_frame_index=10,
        next_global_frame_index=11,
        previous_timestamp_raw=1_000,
        next_timestamp_raw=900,
        prefix_boundary_position=11,
        last_pre_jump_global_frame_index=10,
        total_global_valid_pair_count=1,
        usable_prefix_valid_pair_count=0,
    )
    empty_events = pd.DataFrame()
    monkeypatch.setattr(
        "writingring.board_loader.load_board",
        lambda _recording: SimpleNamespace(
            frames=pd.DataFrame(),
            contacts=pd.DataFrame(),
            chunk_paths=(board_chunk,),
            chunk_reports=(),
        ),
    )
    monkeypatch.setattr(
        "writingring.event_alignment.detect_board_events",
        lambda _frames: SimpleNamespace(events=empty_events, touch_pairs=empty_events),
    )
    monkeypatch.setattr(
        "writingring.event_alignment.select_board_interval_from_presses",
        lambda _frames, _contacts, _detection: (_ for _ in ()).throw(error),
    )
    common = [
        "--data-root", str(data_root),
        "--user", "writer_a", "--action", "letters", "--dataset-id", "0",
        "--offset-output-root", str(tmp_path / "offsets"),
        "--verification-output-root", str(tmp_path / "verification"),
        "--report-output-root", str(tmp_path / "reports"),
        "--unalignable-recording-policy", "skip",
    ]
    assert align_ring_board.main(common) == 0
    assert (
        tmp_path / "offsets" / "writer_a" / "action_letters"
        / "0_ring_board_skip.json"
    ).is_file()


@pytest.mark.skipif(
    not (Path("data") / "user_1" / "4" / "2_ring_0.bin").is_file(),
    reason="repository sample data is unavailable",
)
def test_real_raw_alignment_accepts_duplicate_ring_timestamps(tmp_path: Path) -> None:
    report_root = tmp_path / "reports"
    offset_root = tmp_path / "offsets"
    verification_root = tmp_path / "verification"
    assert align_ring_board.main(
        [
            "--data-root", "data",
            "--user", "user_1", "--action", "4", "--dataset-id", "2",
            "--offset-output-root", str(offset_root),
            "--verification-output-root", str(verification_root),
            "--report-output-root", str(report_root),
            "--overwrite-offset", "--overwrite-verification", "--overwrite-report",
        ]
    ) == 0
    report = json.loads(
        (report_root / "user_1" / "action_4" / "2_alignment_report.json")
        .read_text(encoding="utf-8")
    )
    assert report["timestamp_source"]["duplicate_step_count"] > 0
    assert report["alignment_time_axis"]["strategy"] == "endpoint_reconstruction"
    assert report["alignment_time_axis"]["canonical_timestamps_modified"] is False
    offset = read_alignment_offset_txt(
        offset_root / "user_1" / "action_4" / "2_ring_board_offset.txt",
        expected_user="user_1",
        expected_action="4",
        expected_dataset_id=2,
    )
    assert offset.timestamp_source_duplicate_step_count > 0
    assert offset.canonical_timestamps_modified is False
    assert offset.work_axis_offset_us is not None
    assert offset.offset_domain == "alignment_work_axis"
    assert offset.offset_us == pytest.approx(offset.work_axis_offset_us)
    assert offset.canonical_offset_us is None
    assert offset.canonical_offset_projection_success is False
    assert report["work_axis_alignment_success"] is True
    assert report["canonical_offset_projection_success"] is False
    assert report["alignment_success"] is True
    assert report["canonical_offset_projection"]["representable"] is False


@pytest.mark.skipif(
    not (
        (Path("data") / "user_0" / "0" / "0_ring_0.bin").is_file()
        and (
            Path("outputs")
            / "action0_pipeline"
            / "raw"
            / "aligned-board-events"
            / "spikeEncoding"
            / "custom-wavelet"
            / "user_0"
            / "0"
            / "0"
            / "spikeIMU.npy"
        ).is_file()
    ),
    reason="real user_0/action_0/dataset_0 SpikeIMU artifact is unavailable",
)
def test_real_raw_and_spike_alignment_are_representation_invariant(
    tmp_path: Path,
) -> None:
    data_root = Path("data")
    spike_root = (
        Path("outputs")
        / "action0_pipeline"
        / "raw"
        / "aligned-board-events"
        / "spikeEncoding"
        / "custom-wavelet"
    )
    recording = next(
        item
        for item in discover_recordings(data_root)
        if item.user == "user_0" and item.action == "0" and item.dataset_id == 0
    )
    raw_features = load_raw_ring_features(
        recording,
        gravity_config=GravityRemovalConfig(
            gravity_removal_method="raw",
            strict_calibration=False,
        ),
    )
    spike_features = load_spike_imu_features(recording, spike_root=spike_root)
    np.testing.assert_array_equal(
        raw_features.timestamps_us,
        spike_features.timestamps_us,
    )
    np.testing.assert_allclose(
        raw_features.values[:, 3:9],
        spike_features.values[:, 15:21],
        rtol=0.0,
        atol=1e-6,
    )
    raw_axes = build_alignment_time_axes(
        raw_features.timestamps_us,
        input_kind="raw-ring",
        sampling_rate_hz=raw_features.sampling_rate_hz,
    )
    spike_axes = build_alignment_time_axes(
        spike_features.timestamps_us,
        input_kind="spike-imu",
        sampling_rate_hz=spike_features.sampling_rate_hz,
    )
    np.testing.assert_array_equal(
        raw_axes.work_timestamps_us,
        spike_axes.work_timestamps_us,
    )

    reports: dict[str, dict[str, object]] = {}
    offsets: dict[str, AlignmentOffset] = {}
    for input_kind in ("raw-ring", "spike-imu"):
        output_root = tmp_path / input_kind
        arguments = [
            "--data-root", str(data_root),
            "--user", "user_0", "--action", "0", "--dataset-id", "0",
            "--input-kind", input_kind,
            "--offset-output-root", str(output_root / "offsets"),
            "--report-output-root", str(output_root / "reports"),
            "--verification-output-root", str(output_root / "verification"),
            "--overwrite-offset", "--overwrite-report", "--overwrite-verification",
        ]
        if input_kind == "spike-imu":
            arguments.extend(["--spike-root", str(spike_root)])
        assert align_ring_board.main(arguments) == 0
        report_path = (
            output_root / "reports" / "user_0" / "action_0" / "0_alignment_report.json"
        )
        reports[input_kind] = json.loads(report_path.read_text(encoding="utf-8"))
        offsets[input_kind] = read_alignment_offset_txt(
            output_root / "offsets" / "user_0" / "action_0" / "0_ring_board_offset.txt",
            expected_user="user_0",
            expected_action="0",
            expected_dataset_id=0,
        )

    raw_report = reports["raw-ring"]
    spike_report = reports["spike-imu"]
    for key in (
        "work_axis_alignment_success",
        "alignment_success",
        "canonical_offset_projection_success",
        "matched_event_count",
        "fully_matched_touch_pair_count",
    ):
        assert raw_report[key] == spike_report[key]
    for key in (
        "event_coverage_ratio",
        "median_normalized_peak_distance",
        "work_axis_offset_us",
    ):
        assert raw_report[key] == pytest.approx(spike_report[key])
    assert raw_report["alignment_time_axis"] == spike_report["alignment_time_axis"]
    assert raw_report["feature_sampling_rate_hz"] == pytest.approx(
        raw_features.sampling_rate_hz
    )
    assert spike_report["feature_sampling_rate_hz"] == pytest.approx(
        spike_features.sampling_rate_hz
    )
    assert offsets["raw-ring"].work_axis_offset_us == pytest.approx(
        offsets["spike-imu"].work_axis_offset_us
    )
    assert offsets["raw-ring"].offset_us == pytest.approx(
        offsets["spike-imu"].offset_us
    )
    assert offsets["raw-ring"].offset_domain == "alignment_work_axis"
    assert offsets["raw-ring"].canonical_offset_us is None
    assert offsets["raw-ring"].canonical_offset_projection_success is False
    assert offsets["raw-ring"].offset_us == pytest.approx(84_340.875)
    assert offsets["raw-ring"].work_axis_offset_us == pytest.approx(84_340.875)


@pytest.mark.parametrize(
    ("field", "replacement", "message"),
    [
        ("alignment_signal_source", "raw-ring", "alignment outcome validation"),
        ("feature_values_sha256", "0" * 64, "alignment outcome validation"),
        ("timestamp_sha256", "1" * 64, "alignment outcome validation"),
    ],
)
def test_spike_board_assist_rejects_stale_alignment_provenance(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
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
    board_path = tmp_path / "0_board_0.gz"
    board_path.write_bytes(b"board provenance")
    frames = pd.DataFrame(
        {
            "global_frame_index": np.arange(4),
            "frame_timestamp_raw": [1_000_000.0, 1_001_000.0, 1_002_000.0, 1_003_000.0],
            "chunk_index": 0,
            "contact_count": [0, 0, 0, 0],
        }
    )
    monkeypatch.setattr(
        "writingring.board_event_segmentation.load_board",
        lambda _recording: SimpleNamespace(
            frames=frames,
            contacts=pd.DataFrame({"global_frame_index": np.arange(4)}),
            chunk_paths=(board_path,),
        ),
    )
    _publish_spike_success_outcome(
        offset_root,
        feature_input,
        board_path,
        offset=replace(_matching_spike_offset(feature_input), **{field: replacement}),
    )

    with pytest.raises(BoardEventSegmentationError, match=message):
        segment_user_action_by_aligned_board_events(
            data_root=data_root,
            user="writer_a",
            action="letters",
            output_root=tmp_path / "segmented",
            alignment_offset_root=offset_root,
            input_kind="spike-imu",
            spike_root=spike_root,
            expected_sampling_rate_hz=1_000.0,
        )
    assert not (tmp_path / "segmented" / "writer_a" / "action_letters").exists()


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
