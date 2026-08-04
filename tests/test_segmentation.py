from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from writingring.gravity import GravityRemovalConfig
from writingring.segmentation import (
    SegmentLabel,
    SegmentLabelParseError,
    SegmentationConfig,
    SegmentationError,
    SegmentationOutputError,
    load_timestamp_labels,
    segment_recording_by_labels,
    segment_user_action,
)


def _imu(timestamps: list[float] | np.ndarray) -> np.ndarray:
    values = np.asarray(timestamps, dtype=np.float64)
    return np.column_stack([values + channel for channel in range(6)])


def _labels(*items: tuple[float, str]) -> tuple[SegmentLabel, ...]:
    return tuple(
        SegmentLabel(timestamp_us=timestamp, label=label, source_line_number=index)
        for index, (timestamp, label) in enumerate(items, start=1)
    )


def _write_recording(
    action_dir: Path,
    *,
    dataset_id: int,
    timestamps: np.ndarray,
    labels: list[tuple[float, str]],
) -> None:
    imu = _imu(timestamps)
    np.column_stack((imu, timestamps)).astype(np.float64).tofile(
        action_dir / f"{dataset_id}_ring_0.bin"
    )
    (action_dir / f"{dataset_id}_ring_1.bin").write_bytes(b"never read")
    (action_dir / f"{dataset_id}_timestamp.txt").write_text(
        "".join(f"{timestamp:g} {label}\n" for timestamp, label in labels),
        encoding="utf-8",
    )


def test_segments_are_variable_length_and_half_open_at_label_boundaries() -> None:
    timestamps = np.array(
        [1_000_000, 1_200_000, 1_400_000, 1_600_000, 1_800_000],
        dtype=np.float64,
    )

    samples = segment_recording_by_labels(
        ring_imu=_imu(timestamps),
        ring_timestamps_us=timestamps,
        labels=_labels((1_200_000, "a"), (1_600_000, "b")),
    )

    assert [sample.imu.shape for sample in samples] == [(2, 6), (2, 6)]
    assert [sample.sample_count for sample in samples] == [2, 2]
    np.testing.assert_allclose(samples[0].imu[:, 0], [1_200_000, 1_400_000])
    np.testing.assert_allclose(samples[1].imu[:, 0], [1_600_000, 1_800_000])
    assert samples[0].stop_sample_index_exclusive == samples[1].start_sample_index
    assert samples[1].next_label_timestamp_us is None


def test_duplicate_ring_timestamps_belong_to_the_later_boundary_segment() -> None:
    timestamps = np.array(
        [1_000_000, 1_200_000, 1_200_000, 1_200_000, 1_400_000],
        dtype=np.float64,
    )

    samples = segment_recording_by_labels(
        ring_imu=_imu(timestamps),
        ring_timestamps_us=timestamps,
        labels=_labels((1_000_000, "a"), (1_200_000, "b")),
    )

    assert samples[0].sample_count == 1
    assert samples[1].sample_count == 4
    np.testing.assert_allclose(samples[0].imu[:, 0], [1_000_000])
    np.testing.assert_allclose(
        samples[1].imu[:, 0], [1_200_000, 1_200_000, 1_200_000, 1_400_000]
    )


def test_no_padding_or_truncation_preserves_all_segment_samples() -> None:
    short_timestamps = np.arange(100, 520, dtype=np.float64)
    short = segment_recording_by_labels(
        ring_imu=_imu(short_timestamps),
        ring_timestamps_us=short_timestamps,
        labels=_labels((100, "short")),
    )[0]
    long_timestamps = np.arange(100, 850, dtype=np.float64)
    long = segment_recording_by_labels(
        ring_imu=_imu(long_timestamps),
        ring_timestamps_us=long_timestamps,
        labels=_labels((100, "long")),
    )[0]

    assert short.imu.shape == (420, 6)
    assert long.imu.shape == (750, 6)
    np.testing.assert_allclose(short.imu[-1, 0], 519)
    np.testing.assert_allclose(long.imu[-1, 0], 849)


def test_timestamp_label_parser_preserves_case_and_reports_bad_lines(tmp_path: Path) -> None:
    labels_path = tmp_path / "0_timestamp.txt"
    labels_path.write_text("# note\n100 A b\n\n200 z\n", encoding="utf-8")

    labels = load_timestamp_labels(labels_path)

    assert [(label.timestamp_us, label.label, label.source_line_number) for label in labels] == [
        (100.0, "A b", 2),
        (200.0, "z", 4),
    ]
    labels_path.write_text("100 a\n100 b\n", encoding="utf-8")
    with pytest.raises(SegmentLabelParseError, match=r"0_timestamp\.txt:2"):
        load_timestamp_labels(labels_path)
    labels_path.write_text("not-a-timestamp a\n", encoding="utf-8")
    with pytest.raises(SegmentLabelParseError, match=r"0_timestamp\.txt:1"):
        load_timestamp_labels(labels_path)


def test_invalid_empty_segment_and_invalid_config_are_rejected() -> None:
    timestamps = np.array([100, 300], dtype=np.float64)
    with pytest.raises(SegmentationError, match="empty IMU segment"):
        segment_recording_by_labels(
            ring_imu=_imu(timestamps),
            ring_timestamps_us=timestamps,
            labels=_labels((200, "a"), (250, "b")),
            config=SegmentationConfig(minimum_label_interval_us=0),
        )
    with pytest.raises(SegmentationError, match="output_dtype"):
        segment_recording_by_labels(
            ring_imu=_imu(timestamps),
            ring_timestamps_us=timestamps,
            labels=_labels((100, "a")),
            config=SegmentationConfig(output_dtype="int16"),
        )


def test_user_action_aggregation_removes_gravity_before_segmenting(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data_root = tmp_path / "data"
    action_dir = data_root / "user_0" / "0"
    action_dir.mkdir(parents=True)
    _write_recording(
        action_dir,
        dataset_id=10,
        timestamps=10_000_000 + np.arange(700, dtype=np.float64) * 1_000,
        labels=[(10_000_000, "late"), (10_350_000, "end")],
    )
    _write_recording(
        action_dir,
        dataset_id=2,
        timestamps=2_000_000 + np.arange(900, dtype=np.float64) * 1_000,
        labels=[(2_000_000, "first"), (2_200_000, "wrong"), (2_450_000, "next")],
    )

    gravity_methods: list[str] = []

    def fake_gravity(ring: object, *, config: object) -> SimpleNamespace:
        gravity_methods.append(config.gravity_removal_method)
        dataframe = ring.dataframe
        return SimpleNamespace(
            linear_acceleration_body=dataframe[["acc_x", "acc_y", "acc_z"]].to_numpy(),
            angular_velocity_body_rad_s=dataframe[["gyr_x", "gyr_y", "gyr_z"]].to_numpy(),
        )

    monkeypatch.setattr("writingring.segmentation.process_ring_gravity", fake_gravity)
    output_root = tmp_path / "outputs"
    result = segment_user_action(
        data_root=data_root,
        user="user_0",
        action="0",
        output_root=output_root,
    )

    assert result.raw_imu.shape == (1_350, 6)
    assert result.raw_imu.dtype == np.dtype("float32")
    assert result.labels.tolist() == ["first", "next", "late", "end"]
    np.testing.assert_array_equal(result.segment_lengths, [200, 450, 350, 350])
    np.testing.assert_array_equal(result.segment_offsets, [0, 200, 650, 1_000, 1_350])
    np.testing.assert_allclose(
        result.raw_imu[200:650, 0], 2_450_000 + np.arange(450) * 1_000
    )
    assert result.manifest["dataset_id"].tolist() == [2, 2, 10, 10]
    assert result.summary["segment_count"] == 4
    assert result.summary["boundary_mode"] == "label"
    assert result.summary["source_label_count"] == 5
    assert result.summary["skipped_label_counts_by_reason"] == {"label_is_wrong": 1}
    assert result.summary["gravity_removal"]["method"] == "low-pass"
    assert gravity_methods == ["low-pass", "low-pass"]
    assert result.summary["padding_or_truncation"] == "disabled"
    np.testing.assert_array_equal(
        np.load(result.output_paths.segment_offsets_path, allow_pickle=False),
        result.segment_offsets,
    )
    with pytest.raises(SegmentationOutputError, match="already exists"):
        segment_user_action(
            data_root=data_root,
            user="user_0",
            action="0",
            output_root=output_root,
        )
    overwritten = segment_user_action(
        data_root=data_root,
        user="user_0",
        action="0",
        output_root=output_root,
        overwrite=True,
    )
    assert overwritten.segment_lengths.tolist() == [200, 450, 350, 350]


def test_user_action_raw_imu_bypasses_gravity_removal(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    data_root = tmp_path / "data"
    action_dir = data_root / "user_0" / "0"
    action_dir.mkdir(parents=True)
    timestamps = 1_000_000 + np.arange(500, dtype=np.float64) * 1_000
    _write_recording(
        action_dir,
        dataset_id=0,
        timestamps=timestamps,
        labels=[(1_000_000, "a"), (1_250_000, "b")],
    )

    def unexpected_gravity(*args: object, **kwargs: object) -> object:
        raise AssertionError("raw label mode must not remove gravity")

    monkeypatch.setattr("writingring.segmentation.process_ring_gravity", unexpected_gravity)
    result = segment_user_action(
        data_root=data_root,
        user="user_0",
        action="0",
        output_root=tmp_path / "outputs",
        gravity_config=GravityRemovalConfig(gravity_removal_method="raw"),
    )

    np.testing.assert_array_equal(result.raw_imu, _imu(timestamps).astype(np.float32))
    assert result.manifest["gravity_removal_method"].tolist() == ["raw", "raw"]
    assert result.summary["gravity_removal"]["method"] == "raw"
    assert result.summary["gravity_removal"]["sampling_rate_hz"] is None


def test_wrong_label_ends_previous_segment_without_becoming_a_start() -> None:
    timestamps = np.arange(1_000_000, 2_100_000, 10_000, dtype=np.float64)

    samples = segment_recording_by_labels(
        ring_imu=_imu(timestamps),
        ring_timestamps_us=timestamps,
        labels=_labels(
            (1_000_000, "a"),
            (1_200_000, "WrOnG"),
            (1_600_000, "b"),
        ),
    )

    assert [sample.label for sample in samples] == ["a", "b"]
    np.testing.assert_allclose(samples[0].imu[:, 0], np.arange(1_000_000, 1_200_000, 10_000))
    np.testing.assert_allclose(samples[1].imu[:, 0], np.arange(1_600_000, 2_100_000, 10_000))
    assert samples[0].next_label_timestamp_us == 1_200_000


def test_label_pair_less_than_point_one_seconds_invalidates_both_starts() -> None:
    timestamps = np.arange(1_000_000, 2_100_000, 10_000, dtype=np.float64)

    samples = segment_recording_by_labels(
        ring_imu=_imu(timestamps),
        ring_timestamps_us=timestamps,
        labels=_labels(
            (1_000_000, "a"),
            (1_050_000, "too_close"),
            (1_600_000, "b"),
        ),
    )

    assert [sample.label for sample in samples] == ["b"]
    assert samples[0].source_label_index == 2


def test_segment_longer_than_five_seconds_invalidates_its_start() -> None:
    timestamps = np.arange(1_000_000, 8_100_000, 10_000, dtype=np.float64)

    samples = segment_recording_by_labels(
        ring_imu=_imu(timestamps),
        ring_timestamps_us=timestamps,
        labels=_labels((1_000_000, "too_long"), (7_000_000, "restart")),
    )

    assert [sample.label for sample in samples] == ["restart"]
    np.testing.assert_allclose(samples[0].imu[:, 0], np.arange(7_000_000, 8_100_000, 10_000))


def test_final_label_longer_than_five_seconds_is_invalid() -> None:
    timestamps = np.arange(1_000_000, 7_100_000, 10_000, dtype=np.float64)

    with pytest.raises(SegmentationError, match="no valid label starts"):
        segment_recording_by_labels(
            ring_imu=_imu(timestamps),
            ring_timestamps_us=timestamps,
            labels=_labels((1_000_000, "too_long_final")),
        )
