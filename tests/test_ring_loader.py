from pathlib import Path

import numpy as np
import pytest

from writingring.discovery import Recording, discover_recordings
from writingring.ring_loader import (
    RELATIVE_TIME_COLUMN,
    RING_COLUMNS,
    EmptyRingFileError,
    MalformedRingFileError,
    RingPathError,
    load_ring,
)


def _write_ring(path: Path, rows: list[list[float]] | np.ndarray) -> Path:
    np.asarray(rows, dtype=np.float64).tofile(path)
    return path


def _row(timestamp: float, start: float = 0.0) -> list[float]:
    return [start + offset for offset in range(6)] + [timestamp]


def test_loads_valid_float64_rows_and_confirmed_columns(tmp_path: Path) -> None:
    path = _write_ring(
        tmp_path / "0_ring_0.bin",
        [_row(1_000_000.0), _row(1_005_000.0, start=10.0)],
    )

    result = load_ring(path)

    assert result.source_path == path
    assert result.raw_shape == (2, 7)
    assert list(result.dataframe.columns[:7]) == list(RING_COLUMNS)
    assert result.dataframe.index.name == "sample_index"
    assert result.validation.sample_count == 2
    assert result.validation.value_count == 14
    assert result.validation.file_size_bytes == 14 * 8
    assert result.validation.expected_sample_count_from_file_size == 2
    assert result.validation.raw_columns_present


def test_original_timestamp_values_are_preserved(tmp_path: Path) -> None:
    timestamps = np.array([1_720_000_000_000_001.0, 1_720_000_000_005_001.0])
    path = _write_ring(
        tmp_path / "0_ring_0.bin",
        [_row(timestamps[0]), _row(timestamps[1])],
    )

    result = load_ring(path)

    np.testing.assert_array_equal(
        result.dataframe["timestamp"].to_numpy(),
        timestamps,
    )


def test_relative_time_is_explicitly_labeled_as_inferred(
    tmp_path: Path,
) -> None:
    path = _write_ring(
        tmp_path / "0_ring_0.bin",
        [_row(10_000_000.0), _row(10_005_000.0)],
    )

    result = load_ring(path)

    assert RELATIVE_TIME_COLUMN in result.dataframe
    np.testing.assert_allclose(
        result.dataframe[RELATIVE_TIME_COLUMN],
        [0.0, 0.005],
    )
    metadata = result.timestamp_interpretation
    assert not metadata.raw_timestamp_unit_confirmed
    assert "inferred" in metadata.inferred_timestamp_unit
    assert metadata.relative_time_column == RELATIVE_TIME_COLUMN
    assert not metadata.synthetic_time_axis_used
    assert metadata.nominal_sampling_rate_hz is None


def test_empty_file_raises_custom_exception(tmp_path: Path) -> None:
    path = tmp_path / "0_ring_0.bin"
    path.touch()

    with pytest.raises(EmptyRingFileError, match="empty"):
        load_ring(path)


def test_partial_float64_value_raises_custom_exception(tmp_path: Path) -> None:
    path = tmp_path / "0_ring_0.bin"
    path.write_bytes(b"\x00")

    with pytest.raises(MalformedRingFileError, match="byte length"):
        load_ring(path)


def test_value_count_not_divisible_by_seven_raises(tmp_path: Path) -> None:
    path = tmp_path / "0_ring_0.bin"
    np.arange(8, dtype=np.float64).tofile(path)

    with pytest.raises(MalformedRingFileError, match="not divisible by 7"):
        load_ring(path)


def test_nan_and_infinities_are_reported_by_column(tmp_path: Path) -> None:
    rows = np.array(
        [
            [np.nan, np.inf, -np.inf, 1.0, 2.0, 3.0, 100.0],
            [4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 200.0],
        ],
        dtype=np.float64,
    )
    path = _write_ring(tmp_path / "0_ring_0.bin", rows)

    report = load_ring(path).validation

    assert report.nan_counts_by_column["acc_x"] == 1
    assert report.positive_infinity_counts_by_column["acc_y"] == 1
    assert report.negative_infinity_counts_by_column["acc_z"] == 1
    assert report.per_channel_statistics["acc_x"].finite_count == 1
    assert report.per_channel_statistics["acc_x"].mean == 4.0
    assert any("non-finite values in acc_x" in warning for warning in report.warnings)


def test_strictly_increasing_timestamps(tmp_path: Path) -> None:
    path = _write_ring(
        tmp_path / "0_ring_0.bin",
        [_row(1_000_000.0), _row(1_005_000.0), _row(1_010_000.0)],
    )

    report = load_ring(path).validation

    assert report.timestamps_finite
    assert report.timestamps_nondecreasing
    assert report.timestamps_strictly_increasing
    assert report.duplicate_timestamp_steps == 0
    assert report.backward_timestamp_steps == 0
    assert report.inferred_duration_s == pytest.approx(0.01)
    assert report.inferred_sampling_rate_hz == pytest.approx(200.0)


def test_duplicate_timestamps_are_preserved_and_warned(tmp_path: Path) -> None:
    path = _write_ring(
        tmp_path / "0_ring_0.bin",
        [_row(1_000_000.0), _row(1_000_000.0), _row(1_010_000.0)],
    )

    result = load_ring(path)
    report = result.validation

    assert list(result.dataframe["timestamp"]) == [
        1_000_000.0,
        1_000_000.0,
        1_010_000.0,
    ]
    assert report.timestamps_nondecreasing
    assert not report.timestamps_strictly_increasing
    assert report.duplicate_timestamp_steps == 1
    assert report.inferred_sampling_rate_hz == pytest.approx(200.0)
    assert any("duplicate timestamp" in warning for warning in result.warnings)


def test_backward_timestamps_are_reported_without_repair(
    tmp_path: Path,
) -> None:
    timestamps = [1_000_000.0, 1_010_000.0, 1_005_000.0]
    path = _write_ring(
        tmp_path / "0_ring_0.bin",
        [_row(timestamp) for timestamp in timestamps],
    )

    result = load_ring(path)
    report = result.validation

    assert list(result.dataframe["timestamp"]) == timestamps
    assert not report.timestamps_nondecreasing
    assert not report.timestamps_strictly_increasing
    assert report.backward_timestamp_steps == 1
    assert report.backward_timestamp_step_positions == (2,)
    assert report.inferred_duration_s is None
    assert report.inferred_sampling_rate_hz is None
    assert any("backward timestamp" in warning for warning in result.warnings)


def test_single_sample_has_no_sampling_rate(tmp_path: Path) -> None:
    path = _write_ring(tmp_path / "0_ring_0.bin", [_row(1_000_000.0)])

    result = load_ring(path)

    assert result.validation.sample_count == 1
    assert result.validation.inferred_duration_s is None
    assert result.validation.inferred_sampling_rate_hz is None
    assert result.validation.timestamps_strictly_increasing
    assert any("fewer than two samples" in warning for warning in result.warnings)


def test_zero_duration_has_no_sampling_rate(tmp_path: Path) -> None:
    path = _write_ring(
        tmp_path / "0_ring_0.bin",
        [_row(1_000_000.0), _row(1_000_000.0)],
    )

    report = load_ring(path).validation

    assert report.raw_timestamp_duration == 0.0
    assert report.inferred_duration_s is None
    assert report.inferred_sampling_rate_hz is None


def test_nonfinite_timestamps_disable_interpretation(tmp_path: Path) -> None:
    path = _write_ring(
        tmp_path / "0_ring_0.bin",
        [_row(1_000_000.0), _row(np.nan)],
    )

    result = load_ring(path)

    assert not result.validation.timestamps_finite
    assert result.validation.inferred_duration_s is None
    assert result.validation.inferred_sampling_rate_hz is None
    assert RELATIVE_TIME_COLUMN not in result.dataframe
    assert result.timestamp_interpretation.relative_time_column is None


def test_missing_file_raises_custom_exception(tmp_path: Path) -> None:
    path = tmp_path / "0_ring_0.bin"

    with pytest.raises(RingPathError, match="does not exist"):
        load_ring(path)


def test_directory_path_raises_custom_exception(tmp_path: Path) -> None:
    path = tmp_path / "0_ring_0.bin"
    path.mkdir()

    with pytest.raises(RingPathError, match="not a regular file"):
        load_ring(path)


def test_direct_ring_1_path_is_rejected_before_read(tmp_path: Path) -> None:
    path = _write_ring(tmp_path / "0_ring_1.bin", [_row(1_000_000.0)])

    with pytest.raises(RingPathError, match=r"only \*_ring_0\.bin"):
        load_ring(path)


def test_recording_loads_only_primary_ring_0(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    action = tmp_path / "user_0" / "0"
    action.mkdir(parents=True)
    ring_0 = _write_ring(action / "0_ring_0.bin", [_row(1_000_000.0)])
    _write_ring(action / "0_ring_1.bin", [_row(9_000_000.0)])
    recording = discover_recordings(tmp_path)[0]
    opened_paths: list[Path] = []
    original_fromfile = np.fromfile

    def tracked_fromfile(
        path: str | Path,
        dtype: type[np.float64],
    ) -> np.ndarray:
        opened_paths.append(Path(path))
        return original_fromfile(path, dtype=dtype)

    monkeypatch.setattr(
        "writingring.ring_loader.np.fromfile",
        tracked_fromfile,
    )

    result = load_ring(recording)

    assert result.source_path == ring_0
    assert opened_paths == [ring_0]
    assert result.dataframe["timestamp"].iloc[0] == 1_000_000.0


def test_recording_type_is_supported_directly(tmp_path: Path) -> None:
    path = _write_ring(tmp_path / "0_ring_0.bin", [_row(1_000_000.0)])
    recording = Recording(
        user="user_0",
        action="0",
        dataset_id=0,
        ring_0_path=path,
        ring_1_path=None,
        timestamp_path=None,
        board_chunk_paths=(),
        missing_chunk_indices=(),
        warnings=(),
    )

    assert load_ring(recording).source_path == path

