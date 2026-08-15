from __future__ import annotations

import json
import hashlib
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

from writingring.segment_padding import (
    SegmentPaddingError,
    analyze_segment_lengths,
    build_padding_package,
    collect_segment_length_records,
    publish_padded_root,
    resolve_target_length,
    validate_segmented_root,
    write_segment_length_analysis,
)


def _write_package(
    root: Path,
    *,
    user: str,
    action: str,
    lengths: list[int],
    with_targets: bool = False,
    channel_count: int = 21,
) -> Path:
    directory = root / user / f"action_{action}"
    directory.mkdir(parents=True)
    stem = f"{user}_action_{action}"
    offsets = np.concatenate(([0], np.cumsum(lengths))).astype(np.int64)
    spike_imu = np.arange(
        int(offsets[-1]) * channel_count, dtype=np.float32
    ).reshape(-1, channel_count)
    np.save(directory / f"{stem}_spikeIMU.npy", spike_imu, allow_pickle=False)
    np.save(directory / f"{stem}_labels.npy", np.asarray([f"L{i}" for i in range(len(lengths))]), allow_pickle=False)
    np.save(directory / f"{stem}_segment_offsets.npy", offsets, allow_pickle=False)
    np.save(directory / f"{stem}_segment_lengths.npy", np.asarray(lengths, dtype=np.int32), allow_pickle=False)
    (directory / f"{stem}_segmentation_summary.json").write_text(
        json.dumps({
            "input_kind": "spike-imu",
            "feature_schema": "signed_wavelet_events_plus_imu_v1",
            "channel_count": channel_count,
            "channel_names": [f"channel_{index}" for index in range(channel_count)],
            "units": ["unit"] * channel_count,
            "spike_encoder": {"schema": "custom_wavelet_encoder_spec_v1", "frequencies_hz": [0.5, 1.0, 2.0, 4.0, 8.0]},
            "spike_encoder_spec_sha256": hashlib.sha256(json.dumps({"schema": "custom_wavelet_encoder_spec_v1", "frequencies_hz": [0.5, 1.0, 2.0, 4.0, 8.0]}, sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
        }),
        encoding="utf-8",
    )
    pd.DataFrame({"segment_index": range(len(lengths)), "dataset_id": [7] * len(lengths)}).to_csv(
        directory / f"{stem}_segments.csv", index=False
    )
    if with_targets:
        targets = np.zeros((len(spike_imu), 4), dtype=np.bool_)
        targets[0, 0] = True
        np.save(directory / f"{stem}_board_event_targets.npy", targets, allow_pickle=False)
    return directory


def test_global_analysis_records_statistics_candidates_and_outliers(tmp_path: Path) -> None:
    root = tmp_path / "segments"
    _write_package(root, user="user_a", action="one", lengths=[100, 200])
    _write_package(root, user="user_b", action="two", lengths=[300, 1000])

    datasets = validate_segmented_root(root)
    analysis = analyze_segment_lengths(
        datasets, candidate_lengths=[200, 600, 1000], minimum_coverage=0.75
    )

    assert analysis["segment_count"] == 4
    assert analysis["length_statistics"]["minimum"] == 100
    assert analysis["length_statistics"]["maximum"] == 1000
    assert analysis["length_statistics"]["median"] == 250.0
    assert analysis["recommendations"]["pure_padding"]["target_length"] == 1000
    assert analysis["recommendations"]["balanced"]["target_length"] == 600
    assert analysis["recommendations"]["p99"]["requires_overflow_handling"] is True
    candidates = {item["target_length"]: item for item in analysis["candidate_lengths"]}
    assert candidates[200]["coverage"] == 0.5
    assert candidates[600]["overflow_count"] == 1
    assert candidates[1000]["padding_ratio"] == 0.6
    records = collect_segment_length_records(datasets)
    assert records[-1].dataset_id == 7
    assert records[-1].label == "L1"

    output = tmp_path / "analysis"
    paths = write_segment_length_analysis(analysis, input_root=root, output_dir=output, datasets=datasets)
    report = json.loads(paths["json"].read_text(encoding="utf-8"))
    assert report["packages"][0]["relative_lengths_path"].endswith("segment_lengths.npy")
    assert paths["histogram"].is_file()
    assert paths["ecdf"].is_file()
    outliers = pd.read_csv(paths["outliers"])
    assert {"user", "action", "label", "sample_count", "duration_seconds"} <= set(outliers.columns)
    assert resolve_target_length(
        target_length=None, analysis_report=paths["json"], recommendation="p99",
        datasets=datasets, input_root=root,
    ) == 979


def test_rounding_validation_and_missing_sibling_fail_with_identity(tmp_path: Path) -> None:
    root = tmp_path / "segments"
    directory = _write_package(root, user="user_a", action="one", lengths=[601])
    analysis = analyze_segment_lengths(validate_segmented_root(root), round_to=32)
    assert analysis["recommendations"]["pure_padding"]["target_length"] == 608

    (directory / "user_a_action_one_spikeIMU.npy").unlink()
    with pytest.raises(SegmentPaddingError, match="user='user_a'.*action='one'.*missing required"):
        validate_segmented_root(root)


def test_padding_arrays_skip_overflow_and_preserve_board_targets(tmp_path: Path) -> None:
    root = tmp_path / "segments"
    _write_package(root, user="user_a", action="one", lengths=[3, 5], with_targets=True)
    dataset = validate_segmented_root(root)[0]
    source_hashes = {
        path: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in dataset.paths.input_dir.iterdir()
    }
    result = build_padding_package(dataset, target_length=4)
    assert result.summary["spike_encoder"] == dataset.segmentation_summary["spike_encoder"]
    assert result.summary["spike_encoder_spec_sha256"] == dataset.segmentation_summary["spike_encoder_spec_sha256"]

    assert result.padded_spike_imu.shape == (1, 4, 21)
    np.testing.assert_array_equal(result.labels, ["L0"])
    np.testing.assert_array_equal(result.valid_lengths, [3])
    np.testing.assert_array_equal(result.valid_mask.sum(axis=1), [3])
    assert result.padded_board_event_targets is not None
    assert result.padded_board_event_targets.shape == (1, 4, 4)
    assert not result.padded_board_event_targets[:, 3:].any()
    np.testing.assert_array_equal(result.padded_spike_imu[0, :3], dataset.spike_imu[:3])
    assert result.summary["skipped_segment_count"] == 1
    assert result.summary["overflow_policy"] == "skip"
    assert result.summary["channel_count"] == 21
    assert result.summary["input_kind"] == "spike-imu"
    assert result.summary["feature_schema"] == "signed_wavelet_events_plus_imu_v1"
    manifest = result.manifest
    assert manifest["exported"].tolist() == [True, False]
    assert manifest.loc[1, "skip_reason"] == "length_exceeds_target"
    assert pd.isna(manifest.loc[1, "output_segment_index"])
    assert manifest["channel_count"].notna().all()
    assert manifest["channel_count"].nunique() == 1
    assert int(manifest["channel_count"].iloc[0]) == dataset.spike_imu.shape[1]

    output = tmp_path / "padded"
    summary = publish_padded_root([dataset], input_root=root, output_root=output, target_length=4)
    assert summary["spike_encoder"] == dataset.segmentation_summary["spike_encoder"]
    assert summary["spike_encoder_spec_sha256"] == dataset.segmentation_summary["spike_encoder_spec_sha256"]
    assert summary["failed_package_count"] == 0
    assert summary["segment_count"] == 1
    assert summary["skipped_segment_count"] == 1
    base = output / "user_a" / "action_one"
    assert np.load(base / "user_a_action_one_paddedSpikeIMU.npy", allow_pickle=False).shape == (1, 4, 21)
    assert np.load(base / "user_a_action_one_padded_board_event_targets.npy", allow_pickle=False).shape == (1, 4, 4)
    published_manifest = pd.read_csv(base / "user_a_action_one_padding_manifest.csv")
    assert published_manifest["channel_count"].notna().all()
    assert published_manifest["channel_count"].nunique() == 1
    assert int(published_manifest["channel_count"].iloc[0]) == 21
    np.testing.assert_array_equal(
        np.load(base / "user_a_action_one_labels.npy", allow_pickle=False), ["L0"]
    )
    assert (output / "padding_dataset_manifest.csv").is_file()
    assert source_hashes == {
        path: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in dataset.paths.input_dir.iterdir()
    }


def test_padding_rejects_non_spike_channel_count(tmp_path: Path) -> None:
    root = tmp_path / "segments"
    _write_package(
        root,
        user="user_a",
        action="one",
        lengths=[2, 3],
        channel_count=9,
    )
    with pytest.raises(SegmentPaddingError, match="expected shape .*21"):
        validate_segmented_root(root)


def test_padding_rejects_non_spike_summary(tmp_path: Path) -> None:
    root = tmp_path / "segments"
    directory = _write_package(root, user="user_a", action="one", lengths=[2, 3])
    summary_path = directory / "user_a_action_one_segmentation_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["input_kind"] = "raw-ring"
    summary_path.write_text(json.dumps(summary), encoding="utf-8")

    with pytest.raises(SegmentPaddingError, match="only supports input_kind='spike-imu'"):
        validate_segmented_root(root)


def test_padding_rejects_output_root_that_equals_or_contains_input(tmp_path: Path) -> None:
    root = tmp_path / "segments"
    _write_package(root, user="user_a", action="one", lengths=[3, 5])
    dataset = validate_segmented_root(root)[0]
    source_hashes = {
        path: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in dataset.paths.input_dir.iterdir()
    }

    for output in (root, root.parent):
        with pytest.raises(
            SegmentPaddingError,
            match="output root must not equal or contain the input root",
        ):
            publish_padded_root(
                [dataset], input_root=root, output_root=output,
                target_length=6, overwrite=True,
            )
        assert source_hashes == {
            path: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in dataset.paths.input_dir.iterdir()
        }


def test_padding_value_must_remain_finite_in_source_dtype(tmp_path: Path) -> None:
    root = tmp_path / "segments"
    _write_package(root, user="user_a", action="one", lengths=[3, 5])
    dataset = validate_segmented_root(root)[0]
    output = tmp_path / "padded"

    with pytest.raises(SegmentPaddingError, match="not representable as a finite float32"):
        build_padding_package(dataset, target_length=6, padding_value=1e100)
    with pytest.raises(SegmentPaddingError, match="not representable as a finite float32"):
        publish_padded_root(
            [dataset], input_root=root, output_root=output,
            target_length=6, padding_value=1e100,
        )
    assert not output.exists()

    result = build_padding_package(dataset, target_length=6, padding_value=0.1)
    stored_value = np.float32(0.1).item()
    assert result.summary["padding_value"] == stored_value
    assert np.isfinite(result.padded_spike_imu).all()
    np.testing.assert_array_equal(result.padded_spike_imu[0, 3:], np.full((3, 21), stored_value, dtype=np.float32))


def test_importing_writingring_does_not_change_a_selected_matplotlib_backend() -> None:
    command = (
        "import matplotlib; matplotlib.use('svg'); before = matplotlib.get_backend(); "
        "import writingring; after = matplotlib.get_backend(); "
        "assert before.lower() == after.lower() == 'svg', (before, after)"
    )
    completed = subprocess.run(
        [sys.executable, "-c", command],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
