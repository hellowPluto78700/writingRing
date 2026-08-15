from __future__ import annotations

import json
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import scripts.run_experiment_b as runner_b
import scripts.run_experiment_c as runner_c
import scripts.run_experiment_d as runner_d
from snn.accel_reconstruction_eval.datasets import (
    Action0DatasetError,
    build_cohort_identity,
    fit_acceleration_normalization,
    load_acceleration_data,
    prepare_user_disjoint_splits,
    validate_checkpoint_cohort_identity,
)


def _write_root(
    root: Path,
    *,
    action: str,
    target_length: int = 4,
    sampling_rate_hz: float = 200.0,
    encoder_spec: dict[str, object] | None = None,
    include_encoder_identity: bool = True,
) -> Path:
    root.mkdir(parents=True)
    root_summary = {
        "input_kind": "spike-imu",
        "feature_schema": "signed_wavelet_events_plus_imu_v1",
        "channel_count": 21,
        "target_length": target_length,
        "sampling_rate_hz": sampling_rate_hz,
        "padding_side": "right",
    }
    if include_encoder_identity:
        encoder_spec = encoder_spec or {
            "schema": "custom_wavelet_encoder_spec_v1",
            "frequencies_hz": [0.5, 1.0, 2.0, 4.0, 8.0],
            "wavelet_widths_samples": [400, 200, 100, 50, 25],
            "post_encode_transform": "none",
        }
        root_summary["spike_encoder"] = encoder_spec
        root_summary["spike_encoder_spec_sha256"] = hashlib.sha256(
            json.dumps(encoder_spec, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
    (root / "padding_dataset_summary.json").write_text(
        json.dumps(root_summary), encoding="utf-8"
    )

    for user_index, user in enumerate(("user_0", "user_1", "user_2")):
        action_dir = root / user / f"action_{action}"
        action_dir.mkdir(parents=True)
        stem = f"{user}_action_{action}"
        labels = np.asarray(["a", "b"])
        valid_lengths = np.asarray([target_length, target_length - 1], dtype=np.int64)
        valid_mask = np.arange(target_length)[None, :] < valid_lengths[:, None]
        values = np.zeros((len(labels), target_length, 21), dtype=np.float32)
        values[:, :, 15:18] = float(user_index + int(action) + 1)
        values[~valid_mask] = 0.0
        np.save(action_dir / f"{stem}_paddedSpikeIMU.npy", values, allow_pickle=False)
        np.save(action_dir / f"{stem}_labels.npy", labels, allow_pickle=False)
        np.save(action_dir / f"{stem}_valid_lengths.npy", valid_lengths, allow_pickle=False)
        np.save(action_dir / f"{stem}_valid_mask.npy", valid_mask, allow_pickle=False)
        pd.DataFrame(
            {
                "segment_index": [0, 1],
                "output_segment_index": [0, 1],
                "exported": [True, True],
                "original_length": valid_lengths,
                "target_length": [target_length, target_length],
                "label": labels,
                "dataset_id": ["0", "0"],
            }
        ).to_csv(action_dir / f"{stem}_padding_manifest.csv", index=False)
        package_summary = {
            **root_summary,
            "segment_count": len(labels),
            "padding_side": "right",
            "overflow_policy": "skip",
        }
        (action_dir / f"{stem}_padding_summary.json").write_text(
            json.dumps(package_summary), encoding="utf-8"
        )
    return root


def test_multi_root_loader_reindexes_packages_and_preserves_user_splits(
    tmp_path: Path,
) -> None:
    action0 = _write_root(tmp_path / "action0", action="0")
    action1 = _write_root(tmp_path / "action1", action="1")

    data = load_acceleration_data([action0, action1])

    assert data.padded_roots == (action0.resolve(), action1.resolve())
    assert data.selected_actions == ("0", "1")
    assert len(data.packages) == 6
    assert set(data.sample_manifest["action"]) == {"0", "1"}
    assert data.sample_manifest["sample_id"].is_unique
    assert set(data.sample_manifest["package_index"]) == set(range(6))
    assert set(data.sample_manifest["source_root_index"]) == {0, 1}
    assert set(data.sample_manifest["source_padded_root"]) == {
        str(action0.resolve()),
        str(action1.resolve()),
    }

    split = prepare_user_disjoint_splits(
        data.sample_manifest,
        explicit_train_users=("user_0",),
        explicit_val_users=("user_1",),
        explicit_test_users=("user_2",),
    )
    assert split.sample_manifest.groupby("user")["split"].nunique().eq(1).all()
    assert split.sample_manifest.groupby("user")["action"].nunique().eq(2).all()
    normalization = fit_acceleration_normalization(
        data.packages,
        split.sample_manifest,
        split="train",
        source="raw",
    )
    assert normalization.fitted_on == "raw:train"
    assert normalization.valid_time_points == 14


def test_multi_root_loader_rejects_duplicate_roots_and_package_identities(
    tmp_path: Path,
) -> None:
    action0 = _write_root(tmp_path / "action0", action="0")
    with pytest.raises(Action0DatasetError, match="Duplicate dataset roots"):
        load_acceleration_data([action0, action0])

    duplicate_action0 = _write_root(tmp_path / "duplicate_action0", action="0")
    with pytest.raises(Action0DatasetError, match="Duplicate package identity"):
        load_acceleration_data([action0, duplicate_action0])


def test_multi_root_loader_rejects_incompatible_target_length(tmp_path: Path) -> None:
    action0 = _write_root(tmp_path / "action0", action="0", target_length=4)
    action1 = _write_root(tmp_path / "action1", action="1", target_length=5)

    with pytest.raises(Action0DatasetError, match="Incompatible producer metadata"):
        load_acceleration_data([action0, action1])


def test_multi_root_loader_accepts_same_encoder_spec_across_actions(tmp_path: Path) -> None:
    spec = {
        "schema": "custom_wavelet_encoder_spec_v1",
        "frequencies_hz": [1.0, 2.0, 4.0, 8.0, 16.0],
        "wavelet_widths_samples": [200, 100, 50, 25, 12],
        "post_encode_transform": "none",
    }
    data = load_acceleration_data([
        _write_root(tmp_path / "action0", action="0", encoder_spec=spec),
        _write_root(tmp_path / "action1", action="1", encoder_spec=spec),
    ])
    assert len(data.packages) == 6


@pytest.mark.parametrize(
    "spec_a,spec_b,match",
    [
        (
            {"schema": "custom_wavelet_encoder_spec_v1", "frequencies_hz": [0.5, 1, 2, 4, 8], "wavelet_widths_samples": [400, 200, 100, 50, 25], "post_encode_transform": "none"},
            {"schema": "custom_wavelet_encoder_spec_v1", "frequencies_hz": [1, 2, 4, 8, 16], "wavelet_widths_samples": [200, 100, 50, 25, 12], "post_encode_transform": "none"},
            "Incompatible spike encoder identity.*hash",
        ),
        (
            {"schema": "custom_wavelet_encoder_spec_v1", "frequencies_hz": [1, 2, 4, 8, 16], "wavelet_widths_samples": [200, 100, 50, 25, 12], "post_encode_transform": "none"},
            {"schema": "custom_wavelet_encoder_spec_v1", "frequencies_hz": [1, 2, 4, 8, 16], "wavelet_widths_samples": [200, 100, 50, 25, 12], "post_encode_transform": "AbsRectify"},
            "spec_differences",
        ),
    ],
)
def test_multi_root_loader_rejects_different_encoder_specs(
    tmp_path: Path, spec_a: dict[str, object], spec_b: dict[str, object], match: str
) -> None:
    with pytest.raises(Action0DatasetError, match=match):
        load_acceleration_data([
            _write_root(tmp_path / "action0", action="0", encoder_spec=spec_a),
            _write_root(tmp_path / "action1", action="1", encoder_spec=spec_b),
        ])


def test_multi_root_loader_rejects_missing_encoder_identity(tmp_path: Path) -> None:
    with pytest.raises(Action0DatasetError, match="Missing spike encoder identity"):
        load_acceleration_data([
            _write_root(tmp_path / "action0", action="0", include_encoder_identity=False),
            _write_root(tmp_path / "action1", action="1"),
        ])


def test_cohort_identity_is_path_independent_and_detects_mismatch() -> None:
    manifest = pd.DataFrame(
        {
            "sample_id": [
                "user_0/action_0/stem/segment_000000",
                "user_0/action_1/stem/segment_000000",
            ]
        }
    )
    identity = build_cohort_identity(manifest, selected_actions=("0", "1"))
    relocated = manifest.assign(source_padded_root="/another/location")
    assert build_cohort_identity(
        relocated, selected_actions=("1", "0")
    ) == identity

    checkpoint = {"cohort_identity": identity.to_dict()}
    assert validate_checkpoint_cohort_identity(
        checkpoint,
        manifest,
        selected_actions=("0", "1"),
        selected_root_count=2,
    ) == identity
    with pytest.raises(ValueError, match="canonical sample-ID digest differs"):
        validate_checkpoint_cohort_identity(
            checkpoint,
            manifest.assign(
                sample_id=[
                    "user_0/action_0/stem/segment_000000",
                    "user_0/action_1/stem/segment_999999",
                ]
            ),
            selected_actions=("0", "1"),
            selected_root_count=2,
        )
    with pytest.raises(ValueError, match="selected actions"):
        validate_checkpoint_cohort_identity(
            checkpoint,
            manifest.iloc[:1],
            selected_actions=("0",),
            selected_root_count=1,
        )
    with pytest.raises(ValueError, match="Regenerate Experiment A"):
        validate_checkpoint_cohort_identity(
            {},
            manifest,
            selected_actions=("0", "1"),
            selected_root_count=2,
        )


@pytest.mark.parametrize("runner", (runner_b, runner_c, runner_d))
def test_downstream_runner_context_keeps_two_root_selection(runner: object) -> None:
    data = type(
        "Data",
        (),
        {
            "padded_root": Path("/tmp/action0/segmentation_padded"),
            "padded_roots": (
                Path("/tmp/action0/segmentation_padded"),
                Path("/tmp/action1/segmentation_padded"),
            ),
            "root_arguments": ("action0", "action1"),
            "selected_actions": ("0", "1"),
            "sample_manifest": pd.DataFrame(
                {"action": ["0", "1"], "sample_id": ["a", "b"]}
            ),
        },
    )()

    context = runner._dataset_context(data, [Path("action0"), Path("action1")])

    assert context["selected_actions"] == ("0", "1")
    assert context["selected_root_count"] == 2
    assert context["root_arguments"] == ["action0", "action1"]
