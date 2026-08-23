from __future__ import annotations

import hashlib
import json

from scripts import build_polarity_split_variant


def _sha256(value: dict[str, object]) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def test_patch_summary_publishes_unsigned_polarity_split_contract() -> None:
    source_encoder: dict[str, object] = {
        "schema": "custom_wavelet_encoder_spec_v1",
        "name": "custom-wavelet",
        "channel_order": "axis_major_frequency_minor",
        "event_channel_names": [f"event_{index}" for index in range(15)],
        "post_encode_transform": None,
        "event_representation": "signed_sparse_wavelet_extrema",
    }
    summary: dict[str, object] = {
        "feature_schema": "signed_wavelet_events_plus_imu_v1",
        "channel_count": 21,
        "event_representation": "signed",
        "event_feature_schema": "custom_wavelet_signed_events_v1",
        "event_channel_count": 15,
        "spike_encoder": source_encoder,
        "spike_encoder_spec_sha256": _sha256(source_encoder),
        "channel_names": [f"channel_{index}" for index in range(21)],
    }

    build_polarity_split_variant.patch_summary(summary)

    assert summary["feature_schema"] == "polarity_split_wavelet_events_plus_imu_v1"
    assert summary["channel_count"] == 36
    assert summary["event_representation"] == "unsigned"
    assert (
        summary["event_feature_schema"]
        == "custom_wavelet_polarity_split_abs_events_v1"
    )
    assert summary["event_channel_count"] == 30
    assert summary["event_encoding"] == "polarity_split_sparse_wavelet_extrema"

    derived_encoder = summary["spike_encoder"]
    assert isinstance(derived_encoder, dict)
    assert derived_encoder["post_encode_transform"] == "PolaritySplitAbs"
    assert derived_encoder["event_representation"] == "polarity_split_sparse_wavelet_extrema"
    assert derived_encoder["channel_order"] == (
        "axis_major_frequency_minor_pairwise_positive_negative"
    )
    assert derived_encoder["event_channel_names"] == [
        name
        for index in range(15)
        for name in (f"event_{index}_pos", f"event_{index}_neg_abs")
    ]
    assert summary["spike_encoder_spec_sha256"] == _sha256(derived_encoder)
    assert summary["source_spike_encoder"] == source_encoder
    assert summary["source_spike_encoder_spec_sha256"] == _sha256(source_encoder)
