from __future__ import annotations

import numpy as np

from writingring.spike_encoding.encoders.custom_wavelet import (
    CustomWaveletEncoder,
    CustomWaveletSettings,
    _LocalExtremaDetector,
    acceleration_wavelet,
    prony_iir_coefficients,
)
from writingring.spike_encoding.registry import create_encoder
from writingring.spike_encoding.runner import run_spike_encoder


def test_canonical_encoder_identity_tracks_frequencies_widths_and_transform() -> None:
    default = CustomWaveletEncoder()
    alternate = CustomWaveletEncoder(
        CustomWaveletSettings(frequencies_hz=(1.0, 2.0, 4.0, 8.0, 16.0))
    )
    rectified = CustomWaveletEncoder(
        CustomWaveletSettings(post_encode_transform="AbsRectify")
    )

    assert default.canonical_encoder_spec["frequencies_hz"] == [0.5, 1.0, 2.0, 4.0, 8.0]
    assert alternate.canonical_encoder_spec["wavelet_widths_samples"] == [200, 100, 50, 25, 12]
    assert default.output_channel_names == alternate.output_channel_names
    assert default.canonical_encoder_spec_sha256 != alternate.canonical_encoder_spec_sha256
    assert default.canonical_encoder_spec_sha256 != rectified.canonical_encoder_spec_sha256


def test_canonical_encoder_identity_tracks_event_affecting_settings() -> None:
    baseline = CustomWaveletEncoder()
    changed = CustomWaveletEncoder(
        CustomWaveletSettings(max_filter_frequency_decades=0.25)
    )

    assert baseline.canonical_encoder_spec_sha256 != changed.canonical_encoder_spec_sha256


def test_prony_fit_returns_stable_second_order_coefficients() -> None:
    settings = CustomWaveletSettings(frequencies_hz=(0.5, 1.0, 2.0, 4.0, 8.0))
    encoder = CustomWaveletEncoder(settings)
    kernel = encoder.settings.wavelet_widths_samples[0]
    coefficients = prony_iir_coefficients(
        acceleration_wavelet(kernel, kernel),
        denominator_order=2,
        numerator_order=2,
    )

    assert coefficients[0].shape == (3,)
    assert coefficients[1].shape == (3,)
    assert coefficients[1][0] == 1.0
    assert np.isfinite(coefficients[0]).all()
    assert np.isfinite(coefficients[1]).all()


def test_encoder_emits_fixed_axis_major_channel_count_and_stable_names() -> None:
    encoder = CustomWaveletEncoder(
        CustomWaveletSettings(
            frequencies_hz=(1.0, 2.0, 5.0, 10.0, 20.0),
            output_dtype="float64",
        )
    )
    samples = np.column_stack((np.sin(np.arange(120) / 4.0), np.zeros(120), np.ones(120)))

    result = encoder.encode_sequence(samples)

    assert result.values.shape == (120, 15)
    assert result.values.dtype == np.float64
    assert result.representation == "signed_sparse_wavelet_extrema"
    assert result.channel_names == (
        "event_x_0", "event_x_1", "event_x_2", "event_x_3", "event_x_4",
        "event_y_0", "event_y_1", "event_y_2", "event_y_3", "event_y_4",
        "event_z_0", "event_z_1", "event_z_2", "event_z_3", "event_z_4",
    )
    assert np.isfinite(result.values).all()


def test_post_encode_abs_rectify_preserves_occurrences_and_signed_step() -> None:
    settings = dict(
        frequencies_hz=(1.0, 2.0, 4.0, 8.0, 16.0),
        max_filter_time_s=0.3,
        output_dtype="float64",
    )
    samples = np.column_stack(
        (
            np.sin(np.arange(160) / 3.0),
            np.zeros(160),
            -np.sin(np.arange(160) / 3.0),
        )
    )
    signed_encoder = CustomWaveletEncoder(CustomWaveletSettings(**settings))
    rectified_encoder = CustomWaveletEncoder(
        CustomWaveletSettings(**settings, post_encode_transform="AbsRectify")
    )

    signed_step = signed_encoder.step(samples[0])
    rectified_step = rectified_encoder.step(samples[0])
    np.testing.assert_array_equal(rectified_step, signed_step)
    signed_encoder.reset()
    rectified_encoder.reset()

    signed = signed_encoder.encode_sequence(samples)
    rectified = rectified_encoder.encode_sequence(samples)

    assert signed.representation == "signed_sparse_wavelet_extrema"
    assert rectified.representation == "abs_rectified_sparse_wavelet_extrema"
    np.testing.assert_array_equal(rectified.values, np.abs(signed.values))
    np.testing.assert_array_equal(rectified.values != 0.0, signed.values != 0.0)
    assert rectified.values.shape == signed.values.shape == (160, 15)
    assert rectified.channel_names == signed.channel_names
    assert np.all(rectified.values >= 0.0)
    assert np.count_nonzero(signed.values < 0.0) > 0
    assert not signed.values.flags.writeable
    assert not rectified.values.flags.writeable
    assert rectified.diagnostics["post_encode_transform"] == "AbsRectify"
    assert rectified.diagnostics["event_representation"] == rectified.representation
    assert rectified_encoder.encoding_metadata["post_encode_transform"] == "AbsRectify"
    assert rectified_encoder.encoding_metadata["event_representation"] == rectified.representation


def test_polarity_split_abs_preserves_signed_event_occurrences_and_names() -> None:
    settings = dict(
        frequencies_hz=(1.0, 2.0, 4.0, 8.0, 16.0),
        max_filter_time_s=0.3,
        output_dtype="float64",
    )
    samples = np.column_stack(
        (
            np.sin(np.arange(160) / 3.0),
            np.zeros(160),
            -np.sin(np.arange(160) / 3.0),
        )
    )
    signed = CustomWaveletEncoder(CustomWaveletSettings(**settings)).encode_sequence(samples)
    split_encoder = CustomWaveletEncoder(
        CustomWaveletSettings(**settings, post_encode_transform="PolaritySplitAbs")
    )
    split = split_encoder.encode_sequence(samples)

    assert split.values.shape == (160, 30)
    assert split.representation == "polarity_split_sparse_wavelet_extrema"
    assert split.channel_names[:4] == (
        "event_x_0_pos",
        "event_x_0_neg_abs",
        "event_x_1_pos",
        "event_x_1_neg_abs",
    )
    assert split_encoder.canonical_encoder_spec["channel_order"] == (
        "axis_major_frequency_minor_pairwise_positive_negative"
    )
    assert split_encoder.canonical_encoder_spec["event_channel_names"] == list(split.channel_names)
    assert np.all(split.values >= 0.0)
    np.testing.assert_array_equal(split.values[:, 0::2], np.maximum(signed.values, 0.0))
    np.testing.assert_array_equal(split.values[:, 1::2], np.maximum(-signed.values, 0.0))
    np.testing.assert_array_equal(np.any(split.values != 0.0, axis=1), np.any(signed.values != 0.0, axis=1))
    assert np.count_nonzero(split.values) == np.count_nonzero(signed.values)


def test_encoder_reset_reproduces_one_sequence_and_runner_resets_each_boundary() -> None:
    settings = {
        "frequencies_hz": [1.0, 2.0, 4.0, 8.0, 16.0],
        "max_filter_time_s": 0.02,
        "output_dtype": "float32",
    }
    encoder = create_encoder("custom-wavelet", settings=settings)
    samples = np.column_stack((np.sin(np.arange(20)), np.cos(np.arange(20)), np.arange(20)))

    first = encoder.encode_sequence(samples)
    encoder.reset()
    second = encoder.encode_sequence(samples)
    output = run_spike_encoder(
        encoder,
        acceleration_g=np.vstack((samples, samples)),
        sequence_offsets=np.array([0, 20, 40], dtype=np.int64),
    )

    np.testing.assert_array_equal(first.values, second.values)
    np.testing.assert_array_equal(output.values[:20], output.values[20:])
    assert output.summary["channel_count"] == 15


def test_sequence_occurrence_alignment_matches_padded_causal_detection_path() -> None:
    settings = CustomWaveletSettings(frequencies_hz=(1.0, 2.0, 4.0, 8.0, 16.0))
    samples = np.column_stack((np.arange(80), -np.arange(80), np.ones(80)))
    detected_encoder = CustomWaveletEncoder(settings)
    encoded = CustomWaveletEncoder(settings)

    padding = detected_encoder.padding_samples_each_side
    padded = np.pad(samples, ((padding, padding), (0, 0)), mode="reflect")
    detection_aligned = np.stack([detected_encoder.step(sample) for sample in padded])
    by_sequence = encoded.encode_sequence(samples).values

    np.testing.assert_array_equal(
        by_sequence,
        detection_aligned[2 * padding : 2 * padding + len(samples)],
    )


def test_window_dimensions_follow_settings_and_keep_signed_extrema() -> None:
    encoder = CustomWaveletEncoder(
        CustomWaveletSettings(
            frequencies_hz=(1.0, 2.0, 4.0, 8.0, 16.0),
            max_filter_time_s=0.3,
            max_filter_frequency_decades=0.5,
        )
    )

    assert encoder.max_filter_time_samples == 61
    assert encoder.max_filter_frequency_bands == 1
    result = encoder.encode_sequence(
        np.column_stack((np.sin(np.arange(160) / 3.0), np.zeros(160), -np.sin(np.arange(160) / 3.0)))
    )
    assert np.count_nonzero(result.values > 0.0) > 0
    assert np.count_nonzero(result.values < 0.0) > 0


def test_reference_window_rounding_is_dynamic_for_64_hz() -> None:
    encoder = CustomWaveletEncoder(
        CustomWaveletSettings(
            sampling_rate_hz=64.0,
            frequencies_hz=(0.5, 1.0, 2.0, 4.0, 8.0),
        )
    )

    assert encoder.max_filter_time_samples == 19


def test_extrema_detector_preserves_signed_maxima_and_minima_once() -> None:
    maximum = _LocalExtremaDetector(3, 1, 1)
    minimum = _LocalExtremaDetector(3, 1, 1)
    plateau = _LocalExtremaDetector(3, 1, 1)

    maximum_events = [maximum.step(np.full((3, 1), value)) for value in (1.0, 5.0, 1.0)]
    minimum_events = [minimum.step(np.full((3, 1), value)) for value in (-1.0, -5.0, -1.0)]
    plateau_events = [plateau.step(np.full((3, 1), 2.0)) for _ in range(3)]

    np.testing.assert_array_equal(maximum_events[-1], np.full((3, 1), 5.0))
    np.testing.assert_array_equal(minimum_events[-1], np.full((3, 1), -5.0))
    np.testing.assert_array_equal(plateau_events[-1], np.full((3, 1), 2.0))


def test_padding_length_is_derived_from_the_odd_extrema_window() -> None:
    for rate, expected_padding in ((100.0, 15), (200.0, 30), (400.0, 60)):
        encoder = CustomWaveletEncoder(
            CustomWaveletSettings(
                sampling_rate_hz=rate,
                frequencies_hz=(1.0, 2.0, 4.0, 8.0, 16.0),
            )
        )

        assert encoder.padding_samples_each_side == encoder.max_filter_time_samples // 2
        assert encoder.padding_samples_each_side == expected_padding
        assert encoder.padding_duration_seconds == expected_padding / rate
