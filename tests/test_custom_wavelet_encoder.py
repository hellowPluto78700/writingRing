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


def test_prony_fit_returns_stable_second_order_coefficients() -> None:
    settings = CustomWaveletSettings(frequencies_hz=(8.0,))
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


def test_encoder_emits_dynamic_axis_major_channel_count_and_names() -> None:
    encoder = CustomWaveletEncoder(
        CustomWaveletSettings(
            frequencies_hz=(1.0, 2.0, 5.0, 10.0),
            output_dtype="float64",
        )
    )
    samples = np.column_stack((np.sin(np.arange(120) / 4.0), np.zeros(120), np.ones(120)))

    result = encoder.encode_sequence(samples)

    assert result.values.shape == (120, 12)
    assert result.values.dtype == np.float64
    assert result.representation == "signed_sparse_wavelet_extrema"
    assert result.channel_names == (
        "event_x_1_hz",
        "event_x_2_hz",
        "event_x_5_hz",
        "event_x_10_hz",
        "event_y_1_hz",
        "event_y_2_hz",
        "event_y_5_hz",
        "event_y_10_hz",
        "event_z_1_hz",
        "event_z_2_hz",
        "event_z_5_hz",
        "event_z_10_hz",
    )
    assert np.isfinite(result.values).all()


def test_encoder_reset_reproduces_one_sequence_and_runner_resets_each_boundary() -> None:
    settings = {
        "frequencies_hz": [2.0, 4.0],
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
    assert output.summary["channel_count"] == 6


def test_sequence_occurrence_alignment_matches_padded_causal_detection_path() -> None:
    settings = CustomWaveletSettings(frequencies_hz=(2.0, 4.0))
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
            frequencies_hz=(2.0, 4.0, 8.0),
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
                frequencies_hz=(2.0, 4.0, 8.0),
            )
        )

        assert encoder.padding_samples_each_side == encoder.max_filter_time_samples // 2
        assert encoder.padding_samples_each_side == expected_padding
        assert encoder.padding_duration_seconds == expected_padding / rate
