from __future__ import annotations

import numpy as np
import pytest

from writingring.spike_encoding.encoders.custom_wavelet import (
    CustomWaveletSettings,
    CustomWaveletSettingsError,
    acceleration_wavelet,
)


def test_default_settings_derive_the_documented_widths() -> None:
    settings = CustomWaveletSettings()

    assert settings.wavelet_widths_samples == (400, 200, 100, 50, 25)


def test_custom_settings_derive_dynamic_widths_and_reject_unknown_fields() -> None:
    settings = CustomWaveletSettings.from_mapping(
        {"frequencies_hz": [1.0, 2.0, 5.0, 10.0]}
    )

    assert settings.wavelet_widths_samples == (200, 100, 40, 20)
    with pytest.raises(CustomWaveletSettingsError, match="unknown custom-wavelet settings"):
        CustomWaveletSettings.from_mapping({"wavelet_shape": "acceleration"})
    with pytest.raises(CustomWaveletSettingsError, match="unknown wavelet_name"):
        CustomWaveletSettings(wavelet_name="missing")


@pytest.mark.parametrize(
    ("frequencies", "message"),
    [
        ((4.0, 1.0, 8.0), "strictly increasing"),
        ((1.0, 2.0, 2.0, 4.0), "strictly increasing"),
        ((1.0, 100.0), "Nyquist"),
    ],
)
def test_settings_reject_invalid_frequency_order_and_range(
    frequencies: tuple[float, ...],
    message: str,
) -> None:
    with pytest.raises(CustomWaveletSettingsError, match=message):
        CustomWaveletSettings(frequencies_hz=frequencies)


def test_settings_reject_widths_collapsed_by_integer_truncation() -> None:
    with pytest.raises(CustomWaveletSettingsError, match="collapse to duplicate"):
        CustomWaveletSettings(
            frequencies_hz=(21.0, 22.0),
            sampling_rate_hz=200.0,
        )


def test_settings_reject_width_that_cannot_support_prony_orders() -> None:
    with pytest.raises(CustomWaveletSettingsError, match="exceed the sum"):
        CustomWaveletSettings(frequencies_hz=(50.0,))


def test_acceleration_wavelet_matches_the_reference_equation() -> None:
    length = 7
    scale = 7.0
    x = (np.arange(length, dtype=np.float64) - (length - 1) / 2.0) / scale
    expected = np.sqrt(1.0 / scale) * (
        ((x > -0.5) & (x < 0.5)) * (29.0 / 4.0) * x * (4.0 * x**2 - 1.0)
    )

    np.testing.assert_allclose(acceleration_wavelet(length, scale), expected)
