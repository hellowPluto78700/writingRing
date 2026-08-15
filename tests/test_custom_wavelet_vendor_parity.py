"""Phase 3 parity checks against a checked-in vendor 64 Hz golden fixture.

The fixture is generated only by ``scripts/generate_custom_wavelet_vendor_fixture.py``
from the read-only Neuromorphic-Gravity PyTorch pipeline.  Runtime code and
ordinary tests do not import vendor modules.  The vendor computes Prony fits
and recurrent filtering in float32 while WritingRing deliberately uses NumPy
float64 internally before the configured float32 output cast.  A 5e-4 absolute
tolerance covers the expected accumulated solver/rounding difference; channel
order, extrema positions, and signed-amplitude semantics are otherwise fixed.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from writingring.spike_encoding.encoders.custom_wavelet import (
    CustomWaveletEncoder,
    CustomWaveletSettings,
    acceleration_wavelet,
)


_FIXTURE_PATH = Path(__file__).parent / "fixtures" / "custom_wavelet_vendor_64hz.npz"
_COEFFICIENT_ATOL = 2e-6
_RESPONSE_ATOL = 5e-4


@pytest.fixture(scope="module")
def vendor_fixture() -> dict[str, np.ndarray]:
    """Load vendor results without importing vendor dependencies at test time."""

    with np.load(_FIXTURE_PATH, allow_pickle=False) as loaded:
        return {name: loaded[name].copy() for name in loaded.files}


def _reference_settings(vendor_fixture: dict[str, np.ndarray]) -> CustomWaveletSettings:
    return CustomWaveletSettings(
        sampling_rate_hz=float(vendor_fixture["sampling_rate_hz"]),
        frequencies_hz=tuple(float(value) for value in vendor_fixture["frequencies_hz"]),
        output_dtype="float32",
    )


def test_acceleration_wavelets_match_vendor_golden_fixture(
    vendor_fixture: dict[str, np.ndarray],
) -> None:
    for width in vendor_fixture["widths_samples"]:
        expected = vendor_fixture[f"wavelet_{int(width)}"]
        actual = acceleration_wavelet(int(width), int(width))

        np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1e-7)


def test_default_64_hz_prony_coefficients_match_vendor_golden_fixture(
    vendor_fixture: dict[str, np.ndarray],
) -> None:
    encoder = CustomWaveletEncoder(_reference_settings(vendor_fixture))

    np.testing.assert_allclose(
        encoder.iir_numerator_coefficients,
        vendor_fixture["prony_numerator"],
        rtol=0.0,
        atol=_COEFFICIENT_ATOL,
    )
    np.testing.assert_allclose(
        encoder.iir_denominator_coefficients,
        vendor_fixture["prony_denominator"],
        rtol=0.0,
        atol=_COEFFICIENT_ATOL,
    )
    assert encoder.iir_numerator_coefficients.flags.writeable is False
    assert encoder.iir_denominator_coefficients.flags.writeable is False


@pytest.mark.parametrize("case_name", ("impulse", "sine_mixture", "plateau", "random"))
def test_iir_and_signed_extrema_match_vendor_golden_fixture(
    vendor_fixture: dict[str, np.ndarray],
    case_name: str,
) -> None:
    encoder = CustomWaveletEncoder(_reference_settings(vendor_fixture))
    responses: list[np.ndarray] = []
    events: list[np.ndarray] = []
    for sample in vendor_fixture[f"{case_name}_input"]:
        events.append(encoder.step(sample))
        responses.append(encoder.last_wavelet_response)
    actual_responses = np.stack(responses)
    actual_events = np.stack(events)

    np.testing.assert_allclose(
        actual_responses,
        vendor_fixture[f"{case_name}_iir"],
        rtol=0.0,
        atol=_RESPONSE_ATOL,
    )
    np.testing.assert_allclose(
        actual_events,
        vendor_fixture[f"{case_name}_flattened"],
        rtol=0.0,
        atol=_RESPONSE_ATOL,
    )
    np.testing.assert_array_equal(
        vendor_fixture[f"{case_name}_events"].reshape(len(actual_events), -1),
        vendor_fixture[f"{case_name}_flattened"],
    )


def test_vendor_flatten_order_matches_axis_major_frequency_minor(
    vendor_fixture: dict[str, np.ndarray],
) -> None:
    encoder = CustomWaveletEncoder(_reference_settings(vendor_fixture))

    assert encoder.output_channel_names == (
        "event_x_0", "event_x_1", "event_x_2", "event_x_3", "event_x_4",
        "event_y_0", "event_y_1", "event_y_2", "event_y_3", "event_y_4",
        "event_z_0", "event_z_1", "event_z_2", "event_z_3", "event_z_4",
    )
    flattened = vendor_fixture["impulse_flattened"]
    assert flattened.shape == (96, 15)
    np.testing.assert_array_equal(
        vendor_fixture["impulse_events"].transpose(0, 1, 2).reshape(96, 15),
        flattened,
    )
