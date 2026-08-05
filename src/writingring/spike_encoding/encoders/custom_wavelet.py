"""Stateful Custom Wavelet encoder implemented without the upstream runtime.

The equations, Prony fitting convention, and extrema-window state mirror the
read-only Neuromorphic-Gravity reference.  This implementation intentionally
uses NumPy and the project settings (200 Hz by default), rather than importing
the reference PyTorch pipeline or adopting its caller-specific 64 Hz rate.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from decimal import Decimal
import math

import numpy as np

from writingring.spike_encoding.contracts import (
    SpikeEncodingError,
    SpikeEncodingSequenceResult,
)


class CustomWaveletSettingsError(SpikeEncodingError):
    """Raised when Custom Wavelet settings cannot define one filter bank."""


WaveletFactory = Callable[[int, float], np.ndarray]


def acceleration_wavelet(length: int, scale: float) -> np.ndarray:
    """Return the reference compact acceleration wavelet kernel."""

    if not isinstance(length, int) or isinstance(length, bool) or length <= 0:
        raise CustomWaveletSettingsError("wavelet length must be a positive integer")
    if not _is_finite_real(scale) or float(scale) <= 0.0:
        raise CustomWaveletSettingsError("wavelet scale must be finite and positive")
    x = (np.arange(length, dtype=np.float64) - (length - 1) / 2.0) / float(scale)
    support = (x > -0.5) & (x < 0.5)
    kernel = support * (29.0 / 4.0) * x * (4.0 * x**2 - 1.0)
    return np.sqrt(1.0 / float(scale)) * kernel


WAVELET_REGISTRY: dict[str, WaveletFactory] = {"acceleration": acceleration_wavelet}


@dataclass(frozen=True, slots=True)
class CustomWaveletSettings:
    """Validated, encoder-private configuration for Custom Wavelet."""

    wavelet_name: str = "acceleration"
    frequencies_hz: tuple[float, ...] = (0.5, 1.0, 2.0, 4.0, 8.0)
    sampling_rate_hz: float = 200.0
    prony_denominator_order: int = 2
    prony_numerator_order: int = 2
    max_filter_time_s: float = 0.3
    max_filter_frequency_decades: float = 0.5
    output_dtype: str = "float32"

    def __post_init__(self) -> None:
        if not isinstance(self.wavelet_name, str) or self.wavelet_name not in WAVELET_REGISTRY:
            available = ", ".join(sorted(WAVELET_REGISTRY))
            raise CustomWaveletSettingsError(
                f"unknown wavelet_name {self.wavelet_name!r}; available wavelets: {available}"
            )
        frequencies = _validated_frequencies(self.frequencies_hz, self.sampling_rate_hz)
        object.__setattr__(self, "frequencies_hz", frequencies)
        object.__setattr__(self, "sampling_rate_hz", _positive_finite_float("sampling_rate_hz", self.sampling_rate_hz))
        _positive_integer("prony_denominator_order", self.prony_denominator_order)
        _positive_integer("prony_numerator_order", self.prony_numerator_order)
        object.__setattr__(self, "max_filter_time_s", _positive_finite_float("max_filter_time_s", self.max_filter_time_s))
        if not _is_finite_real(self.max_filter_frequency_decades) or float(self.max_filter_frequency_decades) < 0.0:
            raise CustomWaveletSettingsError(
                "max_filter_frequency_decades must be finite and nonnegative"
            )
        object.__setattr__(self, "max_filter_frequency_decades", float(self.max_filter_frequency_decades))
        if self.output_dtype not in {"float32", "float64"}:
            raise CustomWaveletSettingsError("output_dtype must be 'float32' or 'float64'")
        widths = self.wavelet_widths_samples
        if len(set(widths)) != len(widths):
            raise CustomWaveletSettingsError(
                "configured frequencies collapse to duplicate wavelet widths"
            )
        required = self.prony_denominator_order + self.prony_numerator_order
        if any(width <= required for width in widths):
            raise CustomWaveletSettingsError(
                "wavelet width must exceed the sum of the configured Prony orders"
            )
        if len(widths) != len(self.frequencies_hz) or any(width <= 0 for width in widths):
            raise CustomWaveletSettingsError("wavelet widths must be positive and match frequencies")

    @property
    def wavelet_widths_samples(self) -> tuple[int, ...]:
        """Derive each sampled-kernel width from its configured frequency."""

        return tuple(int(self.sampling_rate_hz / frequency) for frequency in self.frequencies_hz)

    @classmethod
    def from_mapping(cls, settings: Mapping[str, object]) -> "CustomWaveletSettings":
        """Build settings from JSON-like values while rejecting misspelled keys."""

        if not isinstance(settings, Mapping):
            raise CustomWaveletSettingsError("custom-wavelet settings must be an object")
        allowed = set(cls.__dataclass_fields__)
        unknown = sorted(set(settings) - allowed)
        if unknown:
            raise CustomWaveletSettingsError(
                f"unknown custom-wavelet settings: {', '.join(unknown)}"
            )
        payload = dict(settings)
        if "frequencies_hz" in payload:
            raw_frequencies = payload["frequencies_hz"]
            if isinstance(raw_frequencies, (str, bytes)):
                raise CustomWaveletSettingsError("frequencies_hz must be an array of numbers")
            try:
                payload["frequencies_hz"] = tuple(raw_frequencies)  # type: ignore[arg-type]
            except TypeError as error:
                raise CustomWaveletSettingsError("frequencies_hz must be an array of numbers") from error
        try:
            return cls(**payload)  # type: ignore[arg-type]
        except TypeError as error:
            raise CustomWaveletSettingsError(f"invalid custom-wavelet settings: {error}") from error


class CustomWaveletEncoder:
    """Three-axis IIR wavelet bank with signed local-extrema events."""

    name = "custom-wavelet"
    representation = "signed_sparse_wavelet_extrema"

    def __init__(self, settings: CustomWaveletSettings | None = None) -> None:
        self.settings = settings or CustomWaveletSettings()
        self._wavelet_widths = self.settings.wavelet_widths_samples
        factory = WAVELET_REGISTRY[self.settings.wavelet_name]
        coefficients = [
            prony_iir_coefficients(
                factory(width, width),
                denominator_order=self.settings.prony_denominator_order,
                numerator_order=self.settings.prony_numerator_order,
            )
            for width in self._wavelet_widths
        ]
        self._b = np.stack([item[0] for item in coefficients], axis=0)
        self._a = np.stack([item[1] for item in coefficients], axis=0)
        _validate_iir_bank(self._b, self._a)
        self._output_dtype = np.dtype(self.settings.output_dtype)
        self._channel_names = tuple(
            f"event_{axis}_{_frequency_name(frequency)}_hz"
            for axis in ("x", "y", "z")
            for frequency in self.settings.frequencies_hz
        )
        self._time_window_samples = _odd_window(
            self.settings.max_filter_time_s * self.settings.sampling_rate_hz
        )
        self._frequency_window_bands = _frequency_window_bands(
            self.settings.frequencies_hz,
            self.settings.max_filter_frequency_decades,
        )
        self._extrema = _LocalExtremaDetector(
            time_window_samples=self._time_window_samples,
            frequency_window_bands=self._frequency_window_bands,
            band_count=len(self.settings.frequencies_hz),
        )
        self.reset()

    @classmethod
    def from_settings(cls, settings: Mapping[str, object]) -> "CustomWaveletEncoder":
        """Create the registered encoder from generic registry settings."""

        return cls(CustomWaveletSettings.from_mapping(settings))

    @property
    def output_channel_names(self) -> tuple[str, ...]:
        """Return axis-major, frequency-minor event labels."""

        return self._channel_names

    @property
    def output_metadata(self) -> dict[str, object]:
        """Declare the encoder-specific layout of its flattened event channels."""

        return {
            "channel_order": "axis_major_frequency_minor",
            "event_index_semantics": "wavelet_extrema_occurrence_index",
        }

    @property
    def wavelet_widths_samples(self) -> tuple[int, ...]:
        """Expose derived, auditable sampled-kernel widths."""

        return self._wavelet_widths

    @property
    def max_filter_time_samples(self) -> int:
        """Return the validated odd extrema-window size."""

        return self._time_window_samples

    @property
    def occurrence_lookahead_samples(self) -> int:
        """Return the causal extrema confirmation latency in samples."""

        return self._time_window_samples // 2

    @property
    def padding_samples_each_side(self) -> int:
        """Return the reflect-padding width derived from the extrema window."""

        return self.occurrence_lookahead_samples

    @property
    def padding_duration_seconds(self) -> float:
        """Return the actual per-side padding duration after integer rounding."""

        return self.padding_samples_each_side / self.settings.sampling_rate_hz

    @property
    def max_filter_frequency_bands(self) -> int:
        """Return the odd cross-band extrema-window size."""

        return self._frequency_window_bands

    @property
    def iir_numerator_coefficients(self) -> np.ndarray:
        """Return immutable per-frequency ``b`` coefficients for diagnostics."""

        coefficients = self._b.copy()
        coefficients.setflags(write=False)
        return coefficients

    @property
    def iir_denominator_coefficients(self) -> np.ndarray:
        """Return immutable per-frequency ``a`` coefficients for diagnostics."""

        coefficients = self._a.copy()
        coefficients.setflags(write=False)
        return coefficients

    @property
    def last_wavelet_response(self) -> np.ndarray:
        """Return the last pre-extrema three-axis IIR response."""

        if self._last_wavelet_response is None:
            raise SpikeEncodingError("custom-wavelet has not processed a sample")
        response = self._last_wavelet_response.copy()
        response.setflags(write=False)
        return response

    @property
    def encoding_metadata(self) -> dict[str, object]:
        """Return JSON-ready Custom Wavelet provenance without exposing state."""

        return {
            "wavelet_name": self.settings.wavelet_name,
            "frequencies_hz": list(self.settings.frequencies_hz),
            "frequency_order": "strictly_increasing_hz",
            "wavelet_widths_samples": list(self._wavelet_widths),
            "width_calculation": "int(sampling_rate_hz / frequency_hz)",
            "sampling_rate_hz": self.settings.sampling_rate_hz,
            "prony_denominator_order": self.settings.prony_denominator_order,
            "prony_numerator_order": self.settings.prony_numerator_order,
            "max_filter_time_s": self.settings.max_filter_time_s,
            "max_filter_time_samples": self._time_window_samples,
            "occurrence_lookahead_samples": self.occurrence_lookahead_samples,
            "padding_samples_each_side": self.padding_samples_each_side,
            "padding_duration_seconds": self.padding_duration_seconds,
            "padding_mode": "reflect",
            "padding_boundary_assumption": "accepted",
            "boundary_validity_mask_emitted": False,
            "extrema_detection_latency_compensated": True,
            "event_index_semantics": "wavelet_extrema_occurrence_index",
            "iir_delay_compensated": False,
            "iir_warmup_guaranteed": False,
            "max_filter_frequency_decades": self.settings.max_filter_frequency_decades,
            "max_filter_frequency_bands": self._frequency_window_bands,
            "output_dtype": self.settings.output_dtype,
        }

    def reset(self) -> None:
        """Clear IIR and extrema history for one independent sequence."""

        band_count = len(self.settings.frequencies_hz)
        self._input_history = np.zeros(
            (3, self.settings.prony_numerator_order), dtype=np.float64
        )
        self._output_history = np.zeros(
            (3, band_count, self.settings.prony_denominator_order), dtype=np.float64
        )
        self._last_wavelet_response: np.ndarray | None = None
        self._extrema.reset()

    def step(self, acceleration_g: np.ndarray) -> np.ndarray:
        """Encode one finite acceleration sample with retained stream state."""

        input_value = _validated_sample(acceleration_g)
        response = input_value[:, None] * self._b[None, :, 0]
        for index in range(1, self.settings.prony_numerator_order + 1):
            response += self._input_history[:, index - 1, None] * self._b[None, :, index]
        for index in range(1, self.settings.prony_denominator_order + 1):
            response -= self._output_history[:, :, index - 1] * self._a[None, :, index]
        if not np.isfinite(response).all():
            raise SpikeEncodingError("custom-wavelet IIR response became non-finite")
        self._push_input_history(input_value)
        self._push_output_history(response)
        self._last_wavelet_response = response.copy()
        events = self._extrema.step(response)
        return events.reshape(-1).astype(self._output_dtype, copy=False)

    def encode_sequence(self, acceleration_g: np.ndarray) -> SpikeEncodingSequenceResult:
        """Encode one recording and align extrema events to source occurrences.

        The causal extrema detector emits an event ``H`` samples after its
        occurrence, where ``H`` is half of its odd time window.  Reflect
        padding supplies both the leading context and the trailing flush;
        cropping detection rows ``[2H:2H + N]`` restores the source recording
        index without attempting to correct causal IIR delay.
        """

        values = _validated_sequence(acceleration_g)
        padding = self.padding_samples_each_side
        padded = np.pad(
            values,
            pad_width=((padding, padding), (0, 0)),
            mode="reflect",
        )
        detected = np.empty((len(padded), len(self._channel_names)), dtype=self._output_dtype)
        for index, sample in enumerate(padded):
            detected[index] = self.step(sample)
        encoded = detected[2 * padding : 2 * padding + len(values)].copy()
        if encoded.shape != (len(values), len(self._channel_names)):
            raise SpikeEncodingError("custom-wavelet occurrence alignment changed sequence length")
        if not np.isfinite(encoded).all():
            raise SpikeEncodingError("custom-wavelet output became non-finite")
        encoded.setflags(write=False)
        return SpikeEncodingSequenceResult(
            values=encoded,
            channel_names=self._channel_names,
            representation=self.representation,
            diagnostics={
                "wavelet_name": self.settings.wavelet_name,
                "frequencies_hz": list(self.settings.frequencies_hz),
                "wavelet_widths_samples": list(self._wavelet_widths),
                "max_filter_time_samples": self._time_window_samples,
                "occurrence_lookahead_samples": self.occurrence_lookahead_samples,
                "padding_samples_each_side": padding,
                "padding_duration_seconds": self.padding_duration_seconds,
                "padding_mode": "reflect",
                "event_index_semantics": "wavelet_extrema_occurrence_index",
                "max_filter_frequency_bands": self._frequency_window_bands,
                "output_dtype": self.settings.output_dtype,
            },
        )

    def _push_input_history(self, value: np.ndarray) -> None:
        if self._input_history.shape[1] > 1:
            self._input_history[:, 1:] = self._input_history[:, :-1]
        self._input_history[:, 0] = value

    def _push_output_history(self, response: np.ndarray) -> None:
        if self._output_history.shape[2] > 1:
            self._output_history[:, :, 1:] = self._output_history[:, :, :-1]
        self._output_history[:, :, 0] = response


@dataclass(slots=True)
class _LocalExtremaDetector:
    """Reference-style causal history window for signed extrema events."""

    time_window_samples: int
    frequency_window_bands: int
    band_count: int
    _history: np.ndarray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self._history = np.zeros(
            (3, self.band_count, self.time_window_samples - 1), dtype=np.float64
        )

    def step(self, response: np.ndarray) -> np.ndarray:
        memory = np.concatenate((self._history, response[:, :, None]), axis=2)
        centre = memory[:, :, self.time_window_samples // 2]
        maximum = self._frequency_pool(memory, use_minimum=False)
        minimum = self._frequency_pool(memory, use_minimum=True)
        is_maximum = maximum == centre
        is_minimum = minimum == centre
        events = np.where(is_maximum | is_minimum, centre, 0.0)
        self._history = memory[:, :, 1:]
        return events

    def _frequency_pool(self, memory: np.ndarray, *, use_minimum: bool) -> np.ndarray:
        radius = self.frequency_window_bands // 2
        pooled = np.empty((3, self.band_count), dtype=np.float64)
        reducer = np.min if use_minimum else np.max
        for band_index in range(self.band_count):
            low = max(0, band_index - radius)
            high = min(self.band_count, band_index + radius + 1)
            pooled[:, band_index] = reducer(memory[:, low:high, :], axis=(1, 2))
        return pooled


def prony_iir_coefficients(
    kernel: np.ndarray,
    *,
    denominator_order: int,
    numerator_order: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Fit ``B(z) / A(z)`` using the reference convolution-matrix convention."""

    values = np.asarray(kernel, dtype=np.float64)
    _positive_integer("denominator_order", denominator_order)
    _positive_integer("numerator_order", numerator_order)
    if values.ndim != 1 or len(values) == 0 or not np.isfinite(values).all():
        raise CustomWaveletSettingsError("wavelet kernel must be a nonempty finite vector")
    if denominator_order + numerator_order >= len(values):
        raise CustomWaveletSettingsError("wavelet kernel is too short for the configured Prony orders")
    convolution = _convolution_matrix(values, denominator_order + 1)
    sample_count = len(values)
    try:
        denominator_tail = np.linalg.lstsq(
            convolution[numerator_order : sample_count + denominator_order - 1, :denominator_order],
            -convolution[numerator_order + 1 : sample_count + denominator_order, 0],
            rcond=None,
        )[0]
    except np.linalg.LinAlgError as error:
        raise CustomWaveletSettingsError("Prony denominator fit failed") from error
    denominator = np.concatenate((np.array([1.0]), denominator_tail))
    numerator = convolution[: numerator_order + 1, : denominator_order + 1] @ denominator
    _validate_iir_bank(numerator[None, :], denominator[None, :])
    return numerator, denominator


def _convolution_matrix(kernel: np.ndarray, columns: int) -> np.ndarray:
    """Construct the causal convolution matrix used by the reference Prony fit."""

    padded = np.concatenate((np.zeros(columns - 1), kernel, np.zeros(columns - 1)))
    stop = len(kernel) + 2 * columns - 2
    matrix = np.empty((len(kernel) + columns - 1, columns), dtype=np.float64)
    for column in range(columns):
        matrix[:, column] = padded[columns - column - 1 : stop - column]
    return matrix


def _validate_iir_bank(numerator: np.ndarray, denominator: np.ndarray) -> None:
    if numerator.ndim != 2 or denominator.ndim != 2 or numerator.shape[0] != denominator.shape[0]:
        raise CustomWaveletSettingsError("Prony coefficient shapes do not define one filter per frequency")
    if not np.isfinite(numerator).all() or not np.isfinite(denominator).all():
        raise CustomWaveletSettingsError("Prony coefficients must be finite")
    if not np.allclose(denominator[:, 0], 1.0, rtol=0.0, atol=0.0):
        raise CustomWaveletSettingsError("Prony denominator coefficients must start with 1")
    for coefficients in denominator:
        poles = np.roots(coefficients)
        if not np.isfinite(poles).all() or np.any(np.abs(poles) >= 1.0):
            raise CustomWaveletSettingsError("Prony IIR fit is unstable")


def _validated_frequencies(values: object, sampling_rate_hz: object) -> tuple[float, ...]:
    sampling_rate = _positive_finite_float("sampling_rate_hz", sampling_rate_hz)
    if isinstance(values, (str, bytes)):
        raise CustomWaveletSettingsError("frequencies_hz must be an array of numbers")
    try:
        raw_frequencies = tuple(values)  # type: ignore[arg-type]
    except TypeError as error:
        raise CustomWaveletSettingsError("frequencies_hz must be an array of numbers") from error
    if any(not _is_finite_real(value) for value in raw_frequencies):
        raise CustomWaveletSettingsError("frequencies_hz values must be finite numbers")
    frequencies = tuple(float(value) for value in raw_frequencies)
    if not frequencies:
        raise CustomWaveletSettingsError("frequencies_hz must contain at least one value")
    if any(not math.isfinite(value) or value <= 0.0 for value in frequencies):
        raise CustomWaveletSettingsError("frequencies_hz values must be finite and positive")
    if any(value >= sampling_rate / 2.0 for value in frequencies):
        raise CustomWaveletSettingsError("frequencies_hz values must be below the Nyquist frequency")
    if any(left >= right for left, right in zip(frequencies[:-1], frequencies[1:])):
        raise CustomWaveletSettingsError("frequencies_hz must be strictly increasing in Hz")
    return frequencies


def _validated_sequence(values: np.ndarray) -> np.ndarray:
    try:
        result = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise SpikeEncodingError("acceleration_g must be numeric") from error
    if result.ndim != 2 or result.shape[1] != 3 or len(result) == 0:
        raise SpikeEncodingError("acceleration_g must have nonempty shape (N, 3)")
    if not np.isfinite(result).all():
        raise SpikeEncodingError("acceleration_g must contain only finite values")
    return result


def _validated_sample(value: np.ndarray) -> np.ndarray:
    try:
        result = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise SpikeEncodingError("acceleration_g sample must be numeric") from error
    if result.shape != (3,) or not np.isfinite(result).all():
        raise SpikeEncodingError("acceleration_g sample must have finite shape (3,)")
    return result


def _odd_window(value: float) -> int:
    window = int(value)
    return window + 1 if window % 2 == 0 else window


def _frequency_window_bands(frequencies_hz: tuple[float, ...], decades: float) -> int:
    if len(frequencies_hz) == 1:
        return 1
    mean_spacing = float(np.diff(np.log10(frequencies_hz)).mean())
    return _odd_window(decades / mean_spacing)


def _frequency_name(value: float) -> str:
    text = format(Decimal(str(value)).normalize(), "f")
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    return text.replace(".", "p")


def _positive_finite_float(name: str, value: object) -> float:
    if not _is_finite_real(value) or float(value) <= 0.0:
        raise CustomWaveletSettingsError(f"{name} must be finite and positive")
    return float(value)


def _positive_integer(name: str, value: object) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise CustomWaveletSettingsError(f"{name} must be a positive integer")


def _is_finite_real(value: object) -> bool:
    return isinstance(value, (int, float, np.integer, np.floating)) and not isinstance(value, bool) and math.isfinite(float(value))
