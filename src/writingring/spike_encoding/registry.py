"""Extensible registry for WritingRing spike encoders."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import cast

from writingring.spike_encoding.contracts import SpikeEncoder, SpikeEncodingError
from writingring.spike_encoding.encoders.custom_wavelet import CustomWaveletEncoder


EncoderFactory = Callable[[dict[str, object]], SpikeEncoder]
_ENCODER_REGISTRY: dict[str, EncoderFactory] = {}
# Public registry for encoder modules to register their factories.  Mutation
# should go through register_encoder() so duplicate-name validation remains
# centralized.
ENCODER_REGISTRY = _ENCODER_REGISTRY


def register_encoder(
    name: str,
    factory: EncoderFactory,
    *,
    replace: bool = False,
) -> None:
    """Register one encoder factory without coupling the runner to its type."""

    _validated_name(name)
    if not callable(factory):
        raise SpikeEncodingError("encoder factory must be callable")
    if not isinstance(replace, bool):
        raise SpikeEncodingError("replace must be a boolean")
    if name in _ENCODER_REGISTRY and not replace:
        raise SpikeEncodingError(f"encoder is already registered: {name!r}")
    _ENCODER_REGISTRY[name] = factory


def available_encoders() -> tuple[str, ...]:
    """Return encoder names in deterministic lexical order."""

    return tuple(sorted(_ENCODER_REGISTRY))


def create_encoder(
    name: str,
    *,
    settings: dict[str, object],
) -> SpikeEncoder:
    """Create one registered encoder from its encoder-private settings."""

    _validated_name(name)
    if not isinstance(settings, dict):
        raise SpikeEncodingError("encoder settings must be a JSON object")
    factory = _ENCODER_REGISTRY.get(name)
    if factory is None:
        available = ", ".join(available_encoders()) or "none"
        raise SpikeEncodingError(
            f"unknown spike encoder {name!r}; available encoders: {available}"
        )
    encoder = factory(dict(settings))
    _validate_encoder_instance(encoder, expected_name=name)
    return encoder


def _validated_name(name: str) -> None:
    if not isinstance(name, str) or not name or name != name.strip():
        raise SpikeEncodingError("encoder name must be a nonempty trimmed string")


def _validate_encoder_instance(encoder: object, *, expected_name: str) -> None:
    candidate = cast(SpikeEncoder, encoder)
    if getattr(candidate, "name", None) != expected_name:
        raise SpikeEncodingError(
            f"encoder factory for {expected_name!r} returned a mismatched name"
        )
    if not isinstance(getattr(candidate, "representation", None), str) or not candidate.representation:
        raise SpikeEncodingError("encoder representation must be a nonempty string")
    for method_name in ("reset", "encode_sequence"):
        if not callable(getattr(candidate, method_name, None)):
            raise SpikeEncodingError(f"encoder must define callable {method_name}()")
    channel_names = candidate.output_channel_names
    if not isinstance(channel_names, tuple) or not channel_names:
        raise SpikeEncodingError("encoder output_channel_names must be a nonempty tuple")
    if any(not isinstance(item, str) or not item for item in channel_names):
        raise SpikeEncodingError("encoder output channel names must be nonempty strings")


register_encoder("custom-wavelet", CustomWaveletEncoder.from_settings)
