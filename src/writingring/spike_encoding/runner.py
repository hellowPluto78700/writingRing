"""Generic state-boundary runner for independent spike encoders."""

from __future__ import annotations

import numpy as np

from writingring.spike_encoding.contracts import (
    SpikeEncoder,
    SpikeEncodingError,
    SpikeEncodingOutput,
    SpikeEncodingSequenceResult,
)
from writingring.spike_encoding.io import validate_sequence_offsets


def run_spike_encoder(
    encoder: SpikeEncoder,
    *,
    acceleration_g: np.ndarray,
    sequence_offsets: np.ndarray,
    sequence_boundary_semantics: str = "sequence",
) -> SpikeEncodingOutput:
    """Reset and run one stateful encoder for each explicit input sequence."""

    values = _validated_acceleration(acceleration_g)
    offsets = validate_sequence_offsets(sequence_offsets, sample_count=len(values))
    if sequence_boundary_semantics not in {"sequence", "recording"}:
        raise SpikeEncodingError(
            "sequence_boundary_semantics must be 'sequence' or 'recording'"
        )
    expected_names = _encoder_channel_names(encoder)
    outputs: list[np.ndarray] = []
    statistics: list[dict[str, object]] = []
    representation: str | None = None
    for sequence_index, (start, stop) in enumerate(zip(offsets[:-1], offsets[1:])):
        encoder.reset()
        result = encoder.encode_sequence(values[int(start):int(stop)])
        encoded = _validated_sequence_result(
            result,
            sample_count=int(stop - start),
            channel_names=expected_names,
        )
        if result.representation != encoder.representation:
            raise SpikeEncodingError(
                "encoder result representation does not match encoder representation"
            )
        if representation is None:
            representation = result.representation
        elif result.representation != representation:
            raise SpikeEncodingError("encoder representation changed between sequences")
        outputs.append(encoded)
        statistics.append(_sequence_statistics(sequence_index, int(start), int(stop), encoded))
    combined = np.concatenate(outputs, axis=0)
    if len(combined) != len(values):
        raise SpikeEncodingError("encoder output sample count does not match input")
    combined.setflags(write=False)
    summary = {
        "input_sample_count": len(values),
        "output_sample_count": len(combined),
        "channel_count": combined.shape[1],
        "sequence_count": len(statistics),
        "sequence_boundary_semantics": sequence_boundary_semantics,
        "state_reset_boundary": sequence_boundary_semantics,
    }
    return SpikeEncodingOutput(
        values=combined,
        encoder_name=encoder.name,
        representation=representation or encoder.representation,
        channel_names=expected_names,
        sequence_offsets=offsets,
        sequence_statistics=tuple(statistics),
        summary=summary,
    )


def _validated_acceleration(acceleration_g: np.ndarray) -> np.ndarray:
    try:
        values = np.asarray(acceleration_g, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise SpikeEncodingError("acceleration_g must be numeric") from error
    if values.ndim != 2 or values.shape[1] != 3 or len(values) == 0:
        raise SpikeEncodingError("acceleration_g must have nonempty shape (N, 3)")
    if not np.isfinite(values).all():
        raise SpikeEncodingError("acceleration_g must contain only finite values")
    return values.copy()


def _encoder_channel_names(encoder: SpikeEncoder) -> tuple[str, ...]:
    if not isinstance(getattr(encoder, "name", None), str) or not encoder.name:
        raise SpikeEncodingError("encoder name must be a nonempty string")
    if not isinstance(getattr(encoder, "representation", None), str) or not encoder.representation:
        raise SpikeEncodingError("encoder representation must be a nonempty string")
    names = encoder.output_channel_names
    if not isinstance(names, tuple) or not names:
        raise SpikeEncodingError("encoder output_channel_names must be a nonempty tuple")
    if any(not isinstance(name, str) or not name for name in names):
        raise SpikeEncodingError("encoder output channel names must be nonempty strings")
    return names


def _validated_sequence_result(
    result: SpikeEncodingSequenceResult,
    *,
    sample_count: int,
    channel_names: tuple[str, ...],
) -> np.ndarray:
    if not isinstance(result, SpikeEncodingSequenceResult):
        raise SpikeEncodingError("encoder must return SpikeEncodingSequenceResult")
    if result.channel_names != channel_names:
        raise SpikeEncodingError("encoder output channel names changed between sequences")
    if not isinstance(result.representation, str) or not result.representation:
        raise SpikeEncodingError("encoder result representation must be a nonempty string")
    try:
        values = np.asarray(result.values, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise SpikeEncodingError("encoder output must be numeric") from error
    if values.ndim != 2 or values.shape != (sample_count, len(channel_names)):
        raise SpikeEncodingError(
            "encoder output must have shape "
            f"({sample_count}, {len(channel_names)})"
        )
    if not np.isfinite(values).all():
        raise SpikeEncodingError("encoder output must contain only finite values")
    return values.copy()


def _sequence_statistics(
    sequence_index: int,
    start: int,
    stop: int,
    values: np.ndarray,
) -> dict[str, object]:
    nonzero = int(np.count_nonzero(values))
    positive = int(np.count_nonzero(values > 0.0))
    negative = int(np.count_nonzero(values < 0.0))
    return {
        "sequence_index": sequence_index,
        "start_offset": start,
        "stop_offset_exclusive": stop,
        "sample_count": stop - start,
        "nonzero_event_count": nonzero,
        "positive_event_count": positive,
        "negative_event_count": negative,
        "event_density": nonzero / values.size,
    }
