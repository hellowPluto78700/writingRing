"""Schema contracts and event decoding for SpikeIMU reconstruction."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Final

import numpy as np


SIGNED_WAVELET_SCHEMA: Final[str] = "signed_wavelet_events_plus_imu_v1"
POLARITY_SPLIT_WAVELET_SCHEMA: Final[str] = (
    "polarity_split_wavelet_events_plus_imu_v1"
)
CANONICAL_SIGNED_EVENT_CHANNEL_COUNT: Final[int] = 15


@dataclass(frozen=True, slots=True)
class EventSchemaContract:
    """Describe stored SpikeIMU channels and their canonical event decoder."""

    feature_schema: str
    total_channel_count: int
    stored_event_channel_count: int
    event_channel_order: str
    polarity_decoding: str

    @property
    def canonical_signed_event_channel_count(self) -> int:
        return CANONICAL_SIGNED_EVENT_CHANNEL_COUNT


SIGNED_EVENT_CONTRACT: Final[EventSchemaContract] = EventSchemaContract(
    feature_schema=SIGNED_WAVELET_SCHEMA,
    total_channel_count=21,
    stored_event_channel_count=15,
    event_channel_order="axis_major_frequency_minor",
    polarity_decoding="identity",
)

POLARITY_SPLIT_EVENT_CONTRACT: Final[EventSchemaContract] = EventSchemaContract(
    feature_schema=POLARITY_SPLIT_WAVELET_SCHEMA,
    total_channel_count=36,
    stored_event_channel_count=30,
    event_channel_order=(
        "axis_major_frequency_minor_pairwise_positive_negative"
    ),
    polarity_decoding="pairwise_positive_minus_negative",
)

EVENT_SCHEMA_CONTRACTS: Final[dict[str, EventSchemaContract]] = {
    SIGNED_WAVELET_SCHEMA: SIGNED_EVENT_CONTRACT,
    POLARITY_SPLIT_WAVELET_SCHEMA: POLARITY_SPLIT_EVENT_CONTRACT,
}


def _parse_channel_count(value: object, *, context: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{context}: channel_count must be an integer")
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{context}: invalid channel_count={value!r}") from error
    if not math.isfinite(number) or not number.is_integer():
        raise ValueError(f"{context}: channel_count must be an integer")
    return int(number)


def resolve_event_schema_contract(
    feature_schema: object,
    channel_count: object,
    *,
    context: str = "SpikeIMU",
) -> EventSchemaContract:
    """Resolve and validate a declared SpikeIMU schema.

    A missing schema remains compatible only with the historical 21-channel
    signed layout. A 36-channel array must declare the polarity-split schema;
    its meaning is never inferred from channel count alone.
    """

    declared_count = _parse_channel_count(channel_count, context=context)
    if feature_schema is None:
        if declared_count == SIGNED_EVENT_CONTRACT.total_channel_count:
            return SIGNED_EVENT_CONTRACT
        raise ValueError(
            f"{context}: feature_schema is required for channel_count={declared_count}"
        )

    if not isinstance(feature_schema, str):
        raise ValueError(f"{context}: feature_schema must be a string")
    contract = EVENT_SCHEMA_CONTRACTS.get(feature_schema)
    if contract is None:
        raise ValueError(
            f"{context}: unsupported feature_schema={feature_schema!r}; "
            f"expected one of {sorted(EVENT_SCHEMA_CONTRACTS)}"
        )
    if declared_count != contract.total_channel_count:
        raise ValueError(
            f"{context}: feature_schema={feature_schema!r} requires "
            f"channel_count={contract.total_channel_count}, got {declared_count}"
        )
    return contract


def validate_spike_imu_array(
    spike_imu: np.ndarray,
    contract: EventSchemaContract,
    *,
    context: str = "SpikeIMU",
) -> np.ndarray:
    """Validate an array against a resolved schema and return it as an array."""

    values = np.asarray(spike_imu)
    if values.ndim < 2 or values.shape[-1] != contract.total_channel_count:
        raise ValueError(
            f"{context}: expected final shape (..., {contract.total_channel_count}), "
            f"got {values.shape}"
        )
    if not np.issubdtype(values.dtype, np.number):
        raise ValueError(f"{context}: SpikeIMU values must be numeric")
    if not np.isfinite(values).all():
        raise ValueError(f"{context}: SpikeIMU values contain non-finite values")
    return values


def extract_signed_event_channels(
    spike_imu: np.ndarray,
    contract: EventSchemaContract,
    *,
    context: str = "SpikeIMU",
) -> np.ndarray:
    """Decode stored event channels into canonical signed ``(..., 15)`` events."""

    values = validate_spike_imu_array(spike_imu, contract, context=context)
    stored_events = values[..., : contract.stored_event_channel_count]
    if contract.feature_schema == SIGNED_WAVELET_SCHEMA:
        return np.array(
            stored_events[..., :CANONICAL_SIGNED_EVENT_CHANNEL_COUNT],
            copy=True,
        )
    if contract.feature_schema == POLARITY_SPLIT_WAVELET_SCHEMA:
        positive = stored_events[..., 0::2]
        negative_absolute = stored_events[..., 1::2]
        return positive - negative_absolute
    raise ValueError(f"{context}: no decoder registered for {contract.feature_schema!r}")


def event_contract_metadata(contract: EventSchemaContract) -> dict[str, object]:
    """Return stable provenance fields for reconstruction metadata."""

    return {
        "input_feature_schema": contract.feature_schema,
        "stored_event_channel_count": contract.stored_event_channel_count,
        "canonical_signed_event_channel_count": (
            contract.canonical_signed_event_channel_count
        ),
        "event_channel_order": contract.event_channel_order,
        "polarity_decoding": contract.polarity_decoding,
    }


__all__ = [
    "CANONICAL_SIGNED_EVENT_CHANNEL_COUNT",
    "EVENT_SCHEMA_CONTRACTS",
    "EventSchemaContract",
    "POLARITY_SPLIT_EVENT_CONTRACT",
    "POLARITY_SPLIT_WAVELET_SCHEMA",
    "SIGNED_EVENT_CONTRACT",
    "SIGNED_WAVELET_SCHEMA",
    "event_contract_metadata",
    "extract_signed_event_channels",
    "resolve_event_schema_contract",
    "validate_spike_imu_array",
]
