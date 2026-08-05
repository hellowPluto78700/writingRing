"""Public contracts for independent WritingRing spike encoders."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np


class SpikeEncodingError(ValueError):
    """Raised when a spike-encoding contract cannot be satisfied."""


@dataclass(frozen=True, slots=True)
class SpikeEncodingSequenceResult:
    """One encoder result for one continuous input sequence."""

    values: np.ndarray
    channel_names: tuple[str, ...]
    representation: str
    diagnostics: dict[str, object]


@dataclass(frozen=True, slots=True)
class SpikeEncodingOutput:
    """Aggregated encoding values and auditable sequence-level facts."""

    values: np.ndarray
    encoder_name: str
    representation: str
    channel_names: tuple[str, ...]
    sequence_offsets: np.ndarray
    sequence_statistics: tuple[dict[str, object], ...]
    summary: dict[str, object]


class SpikeEncoder(Protocol):
    """Protocol implemented by independently registered spike encoders."""

    name: str
    representation: str

    @property
    def output_channel_names(self) -> tuple[str, ...]:
        """Return deterministic names for each output column."""

    def reset(self) -> None:
        """Reset state before a logically independent sequence."""

    def encode_sequence(
        self,
        acceleration_g: np.ndarray,
    ) -> SpikeEncodingSequenceResult:
        """Encode a finite ``(N, 3)`` g-domain acceleration sequence."""


class SpikeEncoderOutputMetadata(Protocol):
    """Optional extension implemented by encoders that declare output layout."""

    @property
    def output_metadata(self) -> dict[str, object]:
        """Return encoder-specific output metadata such as channel order."""
