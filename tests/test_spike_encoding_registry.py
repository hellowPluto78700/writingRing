from __future__ import annotations

import numpy as np
import pytest

from writingring.spike_encoding.contracts import (
    SpikeEncodingError,
    SpikeEncodingSequenceResult,
)
from writingring.spike_encoding.registry import (
    available_encoders,
    create_encoder,
    register_encoder,
)


class _DummyEncoder:
    name = "phase1-dummy"
    representation = "dummy_values"

    @property
    def output_channel_names(self) -> tuple[str, ...]:
        return ("dummy",)

    def reset(self) -> None:
        return None

    def encode_sequence(self, acceleration_g: np.ndarray) -> SpikeEncodingSequenceResult:
        return SpikeEncodingSequenceResult(
            values=acceleration_g[:, :1],
            channel_names=self.output_channel_names,
            representation=self.representation,
            diagnostics={},
        )


def test_registry_creates_registered_dummy_without_encoder_specific_branches() -> None:
    register_encoder("phase1-dummy", lambda settings: _DummyEncoder(), replace=True)

    encoder = create_encoder("phase1-dummy", settings={"unused": True})

    assert encoder.name == "phase1-dummy"
    assert "phase1-dummy" in available_encoders()


def test_unknown_encoder_is_rejected_with_available_names() -> None:
    with pytest.raises(SpikeEncodingError, match="unknown spike encoder"):
        create_encoder("missing", settings={})
