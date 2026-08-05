from __future__ import annotations

import numpy as np
import pytest

from writingring.spike_encoding.contracts import (
    SpikeEncodingError,
    SpikeEncodingSequenceResult,
)
from writingring.spike_encoding.runner import run_spike_encoder


class _StatefulDummyEncoder:
    name = "dummy"
    representation = "signed_dummy"

    def __init__(self) -> None:
        self.reset_count = 0
        self.state = 0

    @property
    def output_channel_names(self) -> tuple[str, ...]:
        return ("event_a", "event_b")

    def reset(self) -> None:
        self.reset_count += 1
        self.state = 0

    def encode_sequence(self, acceleration_g: np.ndarray) -> SpikeEncodingSequenceResult:
        self.state += 1
        values = np.column_stack((acceleration_g[:, 0], -acceleration_g[:, 1]))
        return SpikeEncodingSequenceResult(
            values=values,
            channel_names=self.output_channel_names,
            representation=self.representation,
            diagnostics={"state": self.state},
        )


def test_runner_resets_at_every_sequence_and_supports_dynamic_channel_count() -> None:
    encoder = _StatefulDummyEncoder()
    acceleration = np.array(
        [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]
    )

    result = run_spike_encoder(
        encoder,
        acceleration_g=acceleration,
        sequence_offsets=np.array([0, 1, 3], dtype=np.int64),
    )

    assert encoder.reset_count == 2
    assert result.values.shape == (3, 2)
    assert result.sequence_offsets.tolist() == [0, 1, 3]
    assert result.sequence_statistics[0]["positive_event_count"] == 1
    assert result.sequence_statistics[1]["negative_event_count"] == 2
    assert result.summary["state_reset_boundary"] == "sequence"


def test_runner_rejects_encoder_shape_changes() -> None:
    class BadEncoder(_StatefulDummyEncoder):
        def encode_sequence(self, acceleration_g: np.ndarray) -> SpikeEncodingSequenceResult:
            result = super().encode_sequence(acceleration_g)
            return SpikeEncodingSequenceResult(
                values=result.values[:, :1],
                channel_names=result.channel_names,
                representation=result.representation,
                diagnostics={},
            )

    with pytest.raises(SpikeEncodingError, match="encoder output must have shape"):
        run_spike_encoder(
            BadEncoder(),
            acceleration_g=np.ones((2, 3)),
            sequence_offsets=np.array([0, 2], dtype=np.int64),
        )
