from __future__ import annotations

import numpy as np

from writingring.spike_encoding.contracts import (
    SpikeEncodingOutput,
    SpikeEncodingSequenceResult,
)


def test_public_result_contracts_preserve_declared_fields() -> None:
    sequence = SpikeEncodingSequenceResult(
        values=np.zeros((2, 4)),
        channel_names=("a", "b", "c", "d"),
        representation="test",
        diagnostics={"source": "unit-test"},
    )
    output = SpikeEncodingOutput(
        values=sequence.values,
        encoder_name="dummy",
        representation=sequence.representation,
        channel_names=sequence.channel_names,
        sequence_offsets=np.array([0, 2]),
        sequence_statistics=({"sequence_index": 0},),
        summary={"sample_count": 2},
    )

    assert output.values.shape == (2, 4)
    assert output.channel_names == sequence.channel_names
    assert output.sequence_offsets.tolist() == [0, 2]
