from __future__ import annotations

import pytest

pytest.importorskip("torch")
import torch

from snn.action0_engine import masked_spike_counts, normalized_spike_scores
from snn.action0_losses import MaskedCrossEntropySpkReg


def test_masked_cross_entropy_ignores_changed_padding_steps() -> None:
    labels = torch.tensor([1], dtype=torch.long)
    valid_mask = torch.tensor([[True, True, False, False]])
    valid_output = torch.tensor([[[0.0, 1.0], [1.0, 0.0]]])
    first_output = torch.cat(
        [valid_output, torch.tensor([[[100.0, -100.0], [-100.0, 100.0]]])], dim=1
    )
    second_output = torch.cat(
        [valid_output, torch.tensor([[[-100.0, 100.0], [100.0, -100.0]]])], dim=1
    )
    criterion = MaskedCrossEntropySpkReg()

    first_loss = criterion(first_output, labels, valid_mask)
    second_loss = criterion(second_output, labels, valid_mask)

    torch.testing.assert_close(first_loss, second_loss)


def test_masked_spike_count_and_prediction_ignore_padding() -> None:
    valid_mask = torch.tensor([[True, True, False, False]])
    first_output = torch.tensor([[[0.0, 1.0], [0.0, 1.0], [99.0, 0.0], [99.0, 0.0]]])
    second_output = torch.tensor([[[0.0, 1.0], [0.0, 1.0], [0.0, 99.0], [0.0, 99.0]]])

    first_counts = masked_spike_counts(first_output, valid_mask)
    second_counts = masked_spike_counts(second_output, valid_mask)

    torch.testing.assert_close(first_counts, second_counts)
    assert first_counts.argmax(dim=1).tolist() == second_counts.argmax(dim=1).tolist() == [1]
    zero_scores = normalized_spike_scores(torch.zeros((1, 2)))
    assert torch.isfinite(zero_scores).all()
    torch.testing.assert_close(zero_scores, torch.full((1, 2), 0.5))


def test_spike_regularization_uses_valid_neuron_time() -> None:
    output = torch.zeros((1, 2, 2))
    labels = torch.tensor([0], dtype=torch.long)
    valid_mask = torch.tensor([[True, True]])
    criterion = MaskedCrossEntropySpkReg(spike_regularization=2.0)

    loss = criterion(
        output,
        labels,
        valid_mask,
        spikes=torch.tensor(8.0),
        spike_neuron_count=4,
    )

    expected = torch.log(torch.tensor(2.0)) + 2.0 * (8.0 / (2 * 4))
    torch.testing.assert_close(loss, expected)
