from __future__ import annotations

import pytest

pytest.importorskip("torch")
import torch
from torch import nn

from snn.action0_engine import masked_spike_counts, normalized_spike_scores, run_epoch
from snn.action0_losses import MaskedCrossEntropySpkReg


class _SequenceModel(nn.Module):
    def __init__(self, outputs: list[torch.Tensor], spikes: list[float]) -> None:
        super().__init__()
        self.outputs = outputs
        self.spikes = spikes
        self.forward_index = 0
        self.spike_neuron_count = 2
        self.spkTotal: torch.Tensor | None = None

    def forward(self, inputs: torch.Tensor, *, valid_mask: torch.Tensor) -> torch.Tensor:
        del valid_mask
        output = self.outputs[self.forward_index].to(inputs.device)
        self.spkTotal = torch.tensor(self.spikes[self.forward_index], device=inputs.device)
        self.forward_index += 1
        return output


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


def test_run_epoch_rejects_nonexact_output_class_count() -> None:
    batch = (
        torch.zeros((1, 2, 1)),
        torch.tensor([0]),
        torch.tensor([[True, True]]),
    )
    model = _SequenceModel([torch.zeros((1, 2, 3))], [0.0])

    with pytest.raises(AssertionError, match="exact shape"):
        run_epoch(
            model,
            [batch],
            MaskedCrossEntropySpkReg(),
            split="val",
            expected_num_classes=2,
        )


def test_run_epoch_loss_is_weighted_by_global_valid_timestep_count() -> None:
    first_batch = (
        torch.zeros((1, 2, 1)),
        torch.tensor([0]),
        torch.tensor([[True, False]]),
    )
    second_batch = (
        torch.zeros((1, 4, 1)),
        torch.tensor([0]),
        torch.tensor([[True, True, True, False]]),
    )
    first_output = torch.tensor([[[0.0, 0.0], [100.0, -100.0]]])
    second_output = torch.tensor(
        [[[0.0, 1.0], [0.0, 1.0], [0.0, 1.0], [100.0, -100.0]]]
    )
    model = _SequenceModel([first_output, second_output], [2.0, 12.0])
    criterion = MaskedCrossEntropySpkReg(spike_regularization=0.5)

    metrics = run_epoch(
        model,
        [first_batch, second_batch],
        criterion,
        split="val",
        expected_num_classes=2,
    )

    first_loss = criterion(first_output, first_batch[1], first_batch[2], spikes=torch.tensor(2.0), spike_neuron_count=2)
    second_loss = criterion(
        second_output,
        second_batch[1],
        second_batch[2],
        spikes=torch.tensor(12.0),
        spike_neuron_count=2,
    )
    expected = (first_loss * 1 + second_loss * 3) / 4
    torch.testing.assert_close(torch.tensor(metrics["loss"]), expected)
