from __future__ import annotations

import pytest

pytest.importorskip("torch")
pytest.importorskip("snntorch")
import torch

from snn.utils_architectures import SynNet


def _expected_alphas(shifts: list[int], neurons_per_shift: int) -> torch.Tensor:
    return torch.tensor(
        [1 - 2 ** (-shift) for shift in shifts for _ in range(neurons_per_shift)],
        dtype=torch.float32,
    )


def test_synnet_preserves_heterogeneous_shift_alphas_and_masked_spike_total() -> None:
    model = SynNet(
        inputSize=15,
        outputSize=3,
        hiddenSizes=[24, 24, 24],
        shiftSyn=2,
        shiftMem=1,
    )

    torch.testing.assert_close(torch.as_tensor(model.lif1.beta), torch.tensor(0.5))
    torch.testing.assert_close(torch.as_tensor(model.lif2.beta), torch.tensor(0.5))
    torch.testing.assert_close(torch.as_tensor(model.lif3.beta), torch.tensor(0.5))
    torch.testing.assert_close(
        torch.as_tensor(model.lif1.alpha), _expected_alphas([2, 3], 12)
    )
    torch.testing.assert_close(
        torch.as_tensor(model.lif2.alpha), _expected_alphas([2, 3, 4, 5], 6)
    )
    torch.testing.assert_close(
        torch.as_tensor(model.lif3.alpha), _expected_alphas(list(range(2, 10)), 3)
    )

    output = model(torch.ones((2, 4, 15)), valid_mask=torch.zeros((2, 4), dtype=torch.bool))
    assert output.shape == (2, 4, 3)
    assert model.spkTotal is not None
    torch.testing.assert_close(model.spkTotal, torch.tensor(0.0))


def test_sample_frequency_does_not_change_alpha_or_beta() -> None:
    models = [
        SynNet(15, outputSize=2, hiddenSizes=[24, 24, 24], sampleFreq=frequency, shiftSyn=2, shiftMem=1)
        for frequency in (64, 100, 200)
    ]
    reference = models[0]
    for model in models[1:]:
        for layer_name in ("lif1", "lif2", "lif3", "lif4"):
            torch.testing.assert_close(
                torch.as_tensor(getattr(model, layer_name).alpha),
                torch.as_tensor(getattr(reference, layer_name).alpha),
            )
            torch.testing.assert_close(
                torch.as_tensor(getattr(model, layer_name).beta),
                torch.as_tensor(getattr(reference, layer_name).beta),
            )
