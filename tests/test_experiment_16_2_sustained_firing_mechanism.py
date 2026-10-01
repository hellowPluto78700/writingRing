from __future__ import annotations

import math

import numpy as np
import torch

from scripts import analyze_experiment_16_2_sustained_firing_mechanism as diag


def test_tau_table_matches_expected_order_and_dc_gain():
    alpha = np.asarray([0.75, 0.875, 0.9375], dtype=np.float64)
    table = diag._tau_table(alpha, 64.0)
    assert table["tau_group"].tolist() == ["short", "medium", "long"]
    tau = table["tau_ms"].to_numpy()
    assert np.all(np.diff(tau) > 0)
    assert 50 < tau[0] < 60
    assert 110 < tau[1] < 125
    assert 230 < tau[2] < 255


def test_original_l2_replay_matches_soft_reset_equation():
    drive = torch.tensor(
        [[[0.30], [0.30], [0.30], [0.30], [0.30]]],
        dtype=torch.float32,
    )
    lengths = torch.tensor([5])
    alpha = torch.tensor([0.75])
    beta = 0.5
    threshold = 0.5
    out = diag._replay_l2(
        drive,
        lengths,
        alpha,
        beta,
        threshold,
        10.0,
        "original",
    )

    syn = torch.tensor([0.0])
    mem = torch.tensor([0.0])
    expected = []
    for t in range(5):
        syn = alpha * syn + drive[0, t]
        pre = beta * mem + syn
        spike = (pre >= threshold).to(pre.dtype)
        mem = pre - threshold * spike
        expected.append(spike.item())
    assert out["spike"][0, :, 0].tolist() == expected


def test_normalized_synapse_removes_large_dc_gain():
    steps = 64
    drive = torch.full((1, steps, 1), 0.1)
    lengths = torch.tensor([steps])
    alpha = torch.tensor([0.9375])
    kwargs = dict(
        actual_drive=drive,
        lengths=lengths,
        alpha=alpha,
        beta=0.5,
        threshold=0.5,
        slope=10.0,
    )
    original = diag._replay_l2(condition="original", **kwargs)
    normalized = diag._replay_l2(condition="normalized_synapse", **kwargs)

    original_final = float(original["current"][0, steps - 1, 0])
    normalized_final = float(normalized["current"][0, steps - 1, 0])
    assert original_final > 1.0
    assert 0.09 < normalized_final < 0.11
    assert original_final > 10 * normalized_final


def test_hard_reset_zeroes_post_spike_membrane():
    drive = torch.full((1, 8, 1), 0.4)
    lengths = torch.tensor([8])
    alpha = torch.tensor([0.875])
    out = diag._replay_l2(
        drive,
        lengths,
        alpha,
        0.5,
        0.5,
        10.0,
        "hard_reset",
    )
    spikes = out["spike"][0, :, 0].bool()
    membrane = out["membrane"][0, :, 0]
    assert spikes.any()
    assert torch.equal(membrane[spikes], torch.zeros_like(membrane[spikes]))


def test_last_spike_step_is_one_based_and_zero_for_no_spike():
    spikes = np.zeros((2, 6, 2), dtype=np.uint8)
    spikes[0, 0, 0] = 1
    spikes[0, 4, 0] = 1
    spikes[1, 5, 1] = 1
    got = diag._tail_last_spike_step(spikes)
    assert got.tolist() == [[5, 0], [0, 6]]


def test_longest_run_respects_per_sample_valid_length():
    spikes = np.asarray(
        [
            [[1], [1], [0], [1], [1], [1]],
            [[1], [1], [1], [1], [1], [1]],
        ],
        dtype=np.uint8,
    )
    lengths = np.asarray([6, 3])
    got = diag._longest_runs(spikes, lengths)
    assert got[:, 0].tolist() == [3, 3]
