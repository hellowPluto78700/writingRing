from __future__ import annotations

import numpy as np
import torch

from scripts import experiment_4_0_fixed250_temporal_snn as exp40


def test_run_matrix_is_complete_and_array_sized() -> None:
    specs = exp40.run_specs()
    assert len(specs) == 120 == exp40.EXPECTED_SNN_RUNS
    assert {spec.architecture for spec in specs} == {"ff", "rsnn"}
    assert {spec.hidden_width for spec in specs} == {32, 64, 128}
    assert {spec.tau_mem_ms for spec in specs} == {250.0, 500.0, 1000.0, 2000.0}
    assert {spec.seed for spec in specs} == {11, 23, 37, 53, 71}
    assert len({spec.key for spec in specs}) == len(specs)


def test_fixed250_channel_count_masks_tail_and_keeps_partial_final_bin() -> None:
    X = np.zeros((2, 8, exp40.EVENT_CHANNELS), dtype=np.float32)
    X[0, :, 0] = 1.0
    X[1, :, 0] = 2.0
    counts, valid = exp40.fixed250_channel_counts(X, np.array([5, 8]), bin_steps=4)
    assert counts.shape == (2, 2, exp40.EVENT_CHANNELS)
    assert valid.tolist() == [2, 2]
    assert counts[0, 0, 0] == 4.0
    assert counts[0, 1, 0] == 1.0
    assert counts[1, 0, 0] == 8.0
    assert counts[1, 1, 0] == 8.0


def test_scaler_uses_only_valid_training_bins_and_preserves_zero() -> None:
    X = np.zeros((2, 3, exp40.EVENT_CHANNELS), dtype=np.float32)
    X[0, 0, 0] = 1.0
    X[0, 1, 0] = 3.0
    X[0, 2, 0] = 1000.0  # invalid and must not affect fit
    X[1, 0, 0] = 5.0
    X[1, 1, 0] = 1000.0  # invalid
    X[1, 2, 0] = 1000.0  # invalid
    scale = exp40.fit_zero_preserving_channel_scale(X, np.array([2, 1]))
    expected = np.std(np.array([1.0, 3.0, 5.0], dtype=np.float32))
    assert np.isclose(scale[0], expected)
    transformed = exp40.apply_channel_scale(np.zeros_like(X), scale)
    assert np.array_equal(transformed, np.zeros_like(X))


def test_valid_and_full_counts_use_same_trajectory_with_different_windows() -> None:
    spikes = torch.tensor(
        [
            [
                [1.0, 0.0],
                [0.0, 1.0],
                [1.0, 1.0],
                [1.0, 0.0],
            ]
        ]
    )
    valid_bins = torch.tensor([2])
    valid = exp40.valid_whole_count(spikes, valid_bins)
    full = exp40.full_whole_count(spikes)
    assert torch.equal(valid, torch.tensor([[1.0, 1.0]]))
    assert torch.equal(full, torch.tensor([[3.0, 2.0]]))
    assert torch.equal(full - valid, torch.tensor([[2.0, 1.0]]))


def test_valid_final_membrane_indexes_endpoint_without_freezing_state() -> None:
    mem = torch.tensor(
        [
            [[1.0, 10.0], [2.0, 20.0], [3.0, 30.0], [4.0, 40.0]],
            [[5.0, 50.0], [6.0, 60.0], [7.0, 70.0], [8.0, 80.0]],
        ]
    )
    valid_bins = torch.tensor([2, 3])
    endpoint = exp40.valid_final_membrane(mem, valid_bins)
    assert torch.equal(endpoint, torch.tensor([[2.0, 20.0], [7.0, 70.0]]))
    assert torch.equal(mem[:, -1], torch.tensor([[4.0, 40.0], [8.0, 80.0]]))


def test_recurrent_parameter_count_is_hidden_width_squared() -> None:
    ff32 = exp40.parameter_counts(32, False, 12)
    rsnn32 = exp40.parameter_counts(32, True, 12)
    ff64 = exp40.parameter_counts(64, False, 12)
    rsnn128 = exp40.parameter_counts(128, True, 12)
    assert ff32["total"] == 30 * 32 + 32 * 12
    assert rsnn32["recurrent"] == 32 * 32
    assert rsnn32["total"] == ff32["total"] + 32 * 32
    assert ff64["total"] == 2688
    assert rsnn128["total"] == 21760


def test_rsnn_contains_recurrent_matrix_but_ff_does_not() -> None:
    ff = exp40.TemporalDecoderSNN("ff", 32, 250.0, 12)
    rsnn = exp40.TemporalDecoderSNN("rsnn", 32, 250.0, 12)
    assert ff.recurrent is None
    assert rsnn.recurrent is not None
    assert tuple(rsnn.recurrent.weight.shape) == (32, 32)
