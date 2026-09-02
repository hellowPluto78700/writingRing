from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from scripts import experiment_4_0_fixed250_temporal_snn as exp40
from scripts import experiment_4_1_causal_position_rsnn as exp41
from scripts import experiment_4_2_continuous_recurrent_controls as exp42


def test_exp41_run_matrix_is_small_and_paired_to_exp40_anchors() -> None:
    specs = exp41.run_specs()
    assert len(specs) == 20 == exp41.EXPECTED_NEW_RUNS
    assert {(spec.hidden_width, spec.tau_mem_ms) for spec in specs} == {
        (64, 1000.0),
        (128, 250.0),
    }
    assert {spec.position_encoding for spec in specs} == {"scalar", "onehot"}
    assert {spec.seed for spec in specs} == {11, 23, 37, 53, 71}
    assert len({spec.key for spec in specs}) == len(specs)


def test_exp41_scalar_position_is_causal_and_zero_after_endpoint() -> None:
    X = np.zeros((2, 4, exp40.EVENT_CHANNELS), dtype=np.float32)
    valid_bins = np.array([2, 4])
    out = exp41.add_causal_position_features(X, valid_bins, "scalar")
    assert out.shape == (2, 4, exp40.EVENT_CHANNELS + 1)
    assert np.allclose(out[0, :, -1], [0.25, 0.50, 0.0, 0.0])
    assert np.allclose(out[1, :, -1], [0.25, 0.50, 0.75, 1.0])
    assert np.array_equal(out[:, :, : exp40.EVENT_CHANNELS], X)


def test_exp41_onehot_position_is_absolute_slot_and_zero_after_endpoint() -> None:
    X = np.zeros((1, 4, exp40.EVENT_CHANNELS), dtype=np.float32)
    out = exp41.add_causal_position_features(X, np.array([2]), "onehot")
    pos = out[0, :, exp40.EVENT_CHANNELS :]
    assert out.shape == (1, 4, exp40.EVENT_CHANNELS + 4)
    assert np.array_equal(pos[0], [1.0, 0.0, 0.0, 0.0])
    assert np.array_equal(pos[1], [0.0, 1.0, 0.0, 0.0])
    assert np.array_equal(pos[2:], np.zeros((2, 4), dtype=np.float32))


def test_exp41_parameter_count_includes_only_position_input_extension() -> None:
    scalar = exp41.parameter_counts(64, 12, "scalar", 16)
    onehot = exp41.parameter_counts(64, 12, "onehot", 16)
    assert scalar["input_hidden"] == 31 * 64
    assert onehot["input_hidden"] == 46 * 64
    assert scalar["recurrent"] == onehot["recurrent"] == 64 * 64


def test_exp42_run_matrix_covers_model_width_objective_seed_grid() -> None:
    specs = exp42.run_specs()
    assert len(specs) == 40 == exp42.EXPECTED_RUNS
    assert {spec.model_type for spec in specs} == {"ff_ann", "rnn"}
    assert {spec.hidden_width for spec in specs} == {64, 128}
    assert {spec.objective for spec in specs} == {"endpoint_ce", "sum_logits_ce"}
    assert {spec.seed for spec in specs} == {11, 23, 37, 53, 71}
    assert len({spec.key for spec in specs}) == len(specs)


def test_exp42_rnn_parameter_count_matches_dense_rsnn_topology() -> None:
    for width in (64, 128):
        rnn = exp42.parameter_counts(width, True, 12)
        rsnn = exp40.parameter_counts(width, True, 12)
        ff_ann = exp42.parameter_counts(width, False, 12)
        ff_snn = exp40.parameter_counts(width, False, 12)
        assert rnn == rsnn
        assert ff_ann == ff_snn


def test_exp42_readouts_respect_valid_endpoint_and_sum_window() -> None:
    logits = torch.tensor(
        [
            [[1.0, 0.0], [2.0, 1.0], [3.0, 4.0], [5.0, 6.0]],
            [[1.0, 2.0], [4.0, 3.0], [7.0, 8.0], [9.0, 10.0]],
        ]
    )
    valid_bins = torch.tensor([2, 3])
    endpoint = exp42.valid_endpoint_logits(logits, valid_bins)
    valid_sum = exp42.valid_sum_logits(logits, valid_bins)
    full_sum = exp42.full_sum_logits(logits)
    assert torch.equal(endpoint, torch.tensor([[2.0, 1.0], [7.0, 8.0]]))
    assert torch.equal(valid_sum, torch.tensor([[3.0, 1.0], [12.0, 13.0]]))
    assert torch.equal(full_sum, torch.tensor([[11.0, 11.0], [21.0, 23.0]]))


def test_exp42_ff_and_rnn_differ_only_by_recurrent_path_at_topology_level() -> None:
    ff = exp42.ContinuousTemporalDecoder("ff_ann", 64, 12)
    rnn = exp42.ContinuousTemporalDecoder("rnn", 64, 12)
    assert ff.recurrent is None
    assert rnn.recurrent is not None
    assert tuple(rnn.recurrent.weight.shape) == (64, 64)
    assert tuple(ff.input_hidden.weight.shape) == tuple(rnn.input_hidden.weight.shape)
    assert tuple(ff.hidden_output.weight.shape) == tuple(rnn.hidden_output.weight.shape)


def test_exp42_ff_ann_zero_input_has_zero_padded_tail() -> None:
    model = exp42.ContinuousTemporalDecoder("ff_ann", 64, 12)
    X = torch.zeros(2, 5, exp40.EVENT_CHANNELS)
    logits, hidden = model.forward_trajectory(X)
    assert torch.equal(logits, torch.zeros_like(logits))
    assert torch.equal(hidden, torch.zeros_like(hidden))


def test_multi_cpu_scripts_respect_combined_50_task_cap() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    exp41_array = (repo_root / "scripts/bash_script/SNN_Bash/run_exp_4_1_cpu_array.bash").read_text()
    exp42_array = (repo_root / "scripts/bash_script/SNN_Bash/run_exp_4_2_cpu_array.bash").read_text()
    combined = (repo_root / "scripts/bash_script/SNN_Bash/submit_exp_4_1_4_2_cpu.bash").read_text()
    assert "#SBATCH --array=0-19%20" in exp41_array
    assert "#SBATCH --array=0-39%30" in exp42_array
    assert "Combined experiment concurrency cap: 50 CPU tasks" in combined
    for script in (exp41_array, exp42_array):
        assert "OMP_NUM_THREADS=1" in script
        assert "MKL_NUM_THREADS=1" in script
        assert "OPENBLAS_NUM_THREADS=1" in script
        assert "NUMEXPR_NUM_THREADS=1" in script
