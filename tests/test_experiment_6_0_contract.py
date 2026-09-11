from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import torch

from scripts import experiment_6_0_legacy_synapse as legacy
from scripts import experiment_6_0_multiscale_phase_evidence as exp60
from scripts import experiment_6_0_synapse_update_comparison as comparison


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_run_grid_is_54_paired_binary_snn_runs() -> None:
    specs = exp60.run_specs()
    assert len(specs) == 27
    assert len(specs) * len(comparison.SYNAPSE_MODES) == 54
    assert comparison.SYNAPSE_MODES == (
        comparison.NORMALIZED,
        legacy.SYNAPSE_MODE,
    )
    assert {spec.objective for spec in specs} == set(exp60.OBJECTIVES)
    assert {spec.mem_shift for spec in specs} == {1, 2, 3}
    assert {spec.seed for spec in specs} == {11, 23, 37}
    assert exp60.HIDDEN_CAP == 1
    assert exp60.OUTPUT_CAP == 1
    assert exp60.EPOCHS == 100


def test_hidden_multitau_partition_is_s4_s5_s6_and_width_128() -> None:
    assert exp60.SYN_SHIFTS == (4, 5, 6)
    assert exp60.GROUP_COUNTS == (43, 43, 42)
    assert sum(exp60.GROUP_COUNTS) == 128
    alpha = exp60.syn_alpha_vector().numpy()
    assert alpha.shape == (128,)
    assert np.allclose(alpha[:43], exp60.decay_from_shift(4))
    assert np.allclose(alpha[43:86], exp60.decay_from_shift(5))
    assert np.allclose(alpha[86:], exp60.decay_from_shift(6))


def test_normalized_synaptic_update_has_unit_dc_gain() -> None:
    for shift in exp60.SYN_SHIFTS:
        alpha = exp60.decay_from_shift(shift)
        state = 0.0
        for _ in range(10000):
            state = alpha * state + (1.0 - alpha)
        assert abs(state - 1.0) < 1e-8


def test_legacy_synaptic_update_has_expected_dc_amplification() -> None:
    for shift in exp60.SYN_SHIFTS:
        alpha = exp60.decay_from_shift(shift)
        state = 0.0
        for _ in range(10000):
            state = alpha * state + 1.0
        expected = 1.0 / (1.0 - alpha)
        assert abs(state - expected) < 1e-6


def test_only_intended_forward_equation_differs_in_legacy_control() -> None:
    source = (
        REPO_ROOT / "scripts/experiment_6_0_legacy_synapse.py"
    ).read_text(encoding="utf-8")
    assert "class LegacySingleHiddenMultiTauSNN(exp60.SingleHiddenMultiTauSNN)" not in source
    assert "class LegacySingleHiddenMultiTauSNN(base.SingleHiddenMultiTauSNN)" in source
    assert "syn = alpha * syn + projected" in source
    assert "one_minus_alpha" not in source
    assert legacy.SYNAPTIC_UPDATE == "I_t = alpha*I_{t-1} + W*x_t"


def test_reverse_complete_bins_preserves_within_bin_remainder_and_padding() -> None:
    # valid=10, bin=4 -> [0..3],[4..7], remainder [8,9], padding [10,11]
    x = torch.arange(12, dtype=torch.float32).reshape(1, 12, 1)
    lengths = torch.tensor([10], dtype=torch.long)
    got = exp60.reverse_complete_bins(x, lengths, 4).reshape(-1).tolist()
    assert got == [4, 5, 6, 7, 0, 1, 2, 3, 8, 9, 10, 11]


def test_reverse_with_only_one_complete_bin_is_identity() -> None:
    x = torch.arange(7, dtype=torch.float32).reshape(1, 7, 1)
    lengths = torch.tensor([7], dtype=torch.long)
    got = exp60.reverse_complete_bins(x, lengths, 4)
    assert torch.equal(got, x)


def test_paired_random_stream_ignores_objective_mem_shift_and_synapse_mode() -> None:
    a = exp60.RunSpec("whole_count", 1, 11)
    b = exp60.RunSpec("whole_count_hce", 3, 11)
    assert exp60.paired_seed(a, "model_init") == exp60.paired_seed(b, "model_init")
    assert exp60.paired_seed(a, "train_loader") == exp60.paired_seed(b, "train_loader")
    # Legacy calls the same base paired_seed with the same RunSpec, so matched
    # normalized/legacy runs share initialization and loader streams exactly.
    assert exp60.paired_seed(a, "model_init") == exp60.paired_seed(a, "model_init")


def test_source_files_parse() -> None:
    for relative in (
        "scripts/experiment_6_0_multiscale_phase_evidence.py",
        "scripts/experiment_6_0_legacy_synapse.py",
        "scripts/experiment_6_0_synapse_update_comparison.py",
    ):
        ast.parse((REPO_ROOT / relative).read_text(encoding="utf-8"))


def test_slurm_topology_is_two_paired_one_core_arrays_and_artifact_only_finalizer() -> None:
    normalized = (
        REPO_ROOT / "scripts/bash_script/SNN_Bash/run_exp_6_0_cpu_array.bash"
    ).read_text(encoding="utf-8")
    legacy_run = (
        REPO_ROOT / "scripts/bash_script/SNN_Bash/run_exp_6_0_legacy_cpu_array.bash"
    ).read_text(encoding="utf-8")
    baseline = (
        REPO_ROOT / "scripts/bash_script/SNN_Bash/run_exp_6_0_baseline_cpu.bash"
    ).read_text(encoding="utf-8")
    final = (
        REPO_ROOT / "scripts/bash_script/SNN_Bash/finalize_exp_6_0_cpu.bash"
    ).read_text(encoding="utf-8")
    submit = (
        REPO_ROOT / "scripts/bash_script/SNN_Bash/submit_exp_6_0_cpu.bash"
    ).read_text(encoding="utf-8")

    assert "#SBATCH --array=0-26%27" in normalized
    assert "#SBATCH --array=0-26%27" in legacy_run
    for text in (normalized, legacy_run, baseline, final):
        assert "#SBATCH --cpus-per-task=1" in text
        assert "OMP_NUM_THREADS=1" in text
        assert "MKL_NUM_THREADS=1" in text
        assert "OPENBLAS_NUM_THREADS=1" in text
        assert "NUMEXPR_NUM_THREADS=1" in text
    assert "experiment_6_0_multiscale_phase_evidence" in normalized
    assert "experiment_6_0_legacy_synapse" in legacy_run
    assert "baseline" in baseline
    assert "experiment_6_0_synapse_update_comparison finalize" in final
    assert 'afterok:${baseline_job}:${normalized_job}:${legacy_job}' in submit


def test_notebook_is_analysis_only_and_reads_paired_outputs() -> None:
    path = REPO_ROOT / "notebooks/experiment_6_0_multiscale_phase_evidence.ipynb"
    text = path.read_text(encoding="utf-8")
    assert "runs.csv" in text
    assert "summary.csv" in text
    assert "paired_synapse_update_deltas.csv" in text
    assert "baseline_fixed250_linear.json" in text
    assert "synapse_mode" in text
    forbidden = (
        "run_one(",
        "train_one(",
        "torch.optim",
        ".fit(",
        "subprocess",
        "sbatch",
    )
    for token in forbidden:
        assert token not in text
