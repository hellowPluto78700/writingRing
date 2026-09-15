from pathlib import Path

import numpy as np
import torch

from scripts import experiment_7_2_6_4_output_interface_decomposition as exp7264


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_core_contract() -> None:
    assert exp7264.ARCHITECTURE == "234x234"
    assert exp7264.REGULARIZATION == "task_only"
    assert exp7264.SEEDS == (11, 23, 37)
    assert exp7264.OUTPUT_ALPHA == 0.0
    assert exp7264.LIF_BETA == 0.5
    assert exp7264.OUTPUT_CAP == 1
    assert exp7264.FLUSH_STEPS == (0, 1, 2, 4, 8, 16, 32, 64, 128)
    assert exp7264.SOFTMAX_TEMPS == (0.25, 0.5, 1.0, 2.0)
    assert exp7264.C_OBJECTIVES == ("raw_wcce", "softmax_vote_wcce", "tsce")
    assert exp7264.CE_GAINS == (1.0, 2.0, 5.0, 10.0)
    assert len(exp7264.source_specs()) == 3
    assert len(exp7264.c_specs()) == 9
    assert len(exp7264.d_specs()) == 27


def test_pairing_reuses_exp7263_baseline_seed() -> None:
    for role in ("model_init", "train_loader", "val_loader"):
        assert exp7264._pair_seed(11, role) == exp7264.exp7263._head_pair_seed(11, role)


def test_zero_input_flush_and_infinite_if_ceiling() -> None:
    counts = np.array([[2.0, 0.0]])
    residual = np.array([[2.4, -3.0]])
    finite, mem, extra = exp7264._flush_from_state(counts, residual, beta=1.0, steps=1)
    assert np.allclose(finite, [[3.0, 0.0]])
    assert np.allclose(extra, [[1.0, 0.0]])
    assert np.allclose(mem, [[1.4, -3.0]])
    infinite = exp7264._infinite_if_counts(counts, residual)
    assert np.allclose(infinite, [[4.0, 0.0]])


def test_softmax_vote_is_positive_and_equal_mass_per_valid_step() -> None:
    l2 = np.array([[[1.0], [0.0], [1.0]]], dtype=np.float64)
    lengths = np.array([2], dtype=np.int64)
    W = np.array([[2.0], [-1.0]], dtype=np.float64)
    scores = exp7264._softmax_vote_scores(l2, lengths, W, temperature=1.0)
    assert np.all(scores >= 0.0)
    assert np.allclose(scores.sum(axis=1), lengths.astype(np.float64))


def test_c1_allows_temporal_compensation_more_than_tsce() -> None:
    y = torch.tensor([0])
    lengths = torch.tensor([2])
    evidence_uneven = torch.tensor(
        [[[np.log(0.9), np.log(0.1)], [np.log(0.1), np.log(0.9)]]],
        dtype=torch.float32,
    )
    evidence_even = torch.zeros((1, 2, 2), dtype=torch.float32)

    c1_uneven, _ = exp7264._c_loss_scores_from_evidence(
        evidence_uneven, y, lengths, "softmax_vote_wcce"
    )
    c1_even, _ = exp7264._c_loss_scores_from_evidence(
        evidence_even, y, lengths, "softmax_vote_wcce"
    )
    c2_uneven, _ = exp7264._c_loss_scores_from_evidence(
        evidence_uneven, y, lengths, "tsce"
    )
    c2_even, _ = exp7264._c_loss_scores_from_evidence(
        evidence_even, y, lengths, "tsce"
    )

    assert torch.allclose(c1_uneven, c1_even, atol=1e-6)
    assert c2_uneven > c2_even


def test_slurm_arrays_and_dependency_chain() -> None:
    scripts = REPO_ROOT / "scripts" / "bash_script" / "SNN_Bash"
    assert "#SBATCH --array=0-2%3" in (
        scripts / "prepare_exp_7_2_6_4_cache_cpu_array.bash"
    ).read_text()
    assert "#SBATCH --array=0-2%3" in (
        scripts / "run_exp_7_2_6_4_part_a_cpu_array.bash"
    ).read_text()
    assert "#SBATCH --array=0-2%3" in (
        scripts / "run_exp_7_2_6_4_part_b_cpu_array.bash"
    ).read_text()
    assert "#SBATCH --array=0-8%9" in (
        scripts / "run_exp_7_2_6_4_part_c_cpu_array.bash"
    ).read_text()
    assert "#SBATCH --array=0-26%12" in (
        scripts / "run_exp_7_2_6_4_part_d_cpu_array.bash"
    ).read_text()
    assert "#SBATCH --array=0-2%3" in (
        scripts / "run_exp_7_2_6_4_gradient_cpu_array.bash"
    ).read_text()

    submit = (scripts / "submit_exp_7_2_6_4_cpu.bash").read_text()
    assert 'afterok:${cache_job}' in submit
    assert 'afterok:${part_a_job}:${part_b_job}:${part_c_job}:${part_d_job}:${grad_job}' in submit


def test_notebook_is_aggregate_only() -> None:
    text = (
        REPO_ROOT
        / "notebooks"
        / "experiment_7_2_6_4_output_interface_decomposition.ipynb"
    ).read_text()
    assert "part_a_flush_summary.csv" in text
    assert "part_b_evidence_semantics_summary.csv" in text
    assert "part_c_objective_training_summary.csv" in text
    assert "part_d_gain_sweep_summary.csv" in text
    assert "part_d_gradient_geometry_summary.csv" in text
    assert "torch" not in text
    assert "subprocess" not in text
    assert "sbatch" not in text
