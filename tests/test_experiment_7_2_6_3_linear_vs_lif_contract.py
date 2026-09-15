from pathlib import Path

import numpy as np
import torch

from scripts import experiment_7_2_6_3_linear_vs_lif_frozen_l2 as exp7263


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_core_contract() -> None:
    assert exp7263.ARCHITECTURE == "234x234"
    assert exp7263.REGULARIZATION == "task_only"
    assert exp7263.SEEDS == (11, 23, 37)
    assert exp7263.CE_LOGIT_GAIN == 1.0
    assert exp7263.OUTPUT_ALPHA == 0.0
    assert exp7263.LIF_BETA == 0.5
    assert exp7263.OUTPUT_CAP == 1
    assert exp7263.HEAD_MODES == ("linear", "lif")
    assert len(exp7263.source_specs()) == 3
    assert len(exp7263.head_specs()) == 6


def test_head_pair_seed_excludes_mode() -> None:
    assert exp7263._head_pair_seed(11, "model_init") == exp7263._head_pair_seed(11, "model_init")
    linear = exp7263.HeadSpec(11, exp7263.HEAD_LINEAR)
    lif = exp7263.HeadSpec(11, exp7263.HEAD_LIF)
    assert linear.seed == lif.seed


def test_mean_ce_gain_is_one_for_both_heads() -> None:
    l2 = torch.tensor([[[1.0, 0.0], [0.0, 1.0], [9.0, 9.0]]])
    lengths = torch.tensor([2])
    mean = exp7263._valid_mean(l2, lengths)
    assert torch.allclose(mean, torch.tensor([[0.5, 0.5]]))
    assert exp7263.CE_LOGIT_GAIN * mean[0, 0] == torch.tensor(0.5)


def test_charge_preserving_if_matches_analog_without_gain() -> None:
    l2 = np.array([[[1.0], [1.0], [1.0]]], dtype=np.float64)
    lengths = np.array([3], dtype=np.int64)
    W = np.array([[0.4], [0.2]], dtype=np.float64)
    analog = exp7263.exp726._analog_scores(l2, lengths, W, scale=1.0)
    sim = exp7263.exp726._simulate_unipolar(l2, lengths, W, beta=1.0, cap=1, scale=1.0)
    assert np.allclose(analog, sim["charge_scores"], atol=1e-10)


def test_slurm_arrays_and_dependency_chain() -> None:
    scripts = REPO_ROOT / "scripts" / "bash_script" / "SNN_Bash"
    assert "#SBATCH --array=0-2%3" in (scripts / "prepare_exp_7_2_6_3_cache_cpu_array.bash").read_text()
    assert "#SBATCH --array=0-2%3" in (scripts / "run_exp_7_2_6_3_mechanism_cpu_array.bash").read_text()
    assert "#SBATCH --array=0-5%6" in (scripts / "run_exp_7_2_6_3_head_cpu_array.bash").read_text()
    assert "#SBATCH --array=0-2%3" in (scripts / "run_exp_7_2_6_3_crosscheck_cpu_array.bash").read_text()
    submit = (scripts / "submit_exp_7_2_6_3_cpu.bash").read_text()
    assert "afterok:${cache_job}" in submit
    assert "afterok:${head_job}" in submit
    assert "afterok:${mech_job}:${cross_job}" in submit


def test_notebook_is_aggregate_only() -> None:
    text = (REPO_ROOT / "notebooks" / "experiment_7_2_6_3_linear_vs_lif_frozen_l2.ipynb").read_text()
    assert "mechanism_summary.csv" in text
    assert "crosscheck_summary.csv" in text
    assert "beta_sweep_summary.csv" in text
    assert "torch" not in text
    assert "subprocess" not in text
    assert "sbatch" not in text
