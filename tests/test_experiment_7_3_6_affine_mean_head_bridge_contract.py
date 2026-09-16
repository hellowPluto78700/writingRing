from pathlib import Path

import numpy as np
import torch

from scripts import experiment_7_3_6_affine_mean_head_bridge as exp736


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_protocol_and_run_mapping() -> None:
    assert exp736.ARCHITECTURE == "234x234"
    assert exp736.SEEDS == (11, 23, 37)
    assert exp736.BACKBONE_OBJECTIVE == "wcce"
    assert exp736.OBJECTIVE == "wcce"
    assert exp736.CASES == (
        "H0_raw_no_bias",
        "H1_raw_bias",
        "H2_scale_no_bias",
        "H3_scale_bias",
    )
    assert len(exp736.run_specs()) == 12
    assert exp736.LIF_BETA == 0.5
    assert exp736.THRESHOLD == 0.5
    assert exp736.OUTPUT_CAP == 1


def test_case_flags_are_exact() -> None:
    specs = {spec.case: spec for spec in exp736.run_specs() if spec.seed == 11}
    assert not specs["H0_raw_no_bias"].use_scale
    assert not specs["H0_raw_no_bias"].use_bias
    assert not specs["H1_raw_bias"].use_scale
    assert specs["H1_raw_bias"].use_bias
    assert specs["H2_scale_no_bias"].use_scale
    assert not specs["H2_scale_no_bias"].use_bias
    assert specs["H3_scale_bias"].use_scale
    assert specs["H3_scale_bias"].use_bias


def test_scale_fold_preserves_raw_function() -> None:
    sigma = np.array([2.0, 0.5], dtype=np.float32)
    head = exp736.AffineHead(2, 2, bias=True)
    with torch.no_grad():
        head.linear.weight.copy_(torch.tensor([[2.0, 1.0], [4.0, -1.0]]))
        head.linear.bias.copy_(torch.tensor([0.25, -0.5]))
    raw_w, raw_b = exp736._raw_parameters(head, sigma, use_scale=True)
    x = np.array([[1.0, 3.0], [2.0, -1.0]], dtype=np.float32)
    scaled_score = (x / sigma[None, :]) @ head.linear.weight.detach().numpy().T + head.linear.bias.detach().numpy()[None, :]
    raw_score = x @ raw_w.T + raw_b[None, :]
    np.testing.assert_allclose(scaled_score, raw_score, rtol=1e-6, atol=1e-6)


def test_beta1_charge_preserves_repeated_bias_affine_sum() -> None:
    l2 = np.array(
        [
            [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]],
            [[0.0, 1.0], [1.0, 0.0], [0.0, 0.0]],
        ],
        dtype=np.float32,
    )
    lengths = np.array([3, 2], dtype=np.int64)
    w = np.array([[0.3, -0.2], [0.1, 0.4]], dtype=np.float32)
    b = np.array([0.05, -0.1], dtype=np.float32)
    _, charge, error = exp736._simulate_if_charge(l2, lengths, w, b)
    assert error <= exp736.CHARGE_TOL

    expected = np.zeros_like(charge)
    for i, length in enumerate(lengths):
        expected[i] = (l2[i, :length] @ w.T + b[None, :]).sum(axis=0)
    np.testing.assert_allclose(charge, expected, rtol=1e-5, atol=1e-5)


def test_slurm_layout_and_afterok_finalizer() -> None:
    root = REPO_ROOT / "scripts" / "bash_script" / "SNN_Bash"
    worker = (root / "run_exp_7_3_6_cpu_array.bash").read_text()
    assert "#SBATCH --array=0-11%12" in worker
    assert "#SBATCH --cpus-per-task=1" in worker
    assert "OMP_NUM_THREADS=1" in worker
    assert "MKL_NUM_THREADS=1" in worker
    assert "python -m scripts.experiment_7_3_6_affine_mean_head_bridge" in worker

    submit = (root / "submit_exp_7_3_6_cpu.bash").read_text()
    assert 'afterok:${train_job}' in submit


def test_notebook_is_analysis_only() -> None:
    text = (
        REPO_ROOT / "notebooks" / "experiment_7_3_6_affine_mean_head_bridge.ipynb"
    ).read_text()
    for name in (
        "method_runs.csv",
        "method_summary.csv",
        "contrast_summary.csv",
        "bias_ablation.csv",
        "lif_realization_summary.csv",
        "history_summary.csv",
        "source_reproduction_checks.json",
    ):
        assert name in text
    assert "import torch" not in text
    assert "subprocess" not in text
    assert "sbatch" not in text
