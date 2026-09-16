from pathlib import Path

import numpy as np

from scripts import experiment_7_3_3_affine_lif_substitution as exp733


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_protocol_and_run_mapping() -> None:
    assert exp733.ARCHITECTURE == "234x234"
    assert exp733.SEEDS == (11, 23, 37)
    assert exp733.BACKBONE_OBJECTIVES == ("tsce", "wcce")
    assert exp733.SOURCE_CASE == "P7_wholecount_scale_center_bias"
    assert len(exp733.run_specs()) == 6
    assert exp733.LIF_BETA == 0.5
    assert exp733.IF_BETA == 1.0
    assert exp733.THRESHOLD == 0.5
    assert exp733.OUTPUT_CAP == 1


def test_methods_cover_affine_lif_and_if_controls() -> None:
    expected = {
        "analog_affine",
        "analog_w_only",
        "lif_beta05_w_only",
        "lif_beta05_bias_start",
        "lif_beta05_bias_end",
        "if_beta1_bias_start_count",
        "if_beta1_bias_start_charge",
        "if_beta1_bias_end_count",
        "if_beta1_bias_end_charge",
    }
    assert set(exp733.METHODS) == expected


def test_beta1_charge_preserves_affine_sum_for_start_and_end_pulses() -> None:
    l2 = np.array(
        [
            [[1, 0], [0, 1], [1, 1]],
            [[0, 1], [1, 0], [0, 0]],
        ],
        dtype=np.uint8,
    )
    lengths = np.array([3, 2], dtype=np.int64)
    weight = np.array(
        [
            [0.3, -0.2],
            [-0.1, 0.4],
        ],
        dtype=np.float64,
    )
    intercept = np.array([0.7, -0.25], dtype=np.float64)
    analog = exp733._analog_affine_scores(l2, lengths, weight, intercept)

    start = exp733._simulate_affine_unipolar(
        l2,
        lengths,
        weight,
        intercept,
        beta=1.0,
        cap=1,
        bias_pulse="start",
    )
    end = exp733._simulate_affine_unipolar(
        l2,
        lengths,
        weight,
        intercept,
        beta=1.0,
        cap=1,
        bias_pulse="end",
    )

    np.testing.assert_allclose(start["input_sum"], analog, atol=1e-12)
    np.testing.assert_allclose(end["input_sum"], analog, atol=1e-12)
    np.testing.assert_allclose(start["charge_scores"], analog, atol=1e-12)
    np.testing.assert_allclose(end["charge_scores"], analog, atol=1e-12)


def test_bias_pulse_is_injected_once_per_nonempty_sample() -> None:
    l2 = np.zeros((3, 4, 2), dtype=np.uint8)
    lengths = np.array([4, 2, 1], dtype=np.int64)
    weight = np.zeros((2, 2), dtype=np.float64)
    intercept = np.array([1.0, -1.0], dtype=np.float64)

    for pulse in ("start", "end"):
        sim = exp733._simulate_affine_unipolar(
            l2,
            lengths,
            weight,
            intercept,
            beta=0.5,
            cap=1,
            bias_pulse=pulse,
        )
        assert sim["diagnostics"]["bias_injection_samples"] == 3


def test_slurm_layout_and_module_invocation() -> None:
    root = REPO_ROOT / "scripts" / "bash_script" / "SNN_Bash"
    worker = (root / "run_exp_7_3_3_cpu_array.bash").read_text()
    assert "#SBATCH --array=0-5%6" in worker
    assert "#SBATCH --cpus-per-task=1" in worker
    assert "OMP_NUM_THREADS=1" in worker
    assert "python -m scripts.experiment_7_3_3_affine_lif_substitution" in worker
    assert "python scripts/experiment_7_3_3_affine_lif_substitution.py" not in worker

    submit = (root / "submit_exp_7_3_3_cpu.bash").read_text()
    assert 'afterok:${bridge_job}' in submit


def test_notebook_is_analysis_only() -> None:
    text = (
        REPO_ROOT / "notebooks" / "experiment_7_3_3_affine_lif_substitution.ipynb"
    ).read_text()
    for name in (
        "method_runs.csv",
        "method_summary.csv",
        "contrast_summary.csv",
        "p7_reproduction_and_charge_checks.csv",
    ):
        assert name in text
    assert "import torch" not in text
    assert "subprocess" not in text
    assert "sbatch" not in text
