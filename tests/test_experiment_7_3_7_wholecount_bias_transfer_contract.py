from pathlib import Path

import numpy as np

from scripts import experiment_7_3_7_wholecount_bias_transfer as exp737


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_protocol_and_run_mapping() -> None:
    assert exp737.ARCHITECTURE == "234x234"
    assert exp737.SEEDS == (11, 23, 37)
    assert exp737.BACKBONE_OBJECTIVE == "wcce"
    assert exp737.LINEAR_METHODS == (
        "L0_linear_no_bias",
        "L1_linear_affine",
    )
    assert exp737.METHODS == (
        "L0_linear_no_bias",
        "L1_linear_affine",
        "L2_same_w_lif_count",
        "L3_same_w_lif_continuous_bias",
        "L4_same_w_lif_integer_bias",
    )
    assert [spec.seed for spec in exp737.run_specs()] == [11, 23, 37]
    assert exp737.LIF_BETA == 0.5
    assert exp737.THRESHOLD == 0.5
    assert exp737.OUTPUT_CAP == 1


def test_wholecount_features_use_only_valid_steps() -> None:
    l2 = np.array(
        [
            [[1, 0], [0, 1], [1, 1], [9, 9]],
            [[0, 1], [1, 0], [8, 8], [8, 8]],
        ],
        dtype=np.uint8,
    )
    y = np.array([1, 0], dtype=np.int64)
    lengths = np.array([3, 2], dtype=np.int64)
    features, got_y = exp737._wholecount_features((l2, y, lengths))
    np.testing.assert_array_equal(features, np.array([[2.0, 2.0], [1.0, 1.0]]))
    np.testing.assert_array_equal(got_y, y)


def test_scale_fold_preserves_affine_scores() -> None:
    coef = np.array([[2.0, -1.0], [0.5, 4.0]], dtype=np.float64)
    intercept = np.array([0.25, -0.75], dtype=np.float64)
    scale = np.array([2.0, 0.5], dtype=np.float64)
    raw_w, raw_b = exp737._fold_scaled_affine(coef, intercept, scale)
    x = np.array([[1.0, 3.0], [2.0, -1.0]], dtype=np.float64)
    scaled_scores = (x / scale[None, :]) @ coef.T + intercept[None, :]
    raw_scores = x @ raw_w.T + raw_b[None, :]
    np.testing.assert_allclose(raw_scores, scaled_scores, rtol=1e-12, atol=1e-12)


def test_count_bias_nonnegative_gauge_and_positive_gain_preserve_argmax() -> None:
    counts = np.array([[3.0, 4.0, 1.0], [2.0, 0.0, 5.0]])
    q = np.array([-1.5, 0.25, 1.25])
    q_nonnegative = q - q.min()
    base = np.argmax(counts + q[None, :], axis=1)
    shifted = np.argmax(counts + q_nonnegative[None, :], axis=1)
    gained = np.argmax(0.37 * (counts + q[None, :]), axis=1)
    np.testing.assert_array_equal(base, shifted)
    np.testing.assert_array_equal(base, gained)
    assert np.all(q_nonnegative >= 0.0)


def test_slurm_layout_and_afterok_finalizer() -> None:
    root = REPO_ROOT / "scripts" / "bash_script" / "SNN_Bash"
    worker = (root / "run_exp_7_3_7_cpu_array.bash").read_text()
    assert "#SBATCH --array=0-2%3" in worker
    assert "#SBATCH --cpus-per-task=1" in worker
    assert "OMP_NUM_THREADS=1" in worker
    assert "MKL_NUM_THREADS=1" in worker
    assert "python -m scripts.experiment_7_3_7_wholecount_bias_transfer" in worker

    submit = (root / "submit_exp_7_3_7_cpu.bash").read_text()
    assert 'afterok:${train_job}' in submit


def test_notebook_is_analysis_only() -> None:
    text = (
        REPO_ROOT / "notebooks" / "experiment_7_3_7_wholecount_bias_transfer.ipynb"
    ).read_text()
    for name in (
        "method_summary.csv",
        "gap_summary.csv",
        "bias_vector_summary.csv",
        "calibration_history_summary.csv",
        "source_reproduction_checks.json",
        "manifest.json",
    ):
        assert name in text
    assert "import torch" not in text
    assert "subprocess" not in text
    assert "sbatch" not in text
