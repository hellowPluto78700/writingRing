from pathlib import Path

import numpy as np

from scripts import experiment_7_3_1_frozen_l2_linear_optimization as exp731


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_protocol_matrix_and_task_counts() -> None:
    assert exp731.ARCHITECTURE == "234x234"
    assert exp731.SEEDS == (11, 23, 37)
    assert exp731.BACKBONE_OBJECTIVES == ("tsce", "wcce")
    assert exp731.HEAD_OBJECTIVES == ("tsce", "wcce")
    assert len(exp731.CASES) == 6
    assert exp731.ADAM_C0_EPOCH_LIMIT == 100
    assert exp731.ADAM_EPOCHS == 500
    assert exp731.LBFGS_MAX_EVAL == 100
    assert exp731.REF_MAX_EVAL == 5000
    assert exp731.CE_GAIN == 1.0
    assert exp731.BIAS is False
    assert exp731.LIF_BETA == 0.5
    assert exp731.OUTPUT_CAP == 1
    assert len(exp731.base_specs()) == 12
    assert len(exp731.candidate_specs("C3_lbfgs_reg")) == 60
    assert len(exp731.candidate_specs("C4_scale_lbfgs_reg")) == 60
    assert len(exp731.candidate_specs("Ref_scale_lbfgs_converged")) == 60
    assert 3 * 2 * 2 * 6 == 72


def test_tsce_and_wcce_feature_construction() -> None:
    l2 = np.array(
        [
            [[1, 0], [0, 1], [1, 1]],
            [[0, 1], [1, 0], [0, 0]],
        ],
        dtype=np.uint8,
    )
    y = np.array([2, 4], dtype=np.int64)
    lengths = np.array([2, 1], dtype=np.int64)
    split = (l2, y, lengths)

    ts_x, ts_y = exp731._objective_features(split, "tsce")
    wc_x, wc_y = exp731._objective_features(split, "wcce")

    assert ts_x.shape == (3, 2)
    assert ts_y.tolist() == [2, 2, 4]
    np.testing.assert_allclose(wc_x, np.array([[0.5, 0.5], [0.0, 1.0]], dtype=np.float32))
    assert wc_y.tolist() == [2, 4]


def test_scale_only_weight_folding_is_exact() -> None:
    w_scaled = np.array([[2.0, -3.0], [1.0, 4.0]])
    sigma = np.array([2.0, 0.5])
    z = np.array([[0.25, 2.0], [1.5, -0.5]])
    w_raw = exp731._fold_scale_weight(w_scaled, sigma)
    np.testing.assert_allclose((z / sigma) @ w_scaled.T, z @ w_raw.T)


def test_all_main_cases_are_linear_train_then_same_w_lif() -> None:
    source = (
        REPO_ROOT / "scripts" / "experiment_7_3_1_frozen_l2_linear_optimization.py"
    ).read_text()
    assert 'Stage2Head("linear"' in source
    assert 'Stage2Head("lif"' not in source
    assert "_cross_evaluate_w" in source
    assert '"same_w_lif_substitution": True' in source
    assert '"no_lif_retraining": True' in source


def test_slurm_multi_cpu_layout_and_dependency_chain() -> None:
    root = REPO_ROOT / "scripts" / "bash_script" / "SNN_Bash"
    assert "#SBATCH --array=0-11%12" in (root / "run_exp_7_3_1_adam_cpu_array.bash").read_text()
    assert "#SBATCH --array=0-11%12" in (root / "run_exp_7_3_1_lbfgs_noreg_cpu_array.bash").read_text()
    for family in ("c3", "c4", "ref"):
        text = (root / f"run_exp_7_3_1_{family}_cpu_array.bash").read_text()
        assert "#SBATCH --array=0-59%50" in text
        assert "OMP_NUM_THREADS=1" in text
    submit = (root / "submit_exp_7_3_1_cpu.bash").read_text()
    assert 'afterok:${adam_job}:${c2_job}:${c3_job}:${c4_job}:${ref_job}' in submit


def test_notebook_is_aggregate_only() -> None:
    text = (
        REPO_ROOT / "notebooks" / "experiment_7_3_1_frozen_l2_linear_optimization.ipynb"
    ).read_text()
    for name in (
        "method_runs.csv",
        "method_summary.csv",
        "contrast_summary.csv",
        "selected_regularization.csv",
    ):
        assert name in text
    assert "import torch" not in text
    assert "subprocess" not in text
    assert "sbatch" not in text
