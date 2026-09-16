from pathlib import Path

import numpy as np

from scripts import experiment_7_3_2_affine_probe_bridge as exp732


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_protocol_matrix_and_task_mapping() -> None:
    assert exp732.ARCHITECTURE == "234x234"
    assert exp732.SEEDS == (11, 23, 37)
    assert exp732.BACKBONE_OBJECTIVES == ("tsce", "wcce")
    assert len(exp732.CASES) == 8
    assert len(exp732.run_specs()) == 48
    assert exp732.MAX_ITER == 5000
    assert tuple(exp732.C_GRID) == (1e-3, 1e-2, 1e-1, 1.0, 10.0)


def test_p7_exactly_matches_legacy_probe_structure() -> None:
    p7 = exp732.CASE_BY_KEY["P7_wholecount_scale_center_bias"]
    assert p7.aggregation == "wholecount"
    assert p7.center is True
    assert p7.bias is True


def test_bridge_cases_cover_full_factorial() -> None:
    combos = {
        (case.aggregation, case.center, case.bias)
        for case in exp732.CASES
    }
    expected = {
        (aggregation, center, bias)
        for aggregation in ("mean", "wholecount")
        for center in (False, True)
        for bias in (False, True)
    }
    assert combos == expected


def test_mean_and_wholecount_feature_construction() -> None:
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

    mean_x, mean_y = exp732._aggregate_features(split, "mean")
    count_x, count_y = exp732._aggregate_features(split, "wholecount")

    np.testing.assert_allclose(mean_x, np.array([[0.5, 0.5], [0.0, 1.0]]))
    np.testing.assert_allclose(count_x, np.array([[1.0, 1.0], [0.0, 1.0]]))
    assert mean_y.tolist() == [2, 4]
    assert count_y.tolist() == [2, 4]


def test_slurm_layout_and_module_invocation() -> None:
    root = REPO_ROOT / "scripts" / "bash_script" / "SNN_Bash"
    worker = (root / "run_exp_7_3_2_cpu_array.bash").read_text()
    assert "#SBATCH --array=0-47%48" in worker
    assert "#SBATCH --cpus-per-task=1" in worker
    assert "OMP_NUM_THREADS=1" in worker
    assert "python -m scripts.experiment_7_3_2_affine_probe_bridge" in worker
    assert "python scripts/experiment_7_3_2_affine_probe_bridge.py" not in worker

    submit = (root / "submit_exp_7_3_2_cpu.bash").read_text()
    assert 'afterok:${bridge_job}' in submit


def test_notebook_is_analysis_only() -> None:
    text = (REPO_ROOT / "notebooks" / "experiment_7_3_2_affine_probe_bridge.ipynb").read_text()
    for name in (
        "method_runs.csv",
        "method_summary.csv",
        "contrast_summary.csv",
        "p7_legacy_reproduction.csv",
    ):
        assert name in text
    assert "import torch" not in text
    assert "subprocess" not in text
    assert "sbatch" not in text
