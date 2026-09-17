from pathlib import Path

from scripts import experiment_8_0_2_joint_l1_l2_supervision as exp802


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_experiment_matrix_and_backbone_contract() -> None:
    assert exp802.ARCHITECTURE == "234x234"
    assert exp802.ARCHITECTURE_SHIFTS == ((2, 3, 4), (2, 3, 4))
    assert exp802.SEEDS == (11, 23, 37)
    assert exp802.METHODS == (
        "l2_only",
        "l1_l2_joint",
        "l2_main_l1_aux",
    )
    assert exp802.AUX_LAMBDA == 0.1
    assert exp802.EXPECTED_RUNS == 9
    assert len(exp802.run_specs()) == 9


def test_model_head_contract() -> None:
    assert exp802.Exp802Net.__mro__[1].__name__ == "Exp80Net"
    assert exp802.exp801.FEATURE_NAMES == (
        "l1_whole",
        "l2_whole",
        "l1_l2_whole",
        "l1_fixed250",
        "l2_fixed250",
        "l1_l2_fixed250",
        "l1whole_l2fixed250",
        "l1fixed250_l2whole",
    )


def test_cpu_array_and_dependency_chain() -> None:
    root = REPO_ROOT / "scripts" / "bash_script" / "SNN_Bash"
    run = (root / "run_exp_8_0_2_cpu_array.bash").read_text()
    assert "#SBATCH --array=0-8%9" in run
    assert "#SBATCH --cpus-per-task=1" in run
    assert "OMP_NUM_THREADS=1" in run
    assert "MKL_NUM_THREADS=1" in run
    assert "OPENBLAS_NUM_THREADS=1" in run
    assert "NUMEXPR_NUM_THREADS=1" in run
    assert "--array-task-id" in run
    assert "experiment_8_0_2_joint_l1_l2_supervision" in run

    submit = (root / "submit_exp_8_0_2_cpu.bash").read_text()
    assert 'afterok:${array_job}' in submit
    assert "run_exp_8_0_2_cpu_array.bash" in submit
    assert "finalize_exp_8_0_2_cpu.bash" in submit


def test_plan_and_notebook_aggregation_contract() -> None:
    plan = (
        REPO_ROOT / "docs" / "plans" / "EXP8_0_2_JOINT_L1_L2_SUPERVISION.md"
    ).read_text()
    assert "3 methods x 3 seeds = 9 independent CPU training jobs" in plan
    assert "l1_l2_joint" in plan
    assert "l2_main_l1_aux" in plan
    assert "lambda=0.1" in plan.replace("\\lambda", "lambda") or "0.1" in plan

    notebook = (
        REPO_ROOT / "notebooks" / "experiment_8_0_2_joint_l1_l2_supervision.ipynb"
    ).read_text()
    assert "method_summary.csv" in notebook
    assert "probe_summary.csv" in notebook
    assert "fusion_gain_summary.csv" in notebook
    assert "correctness_overlap_summary.csv" in notebook
    assert "trained_head_summary.csv" in notebook
    assert "history_runs.csv" in notebook
