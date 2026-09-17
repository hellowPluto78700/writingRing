from pathlib import Path

from scripts import experiment_8_0_1_l1_l2_fusion_probe as exp801


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_source_and_feature_contract() -> None:
    assert exp801.SOURCE_ARCHITECTURE == "234x234"
    assert exp801.SEEDS == (11, 23, 37)
    assert len(exp801.run_specs()) == 3
    assert exp801.FEATURE_NAMES == (
        "l1_whole",
        "l2_whole",
        "l1_l2_whole",
        "l1_fixed250",
        "l2_fixed250",
        "l1_l2_fixed250",
        "l1whole_l2fixed250",
        "l1fixed250_l2whole",
    )


def test_fusion_components_are_explicit() -> None:
    specs = {spec.name: spec.components for spec in exp801.FEATURE_SPECS}
    assert specs["l1_l2_whole"] == (("l1", "whole_count"), ("l2", "whole_count"))
    assert specs["l1_l2_fixed250"] == (("l1", "fixed250"), ("l2", "fixed250"))
    assert specs["l1whole_l2fixed250"] == (("l1", "whole_count"), ("l2", "fixed250"))
    assert specs["l1fixed250_l2whole"] == (("l1", "fixed250"), ("l2", "whole_count"))


def test_cpu_array_and_dependency_chain() -> None:
    root = REPO_ROOT / "scripts" / "bash_script" / "SNN_Bash"
    run = (root / "run_exp_8_0_1_cpu_array.bash").read_text()
    assert "#SBATCH --array=0-2%3" in run
    assert "#SBATCH --cpus-per-task=1" in run
    assert "OMP_NUM_THREADS=1" in run
    assert "MKL_NUM_THREADS=1" in run
    assert "--array-task-id" in run

    submit = (root / "submit_exp_8_0_1_cpu.bash").read_text()
    assert 'afterok:${array_job}' in submit
    assert "run_exp_8_0_1_cpu_array.bash" in submit
    assert "finalize_exp_8_0_1_cpu.bash" in submit


def test_plan_and_notebook_contract() -> None:
    plan = (REPO_ROOT / "docs" / "plans" / "EXP8_0_1_L1_L2_FUSION_PROBE.md").read_text()
    assert "L1-only correct" in plan
    assert "L2-only correct" in plan
    assert "l1whole_l2fixed250" in plan
    assert "3 frozen checkpoints x 1 analysis task = 3 CPU jobs" in plan

    notebook = (REPO_ROOT / "notebooks" / "experiment_8_0_1_l1_l2_fusion_probe.ipynb").read_text()
    assert "probe_summary.csv" in notebook
    assert "fusion_gain_summary.csv" in notebook
    assert "correctness_overlap_summary.csv" in notebook
    assert "coef_block_summary.csv" in notebook
