from __future__ import annotations

import ast
import inspect
from pathlib import Path

import numpy as np
import torch

from scripts import experiment_7_2_4_global_window_eval as exp724

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_run_mapping_reuses_all_96_exp723_checkpoints() -> None:
    specs = exp724.run_specs()
    assert len(specs) == 96
    assert {spec.architecture for spec in specs} == {"234x234", "34x345"}
    assert {spec.family for spec in specs} == set(exp724.FAMILIES)
    assert {spec.regularization for spec in specs} == set(exp724.REGULARIZATIONS)
    assert {spec.seed for spec in specs} == {11, 23, 37}


def test_global_feature_builders_do_not_accept_lengths() -> None:
    assert tuple(inspect.signature(exp724.global_wholecount_features).parameters) == ("l2",)
    assert tuple(inspect.signature(exp724.global_fixed250_features).parameters) == ("l2", "bin_steps")


def test_global_wholecount_includes_post_endpoint_activity() -> None:
    l2 = torch.tensor([[[1.0], [2.0], [10.0], [20.0]]])
    feature = exp724.global_wholecount_features(l2)
    np.testing.assert_allclose(feature, np.array([[33.0]]))


def test_global_fixed250_includes_all_timesteps_in_absolute_bins() -> None:
    l2 = torch.zeros(1, 256, 1)
    l2[0, 0, 0] = 1.0
    l2[0, 15, 0] = 2.0
    l2[0, 16, 0] = 10.0
    l2[0, 255, 0] = 20.0
    feature = exp724.global_fixed250_features(l2, 16)
    assert feature.shape == (1, 16)
    assert feature[0, 0] == 3.0
    assert feature[0, 1] == 10.0
    assert feature[0, 15] == 20.0


def test_global_feature_dimensions_match_valid_probe_dimensions() -> None:
    l2 = torch.zeros(2, 256, 128)
    assert exp724.global_wholecount_features(l2).shape == (2, 128)
    assert exp724.global_fixed250_features(l2, 16).shape == (2, 16 * 128)


def test_global_probe_seed_is_paired_to_valid_probe_seed() -> None:
    spec = exp724.RunSpec("234x234", exp724.FAMILIES[0], "task_only", 11)
    expected_wc = exp724.exp3.dseed(spec.seed, exp724.exp723.EXPERIMENT_ID, spec.key, exp724.VALID_WC)
    expected_f250 = exp724.exp3.dseed(spec.seed, exp724.exp723.EXPERIMENT_ID, spec.key, exp724.VALID_F250)
    assert exp724._probe_seed(spec, exp724.GLOBAL_WC) == expected_wc
    assert exp724._probe_seed(spec, exp724.GLOBAL_F250) == expected_f250


def test_eval_source_has_no_training_path() -> None:
    source = (REPO_ROOT / "scripts/experiment_7_2_4_global_window_eval.py").read_text(encoding="utf-8")
    for forbidden in ("torch.optim", ".backward(", "optimizer.step", "train_one("):
        assert forbidden not in source
    assert "exp723._load_any_model" in source
    assert "exp01._fit_linear_probe" in source


def test_slurm_is_96_eval_tasks_then_afterok_finalizer() -> None:
    run = (REPO_ROOT / "scripts/bash_script/SNN_Bash/run_exp_7_2_4_cpu_array.bash").read_text(encoding="utf-8")
    final = (REPO_ROOT / "scripts/bash_script/SNN_Bash/finalize_exp_7_2_4_cpu.bash").read_text(encoding="utf-8")
    submit = (REPO_ROOT / "scripts/bash_script/SNN_Bash/submit_exp_7_2_4_cpu.bash").read_text(encoding="utf-8")
    assert "#SBATCH --array=0-95%50" in run
    assert "#SBATCH --cpus-per-task=1" in run
    assert "run --array-task-id" in run
    assert "finalize" in final
    assert 'afterok:${eval_job}' in submit
    for text in (run, final):
        for token in ("OMP_NUM_THREADS=1", "MKL_NUM_THREADS=1", "OPENBLAS_NUM_THREADS=1", "NUMEXPR_NUM_THREADS=1"):
            assert token in text


def test_finalizer_emits_separate_task_only_and_regularized_report_tables() -> None:
    source = (REPO_ROOT / "scripts/experiment_7_2_4_global_window_eval.py").read_text(encoding="utf-8")
    assert 'f"report_table_{regularization}.csv"' in source
    for token in (
        "global_window_probe_summary.csv",
        "global_vs_valid_delta_summary.csv",
        "l2_tail_activity_summary.csv",
        "l2_tail_offset_summary.csv",
    ):
        assert token in source


def test_notebook_is_summary_only_and_reports_task_only_first() -> None:
    text = (REPO_ROOT / "notebooks/experiment_7_2_4_global_window_eval.ipynb").read_text(encoding="utf-8")
    for token in (
        "report_table_task_only.csv",
        "report_table_task_plus_reg.csv",
        "global_window_probe_summary.csv",
        "global_vs_valid_delta_summary.csv",
        "l2_tail_activity_summary.csv",
        "l2_tail_offset_summary.csv",
    ):
        assert token in text
    assert text.index("Task-only — valid length known") < text.index("Task + regularizer — valid length known")
    for forbidden in (
        "global_window_probe_runs.csv",
        "global_vs_valid_delta_runs.csv",
        "l2_tail_activity_runs.csv",
        "l2_tail_offset_runs.csv",
        "checkpoints/",
        "evaluations/",
        "torch.optim",
        "sbatch",
    ):
        assert forbidden not in text


def test_source_parses() -> None:
    source = REPO_ROOT / "scripts/experiment_7_2_4_global_window_eval.py"
    ast.parse(source.read_text(encoding="utf-8"))
