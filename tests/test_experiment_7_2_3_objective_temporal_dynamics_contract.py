from __future__ import annotations

import ast
from pathlib import Path

from scripts import experiment_7_2_3_objective_temporal_dynamics as exp723

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_exact_two_backbones_and_run_counts() -> None:
    assert exp723.ARCHITECTURE_ORDER == ("234x234", "34x345")
    assert len(exp723.FAMILIES) == 8
    assert len(exp723.all_specs()) == 96
    assert len(exp723.new_specs()) == 72
    assert len(exp723.reused_specs()) == 24
    assert {s.family for s in exp723.reused_specs()} == {exp723.A1, exp723.S1}
    assert {s.family for s in exp723.new_specs()} == set(exp723.NEW_FAMILIES)


def test_reuse_mapping_targets_exp72_local_and_e2e() -> None:
    a1 = exp723.RunSpec("234x234", exp723.A1, "task_only", 11)
    s1 = exp723.RunSpec("34x345", exp723.S1, "task_plus_reg", 23)
    assert exp723._base_exp72_spec(a1).training_family == "local_tsce"
    assert exp723._base_exp72_spec(s1).training_family == "e2e_wc"


def test_new_training_families_are_exactly_six() -> None:
    assert exp723.NEW_FAMILIES == (
        exp723.A2,
        exp723.F1,
        exp723.F2,
        exp723.S2,
        exp723.S3,
        exp723.S4,
    )


def test_phase_head_has_sixteen_absolute_bins() -> None:
    source = (REPO_ROOT / "scripts/experiment_7_2_3_objective_temporal_dynamics.py").read_text(encoding="utf-8")
    assert "N_BINS = 16" in source
    assert 'torch.einsum("bnd,nkd->bnk"' in source
    assert "phase_weight" in source
    assert "phase_bias" in source


def test_both_frozen_l2_probes_are_always_evaluated() -> None:
    source = (REPO_ROOT / "scripts/experiment_7_2_3_objective_temporal_dynamics.py").read_text(encoding="utf-8")
    assert '"l2_wholecount_linear"' in source
    assert '"l2_fixed250_linear"' in source
    assert '"probe_phase_gain"' in source


def test_spiking_counterfactuals_cover_beta_and_tail() -> None:
    source = (REPO_ROOT / "scripts/experiment_7_2_3_objective_temporal_dynamics.py").read_text(encoding="utf-8")
    for token in (
        "native_w_analog_valid",
        "native_w_analog_full",
        "native_lif_beta05_valid",
        "native_lif_beta10_valid",
        "native_lif_beta05_full",
        "native_lif_beta10_full",
    ):
        assert token in source


def test_slurm_topology_is_72_new_plus_24_reuse_then_finalizer() -> None:
    new = (REPO_ROOT / "scripts/bash_script/SNN_Bash/run_exp_7_2_3_new_cpu_array.bash").read_text(encoding="utf-8")
    reuse = (REPO_ROOT / "scripts/bash_script/SNN_Bash/run_exp_7_2_3_reuse_cpu_array.bash").read_text(encoding="utf-8")
    final = (REPO_ROOT / "scripts/bash_script/SNN_Bash/finalize_exp_7_2_3_cpu.bash").read_text(encoding="utf-8")
    submit = (REPO_ROOT / "scripts/bash_script/SNN_Bash/submit_exp_7_2_3_cpu.bash").read_text(encoding="utf-8")
    assert "#SBATCH --array=0-71%50" in new
    assert "#SBATCH --array=0-23%24" in reuse
    assert "#SBATCH --cpus-per-task=1" in new
    assert "#SBATCH --cpus-per-task=1" in reuse
    assert "run-new" in new
    assert "run-reuse" in reuse
    assert "finalize" in final
    assert 'afterok:${new_job}:${reuse_job}' in submit
    for text in (new, reuse, final):
        for token in ("OMP_NUM_THREADS=1", "MKL_NUM_THREADS=1", "OPENBLAS_NUM_THREADS=1", "NUMEXPR_NUM_THREADS=1"):
            assert token in text


def test_notebook_is_aggregate_only() -> None:
    text = (REPO_ROOT / "notebooks/experiment_7_2_3_objective_temporal_dynamics.ipynb").read_text(encoding="utf-8")
    for token in (
        "architecture_table.csv",
        "performance_summary.csv",
        "representation_probe_summary.csv",
        "paired_delta_summary.csv",
        "output_dynamics_summary.csv",
    ):
        assert token in text
    for token in (
        "performance_runs.csv",
        "representation_probe_runs.csv",
        "paired_delta_runs.csv",
        "output_dynamics_runs.csv",
        "checkpoints/",
        "evaluations/",
        "histories/",
        "torch.optim",
        "sbatch",
    ):
        assert token not in text


def test_source_parses() -> None:
    source = REPO_ROOT / "scripts/experiment_7_2_3_objective_temporal_dynamics.py"
    ast.parse(source.read_text(encoding="utf-8"))
