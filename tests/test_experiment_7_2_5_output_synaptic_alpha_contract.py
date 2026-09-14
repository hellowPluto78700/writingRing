from __future__ import annotations

import ast
from pathlib import Path

import torch

from scripts import experiment_3_0_1_single_tau_objectives as exp3
from scripts import experiment_7_2_3_objective_temporal_dynamics as exp723
from scripts import experiment_7_2_5_output_synaptic_alpha as exp725

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_exact_run_counts_and_source_families() -> None:
    assert exp725.SOURCE_FAMILIES == (exp723.S1, exp723.S2, exp723.S3, exp723.S4)
    assert exp725.ARCHITECTURE_ORDER == ("234x234", "34x345")
    assert exp725.OUTPUT_ALPHAS == (0.0, 0.5)
    assert exp725.OUTPUT_BETA == 0.5
    assert len(exp725.source_specs()) == 48
    assert len(exp725.frozen_head_specs()) == 192
    assert len(exp725.e2e_specs()) == 48


def test_alpha_pairing_seeds_exclude_alpha() -> None:
    f0 = exp725.FrozenHeadSpec("234x234", exp723.S1, "task_only", 11, "whole_count_ce", 0.0)
    f5 = exp725.FrozenHeadSpec("234x234", exp723.S1, "task_only", 11, "whole_count_ce", 0.5)
    assert exp725.frozen_pair_seed(f0, "model_init") == exp725.frozen_pair_seed(f5, "model_init")
    assert exp725.frozen_pair_seed(f0, "train_loader") == exp725.frozen_pair_seed(f5, "train_loader")

    e0 = exp725.E2ESpec("34x345", "task_plus_reg", 23, "timestep_ce", 0.0)
    e5 = exp725.E2ESpec("34x345", "task_plus_reg", 23, "timestep_ce", 0.5)
    assert exp725.e2e_pair_seed(e0, "model_init") == exp725.e2e_pair_seed(e5, "model_init")
    assert exp725.e2e_pair_seed(e0, "train_loader") == exp725.e2e_pair_seed(e5, "train_loader")


def test_paired_e2e_initialization_is_identical_across_alpha() -> None:
    spec0 = exp725.E2ESpec("234x234", "task_only", 11, "whole_count_ce", 0.0)
    spec5 = exp725.E2ESpec("234x234", "task_only", 11, "whole_count_ce", 0.5)
    exp3.seed_all(exp725.e2e_pair_seed(spec0, "model_init"))
    m0 = exp725.PairedE2ESNN(spec0, n_classes=12, fs=64.0)
    exp3.seed_all(exp725.e2e_pair_seed(spec5, "model_init"))
    m5 = exp725.PairedE2ESNN(spec5, n_classes=12, fs=64.0)
    for name in ("hidden_linears.0.weight", "hidden_linears.1.weight", "output_linear.weight"):
        assert torch.equal(m0.state_dict()[name], m5.state_dict()[name])


def test_output_synaptic_equations_and_no_gain_normalization() -> None:
    source = (REPO_ROOT / "scripts/experiment_7_2_5_output_synaptic_alpha.py").read_text(encoding="utf-8")
    assert "syn = self.output_alpha * syn + self.output_linear(l2[:, t])" in source
    assert "out_syn = self.output_alpha * out_syn + evidence" in source
    assert "beta=OUTPUT_BETA" in source
    assert "(1.0 - self.output_alpha)" not in source
    assert "(1 - self.output_alpha)" not in source


def test_valid_length_training_and_checkpoint_selection_contract() -> None:
    source = (REPO_ROOT / "scripts/experiment_7_2_5_output_synaptic_alpha.py").read_text(encoding="utf-8")
    assert "exp50.objective_loss(spikes, lengths" in source
    assert "exp50.deployment_logits(spikes, lengths" in source
    assert 'val_metrics["valid"]["balanced_accuracy"]' in source
    assert 'val_metrics["objective_loss"] < best_loss' in source


def test_e2e_regularizer_is_hidden_only_and_both_l2_probes_run() -> None:
    source = (REPO_ROOT / "scripts/experiment_7_2_5_output_synaptic_alpha.py").read_text(encoding="utf-8")
    assert 'params = tuple(layer.weight for layer in model.hidden_linears)' in source
    assert '"output_regularized": False' in source
    assert 'VALID_WC_PROBE = "l2_wholecount_linear"' in source
    assert 'VALID_F250_PROBE = "l2_fixed250_linear"' in source
    assert "for source in PROBE_SOURCES" in source


def test_slurm_topology_is_48_then_192_then_48_then_finalizer() -> None:
    prep = (REPO_ROOT / "scripts/bash_script/SNN_Bash/prepare_exp_7_2_5_frozen_l2_cpu_array.bash").read_text(encoding="utf-8")
    frozen = (REPO_ROOT / "scripts/bash_script/SNN_Bash/run_exp_7_2_5_frozen_head_cpu_array.bash").read_text(encoding="utf-8")
    e2e = (REPO_ROOT / "scripts/bash_script/SNN_Bash/run_exp_7_2_5_e2e_cpu_array.bash").read_text(encoding="utf-8")
    final = (REPO_ROOT / "scripts/bash_script/SNN_Bash/finalize_exp_7_2_5_cpu.bash").read_text(encoding="utf-8")
    submit = (REPO_ROOT / "scripts/bash_script/SNN_Bash/submit_exp_7_2_5_cpu.bash").read_text(encoding="utf-8")

    assert "#SBATCH --array=0-47%48" in prep
    assert "#SBATCH --array=0-191%50" in frozen
    assert "#SBATCH --array=0-47%48" in e2e
    for text in (prep, frozen, e2e, final):
        assert "#SBATCH --cpus-per-task=1" in text
        for token in ("OMP_NUM_THREADS=1", "MKL_NUM_THREADS=1", "OPENBLAS_NUM_THREADS=1", "NUMEXPR_NUM_THREADS=1"):
            assert token in text
    assert 'afterok:${prep_job}' in submit
    assert 'afterok:${frozen_job}' in submit
    assert 'afterok:${e2e_job}' in submit
    assert "prepare-frozen" in prep
    assert "run-frozen" in frozen
    assert "run-e2e" in e2e
    assert "finalize" in final


def test_notebook_is_summary_only_and_orders_task_only_first() -> None:
    text = (REPO_ROOT / "notebooks/experiment_7_2_5_output_synaptic_alpha.ipynb").read_text(encoding="utf-8")
    for token in (
        "report_frozen_task_only.csv",
        "report_e2e_task_only_wc.csv",
        "report_e2e_task_only_tsce.csv",
        "report_frozen_task_plus_reg.csv",
        "report_e2e_task_plus_reg_wc.csv",
        "report_e2e_task_plus_reg_tsce.csv",
        "frozen_alpha_delta_summary.csv",
        "e2e_alpha_delta_summary.csv",
        "e2e_l2_probe_summary.csv",
        "mechanism_alpha_gain_summary.csv",
    ):
        assert token in text
    for token in (
        "frozen_head_performance_runs.csv",
        "e2e_performance_runs.csv",
        "e2e_l2_probe_runs.csv",
        "checkpoints/",
        "evaluations/",
        "histories/",
        "torch.optim",
        "sbatch",
    ):
        assert token not in text
    assert text.index("report_frozen_task_only.csv") < text.index("report_frozen_task_plus_reg.csv")


def test_source_parses() -> None:
    source = REPO_ROOT / "scripts/experiment_7_2_5_output_synaptic_alpha.py"
    ast.parse(source.read_text(encoding="utf-8"))
