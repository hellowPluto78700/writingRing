from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import torch

from scripts import experiment_5_1_boundary_free_temporal_decoder as exp51
from scripts import experiment_5_2_2_frozen_local_multitau_syn as exp522


REPO_ROOT = Path(__file__).resolve().parents[1]
PREPARE = REPO_ROOT / "scripts" / "bash_script" / "SNN_Bash" / "prepare_exp_5_2_2_local_cpu_array.bash"
RUNNER = REPO_ROOT / "scripts" / "bash_script" / "SNN_Bash" / "run_exp_5_2_2_cpu_array.bash"
FINALIZER = REPO_ROOT / "scripts" / "bash_script" / "SNN_Bash" / "finalize_exp_5_2_2_cpu.bash"
SUBMIT = REPO_ROOT / "scripts" / "bash_script" / "SNN_Bash" / "submit_exp_5_2_2_cpu.bash"
README = REPO_ROOT / "scripts" / "experiment_5_2_2" / "README.md"
NOTEBOOK = REPO_ROOT / "notebooks" / "experiment_5_2_2_frozen_local_multitau_syn.ipynb"


def _fake_data() -> SimpleNamespace:
    return SimpleNamespace(labels=list(range(12)), fs=64.0, bin_steps=16)


def test_run_matrix_reuses_exp51_frozen_source() -> None:
    specs = exp522.run_specs()
    assert exp522.PROTOCOL_VERSION == "frozen_exp51_l2_multitau_syn_wholecount_v1"
    assert exp522.SOURCE_EXPERIMENT_ID == exp51.EXPERIMENT_ID
    assert exp522.SOURCE_PROTOCOL_VERSION == exp51.PROTOCOL_VERSION
    assert exp522.SEEDS == exp51.SEEDS == (11, 23, 101)
    assert exp522.READOUTS == ("hidden_count_linear", "output_lif")
    assert len(exp522.PROFILE_SHIFTS) == 10
    assert len(specs) == 60
    assert len({spec.key for spec in specs}) == 60


def test_temporal_profiles_and_neuron_allocations_are_fixed() -> None:
    assert exp522.PROFILE_SHIFTS["s234567"] == (2, 3, 4, 5, 6, 7)
    assert exp522.PROFILE_GROUP_COUNTS["s234567"] == (22, 21, 21, 21, 21, 22)
    assert exp522.PROFILE_GROUP_COUNTS["s4567"] == (32, 32, 32, 32)
    assert exp522.PROFILE_GROUP_COUNTS["s567"] == (43, 42, 43)
    assert exp522.PROFILE_GROUP_COUNTS["s67"] == (64, 64)
    for profile in exp522.PROFILE_SHIFTS:
        slices = exp522.profile_group_slices(profile)
        assert sum(group.stop - group.start for group in slices.values()) == 128
        alpha = exp522.alpha_vector(profile)
        assert alpha.shape == (128,)
        for shift, group in slices.items():
            assert torch.allclose(
                alpha[group], torch.full_like(alpha[group], exp522.base.alpha(shift))
            )


def test_model_is_ff_multitau_syn_with_short_membrane_and_two_readouts() -> None:
    hidden = exp522.SynapticPhaseDecoder("s234567", "hidden_count_linear", 12, 64.0, 16)
    output = exp522.SynapticPhaseDecoder("s234567", "output_lif", 12, 64.0, 16)
    assert hidden.input_hidden.in_features == 128
    assert hidden.input_hidden.out_features == 128
    assert hidden.output_linear.in_features == 128
    assert hidden.output_linear.out_features == 12
    assert hidden.output_linear.bias is None
    assert hidden.output_lif is None
    assert output.output_lif is not None
    assert not hasattr(hidden, "recurrent")
    assert exp522.TAU_MEM_MS == 22.54
    assert exp522.HIDDEN_CAP == 1
    assert exp522.OUTPUT_CAP == 1
    assert exp522.parameter_counts(12) == {"W3": 16384, "Wo": 1536, "trainable_total": 17920}


def test_hidden_count_readout_equals_summed_shared_linear_evidence() -> None:
    model = exp522.SynapticPhaseDecoder("s2", "hidden_count_linear", 3, 64.0, 4)
    x = torch.randn(2, 8, 128)
    lengths = torch.tensor([5, 8])
    trajectory = model.forward_trajectory(x)
    native = model.native_logits(trajectory, lengths)
    pre = model.pre_lif_logits(trajectory, lengths)
    assert torch.allclose(native, pre, atol=1e-6)


def test_reset_intervention_contract_and_profile_intersection() -> None:
    assert exp522.INTERVENTIONS == (
        "normal",
        "resetall250",
        "reset234",
        "reset2345",
        "reset23456",
        "reset67",
        "reset567",
    )
    assert exp522.effective_reset_shifts("s4567", "reset234") == (4,)
    assert exp522.effective_reset_shifts("s4567", "reset2345") == (4, 5)
    assert exp522.effective_reset_shifts("s67", "reset234") == ()
    assert exp522.effective_reset_shifts("s67", "resetall250") == (6, 7)
    assert exp522.effective_reset_shifts("s67", "reset567") == (6, 7)
    assert exp522.effective_mask_id("s67", "reset234") == "none"
    assert exp522.effective_reset_neuron_count("s234567", "reset67") == 43


def test_paired_initialization_matches_across_profiles_and_readouts() -> None:
    data = _fake_data()
    device = torch.device("cpu")
    specs = [
        exp522.RunSpec("s2", "hidden_count_linear", 11),
        exp522.RunSpec("s7", "hidden_count_linear", 11),
        exp522.RunSpec("s234567", "output_lif", 11),
        exp522.RunSpec("s67", "output_lif", 11),
    ]
    models = [exp522._initialize_model(spec, data, device) for spec in specs]
    for model in models[1:]:
        assert torch.equal(models[0].input_hidden.weight, model.input_hidden.weight)
        assert torch.equal(models[0].output_linear.weight, model.output_linear.weight)


def test_multi_cpu_afterok_and_one_core_contract() -> None:
    prepare = PREPARE.read_text(encoding="utf-8")
    runner = RUNNER.read_text(encoding="utf-8")
    finalizer = FINALIZER.read_text(encoding="utf-8")
    submit = SUBMIT.read_text(encoding="utf-8")
    assert "#SBATCH --array=0-2%3" in prepare
    assert "#SBATCH --array=0-59%50" in runner
    for text in (prepare, runner, finalizer):
        assert "#SBATCH --cpus-per-task=1" in text
        assert "module load conda/latest" in text
        assert "conda activate writingring-gpu" in text
        for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
            assert f"export {name}=1" in text
    assert 'afterok:${LOCAL_JOB}' in submit
    assert 'afterok:${RUN_JOB}' in submit
    assert "60 runs" in submit
    assert "max 50 concurrent" in submit


def test_notebook_is_analysis_only_and_contains_required_diagnostics() -> None:
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    assert notebook["nbformat"] == 4
    sources: list[str] = []
    for index, cell in enumerate(notebook["cells"]):
        source = "".join(cell.get("source", []))
        sources.append(source)
        if cell.get("cell_type") == "code":
            compile(source, f"exp522-notebook-cell-{index}", "exec")
    joined = "\n".join(sources)
    for token in (
        "runs.csv",
        "ablation_runs.csv",
        "probes.csv",
        "activity.csv",
        "histories.csv",
        "local_reference.csv",
        "manifest.json",
        "native_val_balanced_accuracy",
        "resetall250",
        "reset234",
        "reset2345",
        "reset23456",
        "reset67",
        "reset567",
        "organization_gap",
        "pre_lif_test_balanced_accuracy",
        "l3_whole_count",
        "l3_fixed250_ordered",
        "l3_relative10_ordered",
        "event_rate_hz",
        "plt.subplots",
    ):
        assert token in joined
    for forbidden in (
        "optimizer.step(",
        ".backward()",
        "subprocess",
        "sbatch",
        "run-one",
        "prepare-local",
    ):
        assert forbidden not in joined


def test_readme_documents_full_training_and_intervention_contract() -> None:
    text = README.read_text(encoding="utf-8")
    for token in (
        "Experiment 5.1",
        "s234567",
        "s4567",
        "s567",
        "s67",
        "hidden_count_linear",
        "output_lif",
        "17,920",
        "60 independent training runs",
        "resetall250",
        "reset234",
        "reset2345",
        "reset23456",
        "reset67",
        "reset567",
        "I_L3 = 0",
        "U_L3 = 0",
        "output-LIF state is never reset",
        "0-59%50",
        "analysis-only",
    ):
        assert token in text
