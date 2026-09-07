from __future__ import annotations

import json
import math
from pathlib import Path

import torch

from scripts import experiment_5_3_1_factorized_gate_attribution as exp531


REPO_ROOT = Path(__file__).resolve().parents[1]
PREP = REPO_ROOT / "scripts/bash_script/SNN_Bash/prepare_exp_5_3_1_local_cpu_array.bash"
RUNNER = REPO_ROOT / "scripts/bash_script/SNN_Bash/run_exp_5_3_1_cpu_array.bash"
FINALIZER = REPO_ROOT / "scripts/bash_script/SNN_Bash/finalize_exp_5_3_1_cpu.bash"
SUBMIT = REPO_ROOT / "scripts/bash_script/SNN_Bash/submit_exp_5_3_1_cpu.bash"
README = REPO_ROOT / "scripts/experiment_5_3_1/README.md"
NOTEBOOK = REPO_ROOT / "notebooks/experiment_5_3_1_factorized_gate_attribution.ipynb"


def test_run_matrix_is_exactly_four_conditions_by_five_seeds() -> None:
    assert exp531.PROTOCOL_VERSION == "factorized_gate_causal_attribution_v1"
    assert exp531.SEEDS == (11, 23, 37, 53, 71)
    assert exp531.CONDITION_NAMES == (
        "factorized_ff_gate",
        "factorized_lif22",
        "factorized_lif242",
        "factorized_rsnn22",
    )
    specs = exp531.run_specs()
    assert len(specs) == 20
    assert len({spec.key for spec in specs}) == 20


def test_condition_hierarchy_is_content_then_passive_then_recurrent() -> None:
    ff = exp531.CONDITION_BY_NAME["factorized_ff_gate"]
    lif22 = exp531.CONDITION_BY_NAME["factorized_lif22"]
    lif242 = exp531.CONDITION_BY_NAME["factorized_lif242"]
    rsnn = exp531.CONDITION_BY_NAME["factorized_rsnn22"]
    assert (ff.beta, ff.recurrent, ff.stateful) == (0.0, False, False)
    assert (lif22.beta, lif22.recurrent, lif22.stateful) == (0.50, False, True)
    assert (lif242.beta, lif242.recurrent, lif242.stateful) == (0.9375, False, True)
    assert (rsnn.beta, rsnn.recurrent, rsnn.stateful) == (0.50, True, True)
    assert math.isclose(exp531.tau_ms_from_beta(0.50, 64.0), 22.54, rel_tol=0.02)
    assert math.isclose(exp531.tau_ms_from_beta(0.9375, 64.0), 242.1, rel_tol=0.02)
    assert exp531.tau_ms_from_beta(0.0, 64.0) == 0.0


def test_all_conditions_preserve_what_with_same_gate_shape_and_bias_free_head() -> None:
    x = torch.zeros(2, 7, 128)
    for condition in exp531.CONDITION_NAMES:
        model = exp531.FactorizedGateNet(condition, n_classes=12, fs=64.0)
        trajectory = model.forward_trajectory(x)
        assert trajectory.contextual.shape == (2, 7, 128)
        assert trajectory.evidence.shape == (2, 7, 12)
        assert trajectory.spikes.shape == (2, 7, 128)
        assert trajectory.membranes.shape == (2, 7, 128)
        assert trajectory.gates.shape == (2, 7, 128)
        assert model.evidence_head.bias is None
        assert model.gate_projection.bias is not None


def test_ff_gate_is_intrinsically_history_free() -> None:
    model = exp531._initialize_model(
        exp531.RunSpec("factorized_ff_gate", 11),
        12,
        64.0,
        torch.device("cpu"),
    )
    x = torch.randn(3, 9, 128)
    normal = model.forward_trajectory(x, reset_state_each_step=False)
    reset = model.forward_trajectory(x, reset_state_each_step=True)
    assert torch.equal(normal.contextual, reset.contextual)
    assert torch.equal(normal.gates, reset.gates)
    assert torch.equal(normal.evidence, reset.evidence)


def test_only_rsnn_condition_has_recurrent_matrix() -> None:
    for condition in exp531.CONDITION_NAMES:
        model = exp531.FactorizedGateNet(condition, n_classes=12, fs=64.0)
        if condition == "factorized_rsnn22":
            assert model.recurrent is not None
            assert model.recurrent.bias is None
        else:
            assert model.recurrent is None


def test_shared_module_initialization_is_paired_by_seed() -> None:
    models = [
        exp531._initialize_model(
            exp531.RunSpec(condition, 23),
            12,
            64.0,
            torch.device("cpu"),
        )
        for condition in exp531.CONDITION_NAMES
    ]
    reference = models[0]
    for model in models[1:]:
        assert torch.equal(reference.input_projection.weight, model.input_projection.weight)
        assert torch.equal(reference.gate_projection.weight, model.gate_projection.weight)
        assert torch.equal(reference.gate_projection.bias, model.gate_projection.bias)
        assert torch.equal(reference.evidence_head.weight, model.evidence_head.weight)
    assert torch.count_nonzero(reference.gate_projection.bias) == 0


def test_final_accumulator_masks_padding_and_uses_no_per_step_class_bias() -> None:
    model = exp531._initialize_model(
        exp531.RunSpec("factorized_lif242", 11),
        3,
        64.0,
        torch.device("cpu"),
    )
    assert model.evidence_head.bias is None
    prefix = torch.randn(1, 2, 128)
    x1 = torch.cat([prefix, torch.zeros(1, 3, 128)], dim=1)
    x2 = torch.cat([prefix, torch.randn(1, 3, 128) * 100.0], dim=1)
    lengths = torch.tensor([2])
    accumulator1 = model.forward_accumulator(x1, lengths)
    accumulator2 = model.forward_accumulator(x2, lengths)
    assert torch.allclose(accumulator1, accumulator2)


def test_probe_and_ablation_contract() -> None:
    assert exp531.CLASSIFICATION_PROBES == (
        "hidden_whole_count",
        "hidden_fixed250_ordered",
        "hidden_relative10_ordered",
    )
    assert exp531.PHASE_PROBES == (
        "phase_contextual",
        "phase_membrane",
        "phase_gate",
    )
    assert exp531.SHUFFLE_REPLICATES == 5
    assert callable(exp531._fit_ordered_transfer_probe)
    assert callable(exp531._apply_transfer_probe)
    assert callable(exp531.history_sensitivity)


def test_multi_cpu_and_afterok_contract() -> None:
    prep = PREP.read_text(encoding="utf-8")
    runner = RUNNER.read_text(encoding="utf-8")
    finalizer = FINALIZER.read_text(encoding="utf-8")
    submit = SUBMIT.read_text(encoding="utf-8")
    assert "#SBATCH --array=0-4%5" in prep
    assert "#SBATCH --array=0-19%20" in runner
    for text in (prep, runner, finalizer):
        assert "#SBATCH --cpus-per-task=1" in text
        assert "module load conda/latest" in text
        assert "conda activate writingring-gpu" in text
        for name in (
            "OMP_NUM_THREADS",
            "MKL_NUM_THREADS",
            "OPENBLAS_NUM_THREADS",
            "NUMEXPR_NUM_THREADS",
        ):
            assert f"export {name}=1" in text
    assert 'afterok:${LOCAL_JOB}' in submit
    assert 'afterok:${RUN_JOB}' in submit
    assert "20 runs" in submit


def test_notebook_is_analysis_only_and_uses_new_matrix() -> None:
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    assert notebook["nbformat"] == 4
    sources: list[str] = []
    for index, cell in enumerate(notebook["cells"]):
        source = "".join(cell.get("source", []))
        sources.append(source)
        if cell.get("cell_type") == "code":
            compile(source, f"exp531-notebook-cell-{index}", "exec")
    joined = "\n".join(sources)
    for token in (
        "runs.csv",
        "histories.csv",
        "probe_runs.csv",
        "phase_probe_runs.csv",
        "ablation_runs.csv",
        "history_sensitivity_runs.csv",
        "local_reference.csv",
        "manifest.json",
        "hidden_whole_count",
        "hidden_relative10_ordered",
        "internalization_gain",
        "factorized_ff_gate",
        "factorized_lif22",
        "factorized_lif242",
        "factorized_rsnn22",
        "state_reset",
        "temporal_shuffle",
        "transfer_mean",
        "active_feature_gate_history_mae",
        "plt.subplots",
        "fill_between",
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


def test_readme_states_attribution_and_aggregation_contract() -> None:
    text = README.read_text(encoding="utf-8")
    for token in (
        "Factorized gate causal attribution",
        "instantaneous content-conditioned nonlinear gating",
        "factorized_ff_gate",
        "factorized_lif22",
        "factorized_lif242",
        "factorized_rsnn22",
        "HiddenWholeCount",
        "HiddenRelative10",
        "ordered-trained transfer probe",
        "active_feature_gate_history_mae",
        "Four conditions x five seeds",
        "0-19%20",
        "afterok",
        "finalizer never retrains",
        "analysis-only notebook",
    ):
        assert token in text
