from __future__ import annotations

import json
import math
from pathlib import Path

import torch

from scripts import experiment_5_3_synaptic_contextual_evidence as exp53


REPO_ROOT = Path(__file__).resolve().parents[1]
PREP = (
    REPO_ROOT
    / "scripts"
    / "bash_script"
    / "SNN_Bash"
    / "prepare_exp_5_3_local_cpu_array.bash"
)
RUNNER = (
    REPO_ROOT
    / "scripts"
    / "bash_script"
    / "SNN_Bash"
    / "run_exp_5_3_cpu_array.bash"
)
FINALIZER = (
    REPO_ROOT
    / "scripts"
    / "bash_script"
    / "SNN_Bash"
    / "finalize_exp_5_3_cpu.bash"
)
SUBMIT = (
    REPO_ROOT
    / "scripts"
    / "bash_script"
    / "SNN_Bash"
    / "submit_exp_5_3_cpu.bash"
)
README = REPO_ROOT / "scripts" / "experiment_5_3" / "README.md"
NOTEBOOK = (
    REPO_ROOT
    / "notebooks"
    / "experiment_5_3_synaptic_contextual_evidence.ipynb"
)


def test_run_matrix_and_condition_contract() -> None:
    assert exp53.PROTOCOL_VERSION == "frozen_local_synaptic_history_context_v1"
    assert exp53.SEEDS == (11, 23, 37, 53, 71)
    assert exp53.CONDITION_NAMES == (
        "direct",
        "lif_beta100",
        "lif_beta050",
        "syn_single_s4",
        "syn_single_s5",
        "syn_single_s6",
        "syn_multi_s23456",
        "syn_multi_s456",
        "syn_multi_s56",
        "rsnn_beta050",
        "factorized_rsnn_beta050",
    )
    specs = exp53.run_specs()
    assert len(specs) == 55
    assert len({spec.key for spec in specs}) == 55


def test_lif_beta100_and_beta050_are_explicit_controls() -> None:
    beta100 = exp53.CONDITION_BY_NAME["lif_beta100"]
    beta050 = exp53.CONDITION_BY_NAME["lif_beta050"]
    assert beta100.family == beta050.family == "lif"
    assert beta100.beta == 1.00
    assert beta050.beta == 0.50
    assert beta100.shifts_syn == beta050.shifts_syn == ()
    assert math.isinf(exp53.tau_ms_from_decay(1.00, 64.0))
    assert math.isclose(
        exp53.tau_ms_from_decay(0.50, 64.0), 22.54, rel_tol=0.02
    )


def test_single_and_multi_tau_grids_are_exact() -> None:
    assert exp53.CONDITION_BY_NAME["syn_single_s4"].shifts_syn == (4,)
    assert exp53.CONDITION_BY_NAME["syn_single_s5"].shifts_syn == (5,)
    assert exp53.CONDITION_BY_NAME["syn_single_s6"].shifts_syn == (6,)
    assert exp53.CONDITION_BY_NAME["syn_multi_s23456"].shifts_syn == (
        2,
        3,
        4,
        5,
        6,
    )
    assert exp53.CONDITION_BY_NAME["syn_multi_s456"].shifts_syn == (4, 5, 6)
    assert exp53.CONDITION_BY_NAME["syn_multi_s56"].shifts_syn == (5, 6)
    taus = [
        exp53.tau_ms_from_shift(shift, 64.0)
        for shift in (2, 3, 4, 5, 6)
    ]
    assert all(left < right for left, right in zip(taus, taus[1:]))
    assert math.isclose(taus[0], 54.3, rel_tol=0.02)
    assert math.isclose(taus[-1], 992.2, rel_tol=0.02)


def test_multi_tau_assignment_is_fixed_and_balanced() -> None:
    for shifts in ((2, 3, 4, 5, 6), (4, 5, 6), (5, 6)):
        alpha, counts = exp53._alpha_layout(shifts)
        assert alpha.shape == (128,)
        assert sum(counts) == 128
        assert max(counts) - min(counts) <= 1
        expected = {exp53.decay_from_shift(shift) for shift in shifts}
        actual = {float(value) for value in torch.unique(alpha)}
        assert actual == expected


def test_final_accumulator_objective_uses_valid_length_and_no_head_bias() -> None:
    model = exp53.ContextualEvidenceNet("direct", n_classes=3, fs=64.0)
    assert model.evidence_head.bias is None
    with torch.no_grad():
        model.evidence_head.weight.zero_()
        model.evidence_head.weight[0, 0] = 1.0
    x = torch.zeros(2, 5, 128)
    x[0, :2, 0] = 1.0
    x[0, 2:, 0] = 100.0
    x[1, :4, 0] = 1.0
    lengths = torch.tensor([2, 4])
    y = torch.tensor([0, 0])
    loss, logits, accumulator = model.loss_logits(x, lengths, y)
    assert loss.ndim == 0
    assert torch.equal(accumulator[:, 0], torch.tensor([2.0, 4.0]))
    assert torch.allclose(logits[:, 0], torch.tensor([5.0, 5.0]))


def test_trajectory_shapes_and_state_reset_path() -> None:
    x = torch.zeros(2, 7, 128)
    for condition in exp53.CONDITION_NAMES:
        model = exp53.ContextualEvidenceNet(condition, n_classes=12, fs=64.0)
        normal = model.forward_trajectory(x)
        reset = model.forward_trajectory(x, reset_state_each_step=True)
        for trajectory in (normal, reset):
            assert trajectory.contextual.shape == (2, 7, 128)
            assert trajectory.evidence.shape == (2, 7, 12)
            assert trajectory.spikes.shape == (2, 7, 128)
            assert trajectory.membranes.shape == (2, 7, 128)
            assert trajectory.synaptic.shape == (2, 7, 128)
            assert trajectory.gates.shape == (2, 7, 128)


def test_initialization_is_paired_for_shared_modules() -> None:
    device = torch.device("cpu")
    specs = [
        exp53.RunSpec("lif_beta100", 11),
        exp53.RunSpec("lif_beta050", 11),
        exp53.RunSpec("syn_single_s4", 11),
        exp53.RunSpec("syn_multi_s456", 11),
        exp53.RunSpec("rsnn_beta050", 11),
    ]
    models = [
        exp53._initialize_model(spec, 12, 64.0, device) for spec in specs
    ]
    for model in models[1:]:
        assert torch.equal(
            models[0].input_projection.weight,
            model.input_projection.weight,
        )
        assert torch.equal(
            models[0].evidence_head.weight,
            model.evidence_head.weight,
        )


def test_factorized_gate_starts_with_zero_bias() -> None:
    model = exp53._initialize_model(
        exp53.RunSpec("factorized_rsnn_beta050", 11),
        12,
        64.0,
        torch.device("cpu"),
    )
    assert model.gate_projection is not None
    assert model.gate_projection.bias is not None
    assert torch.count_nonzero(model.gate_projection.bias) == 0


def test_relative_phase_examples_are_balanced_per_gesture() -> None:
    sequence = torch.arange(
        2 * 20 * 3, dtype=torch.float32
    ).reshape(2, 20, 3)
    means = exp53._relative_bin_means(sequence, torch.tensor([20, 13]))
    assert means.shape == (2, 10, 3)
    targets = exp53._phase_targets(2)
    assert targets.shape == (20,)
    assert torch.equal(torch.tensor(targets[:10]), torch.arange(10))
    assert torch.equal(torch.tensor(targets[10:]), torch.arange(10))


def test_probe_contract() -> None:
    assert exp53.CLASSIFICATION_PROBES == (
        "hidden_whole_count",
        "hidden_fixed250_ordered",
        "hidden_relative10_ordered",
        "hidden_uend",
    )
    assert exp53.PHASE_PROBES == (
        "phase_contextual",
        "phase_synaptic_state",
        "phase_membrane",
        "phase_spike",
        "phase_gate",
    )
    assert exp53.SHUFFLE_REPLICATES == 5


def test_multi_cpu_and_afterok_contract() -> None:
    prep = PREP.read_text(encoding="utf-8")
    runner = RUNNER.read_text(encoding="utf-8")
    finalizer = FINALIZER.read_text(encoding="utf-8")
    submit = SUBMIT.read_text(encoding="utf-8")
    assert "#SBATCH --array=0-4%5" in prep
    assert "#SBATCH --array=0-54%50" in runner
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
    assert "55 runs" in submit


def test_notebook_is_analysis_only_and_uses_validation_selection() -> None:
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    assert notebook["nbformat"] == 4
    sources: list[str] = []
    for index, cell in enumerate(notebook["cells"]):
        source = "".join(cell.get("source", []))
        sources.append(source)
        if cell.get("cell_type") == "code":
            compile(source, f"exp53-notebook-cell-{index}", "exec")
    joined = "\n".join(sources)
    for token in (
        "runs.csv",
        "histories.csv",
        "probe_runs.csv",
        "phase_probe_runs.csv",
        "ablation_runs.csv",
        "local_reference.csv",
        "manifest.json",
        "val_ba_mean",
        "selected_single_condition",
        "selected_multi_condition",
        "context_writing_gain",
        "internalization_ratio",
        "state_reset",
        "temporal_shuffle",
        "lif_beta050",
        "lif_beta100",
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


def test_readme_states_scientific_and_execution_contract() -> None:
    text = README.read_text(encoding="utf-8")
    for token in (
        "Synaptic history contextualization",
        "lif_beta100",
        "lif_beta050",
        "single-tau sweep is therefore `(4, 5, 6)`",
        "`(2,3,4,5,6)` versus `(4,5,6)` versus `(5,6)`",
        "final accumulated supervision",
        "non-leaky, non-spiking class-evidence accumulator",
        "hidden_whole_count",
        "state_reset",
        "temporal_shuffle",
        "11 conditions x 5 seeds = 55 independent runs",
        "0-54%50",
        "analysis-only",
    ):
        assert token in text
