from __future__ import annotations

import json
import math
from pathlib import Path

import torch

from scripts import experiment_5_3_2_when_representation_runtime as exp532


REPO_ROOT = Path(__file__).resolve().parents[1]
PREP = REPO_ROOT / "scripts/bash_script/SNN_Bash/prepare_exp_5_3_2_local_cpu_array.bash"
RUNNER = REPO_ROOT / "scripts/bash_script/SNN_Bash/run_exp_5_3_2_cpu_array.bash"
FINALIZER = REPO_ROOT / "scripts/bash_script/SNN_Bash/finalize_exp_5_3_2_cpu.bash"
SUBMIT = REPO_ROOT / "scripts/bash_script/SNN_Bash/submit_exp_5_3_2_cpu.bash"
README = REPO_ROOT / "scripts/experiment_5_3_2/README.md"
NOTEBOOK = REPO_ROOT / "notebooks/experiment_5_3_2_when_representation.ipynb"


def test_run_matrix_is_nine_conditions_by_five_seeds() -> None:
    assert exp532.PROTOCOL_VERSION == "supervised_causal_when_v1"
    assert exp532.SEEDS == (11, 23, 37, 53, 71)
    assert exp532.CONDITION_NAMES == (
        "mem_single_s4",
        "mem_single_s5",
        "mem_single_s6",
        "mem_multi_s456",
        "syn_single_s4",
        "syn_single_s5",
        "syn_single_s6",
        "syn_multi_s456",
        "rsnn_shortmem",
    )
    specs = exp532.run_specs()
    assert len(specs) == 45
    assert len({spec.key for spec in specs}) == 45


def test_membrane_and_synaptic_sweeps_change_only_the_intended_memory() -> None:
    for shift in (4, 5, 6):
        mem = exp532.CONDITION_BY_NAME[f"mem_single_s{shift}"]
        syn = exp532.CONDITION_BY_NAME[f"syn_single_s{shift}"]
        assert mem.shifts_mem == (shift,)
        assert mem.shifts_syn == (exp532.SHORT_SHIFT,)
        assert not mem.recurrent
        assert syn.shifts_mem == (exp532.SHORT_SHIFT,)
        assert syn.shifts_syn == (shift,)
        assert not syn.recurrent
    assert exp532.CONDITION_BY_NAME["mem_multi_s456"].shifts_mem == (4, 5, 6)
    assert exp532.CONDITION_BY_NAME["mem_multi_s456"].shifts_syn == (exp532.SHORT_SHIFT,)
    assert exp532.CONDITION_BY_NAME["syn_multi_s456"].shifts_mem == (exp532.SHORT_SHIFT,)
    assert exp532.CONDITION_BY_NAME["syn_multi_s456"].shifts_syn == (4, 5, 6)
    rsnn = exp532.CONDITION_BY_NAME["rsnn_shortmem"]
    assert rsnn.shifts_mem == (exp532.SHORT_SHIFT,)
    assert rsnn.shifts_syn == (exp532.SHORT_SHIFT,)
    assert rsnn.recurrent


def test_shift_times_are_about_242_492_992_ms_at_64_hz() -> None:
    expected = {4: 242.1, 5: 492.1, 6: 992.2}
    for shift, tau in expected.items():
        assert math.isclose(exp532.tau_ms_from_shift(shift, 64.0), tau, rel_tol=0.02)


def test_multi_tau_layout_is_deterministic_43_43_42() -> None:
    model_mem = exp532.WhenBranchNet("mem_multi_s456", fs=64.0)
    model_syn = exp532.WhenBranchNet("syn_multi_s456", fs=64.0)
    assert model_mem.beta_counts == (43, 43, 42)
    assert model_syn.alpha_counts == (43, 43, 42)
    assert model_mem.group_indices("membrane") == {
        "membrane_s4": list(range(0, 128, 3)),
        "membrane_s5": list(range(1, 128, 3)),
        "membrane_s6": list(range(2, 128, 3)),
    }
    assert model_syn.group_indices("synaptic") == {
        "synaptic_s4": list(range(0, 128, 3)),
        "synaptic_s5": list(range(1, 128, 3)),
        "synaptic_s6": list(range(2, 128, 3)),
    }


def test_when_targets_are_relative_progress_and_ignore_padding() -> None:
    lengths = torch.tensor([3, 5])
    phase, progress, valid = exp532._phase_progress_targets(lengths, 7, torch.float32)
    assert phase.shape == progress.shape == valid.shape == (2, 7)
    assert torch.equal(phase[0, :3], torch.tensor([0, 5, 9]))
    assert torch.allclose(progress[0, :3], torch.tensor([0.0, 0.5, 1.0]))
    assert torch.equal(phase[1, :5], torch.tensor([0, 2, 5, 7, 9]))
    assert valid[0].sum().item() == 3
    assert valid[1].sum().item() == 5


def test_loss_is_phase_plus_progress_from_membrane_and_sample_balanced() -> None:
    model = exp532._initialize_model(
        exp532.RunSpec("mem_single_s4", 11), 64.0, torch.device("cpu")
    )
    x = torch.randn(2, 6, 128)
    lengths = torch.tensor([2, 5])
    total, phase, progress, logits, progress_pred, trajectory = model.loss_components(x, lengths)
    assert torch.allclose(total, phase + progress)
    assert logits.shape == (2, 6, 10)
    assert progress_pred.shape == (2, 6)
    assert trajectory.membranes.shape == (2, 6, 128)
    assert model.phase_head.in_features == exp532.WHEN_WIDTH
    assert model.progress_head.in_features == exp532.WHEN_WIDTH

    x_changed = x.clone()
    x_changed[0, 2:] = torch.randn_like(x_changed[0, 2:]) * 100.0
    total_changed, phase_changed, progress_changed, *_ = model.loss_components(x_changed, lengths)
    assert torch.allclose(total, total_changed)
    assert torch.allclose(phase, phase_changed)
    assert torch.allclose(progress, progress_changed)


def test_only_rsnn_has_recurrent_weights() -> None:
    for condition in exp532.CONDITION_NAMES:
        model = exp532.WhenBranchNet(condition, fs=64.0)
        if condition == "rsnn_shortmem":
            assert model.recurrent is not None
            assert model.recurrent.bias is None
        else:
            assert model.recurrent is None


def test_state_reset_removes_carried_history_but_not_current_input() -> None:
    model = exp532._initialize_model(
        exp532.RunSpec("syn_single_s6", 23), 64.0, torch.device("cpu")
    )
    x = torch.randn(2, 8, 128)
    lengths = torch.tensor([8, 8])
    ordered = model.forward_trajectory(x, lengths, reset_state_each_step=False)
    reset = model.forward_trajectory(x, lengths, reset_state_each_step=True)
    assert ordered.membranes.shape == reset.membranes.shape == (2, 8, 128)
    assert not torch.equal(ordered.synaptic[:, 1:], reset.synaptic[:, 1:])
    one_step = model.forward_trajectory(x[:, :1], torch.ones(2, dtype=torch.long), True)
    assert torch.allclose(reset.membranes[:, :1], one_step.membranes)


def test_trailing_spike_counts_are_causal() -> None:
    spikes = torch.zeros(1, 6, 1)
    spikes[0, 1, 0] = 1.0
    spikes[0, 4, 0] = 1.0
    counts = exp532._trailing_count(spikes, window_steps=3)
    assert torch.equal(counts.flatten(), torch.tensor([0.0, 1.0, 1.0, 1.0, 1.0, 1.0]))


def test_multi_cpu_afterok_and_environment_contract() -> None:
    prep = PREP.read_text(encoding="utf-8")
    runner = RUNNER.read_text(encoding="utf-8")
    finalizer = FINALIZER.read_text(encoding="utf-8")
    submit = SUBMIT.read_text(encoding="utf-8")
    assert "#SBATCH --array=0-4%5" in prep
    assert "#SBATCH --array=0-44%45" in runner
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
    assert "45 runs" in submit


def test_notebook_is_analysis_only_and_reads_finalized_artifacts() -> None:
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    assert notebook["nbformat"] == 4
    sources: list[str] = []
    for index, cell in enumerate(notebook["cells"]):
        source = "".join(cell.get("source", []))
        sources.append(source)
        if cell.get("cell_type") == "code":
            compile(source, f"exp532-notebook-cell-{index}", "exec")
    joined = "\n".join(sources)
    for token in (
        "runs.csv",
        "histories.csv",
        "probe_runs.csv",
        "group_probe_runs.csv",
        "ablation_runs.csv",
        "baseline_runs.csv",
        "local_reference.csv",
        "manifest.json",
        "mem_single_s4",
        "mem_multi_s456",
        "syn_single_s4",
        "syn_multi_s456",
        "rsnn_shortmem",
        "spike250",
        "spike500",
        "state_reset",
        "temporal_shuffle",
        "H_reset",
        "H_shuffle",
        "plt.subplots",
    ):
        assert token in joined
    for forbidden in (
        "optimizer.step(",
        ".backward()",
        "LogisticRegression(",
        "Ridge(",
        "subprocess",
        "sbatch",
        "run-one",
        "prepare-local",
    ):
        assert forbidden not in joined


def test_readme_states_when_only_and_aggregation_contract() -> None:
    text = README.read_text(encoding="utf-8")
    for token in (
        "Supervised causal WHEN representation benchmark",
        "mem_single_s4",
        "mem_single_s5",
        "mem_single_s6",
        "mem_multi_s456",
        "syn_single_s4",
        "syn_single_s5",
        "syn_single_s6",
        "syn_multi_s456",
        "rsnn_shortmem",
        "L_WHEN = L_phase(U_t) + L_progress(U_t)",
        "WHAT-only",
        "Elapsed-time-only",
        "state_reset",
        "temporal_shuffle",
        "45 independent runs",
        "0-44%45",
        "afterok",
        "finalizer never retrains",
        "Analysis-only notebook",
    ):
        assert token in text
