from __future__ import annotations

import inspect
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from scripts import experiment_5_5_2_progress_anchored_history_residual as exp552


REPO_ROOT = Path(__file__).resolve().parents[1]
SLURM_DIR = REPO_ROOT / "scripts/bash_script/SNN_Bash"
README = REPO_ROOT / "scripts/experiment_5_5_2/README.md"
NOTEBOOK = REPO_ROOT / "notebooks/experiment_5_5_2_progress_anchored_history_residual.ipynb"
RUNNERS = (
    "validate_exp_5_5_2_source_cpu_array.bash",
    "run_exp_5_5_2_clock_screen_cpu_array.bash",
    "select_exp_5_5_2_clock_screen_cpu.bash",
    "run_exp_5_5_2_clock_final_cpu_array.bash",
    "run_exp_5_5_2_oracle_final_cpu_array.bash",
    "finalize_exp_5_5_2_cpu.bash",
)
SUBMITTER = "submit_exp_5_5_2_cpu.bash"


def _model(mode: str, *, anchor: str = exp552.CLOCK, alpha: float = 0.5) -> exp552.ProgressAnchoredResidual:
    torch.manual_seed(123)
    experts = torch.randn(exp552.N_STATES, exp552.N_CLASSES, exp552.WHAT_WIDTH)
    bias = torch.randn(exp552.N_CLASSES)
    return exp552.ProgressAnchoredResidual(
        anchor=anchor,
        residual_mode=mode,
        alpha=alpha,
        expert_weight=experts,
        class_bias=bias,
    )


def _activate_residual_head(model: exp552.ProgressAnchoredResidual) -> None:
    torch.manual_seed(456)
    with torch.no_grad():
        model.residual_head.weight.normal_(0.0, 0.05)
        model.residual_head.bias.normal_(0.0, 0.05)


def test_protocol_and_run_mapping() -> None:
    assert exp552.PROTOCOL_VERSION == "progress_anchored_history_residual_v1"
    assert exp552.SEEDS == (11, 23, 37, 53, 71)
    assert exp552.WHAT_WIDTH == 128
    assert exp552.GRU_WIDTH == 64
    assert exp552.N_STATES == 8
    assert exp552.ALPHAS == (0.25, 0.5, 1.0)
    assert exp552.RESIDUAL_MODES == ("coordinate", "reset", "lag1", "ordered")
    specs = exp552.screen_specs()
    assert len(specs) == 60
    assert len({spec.key for spec in specs}) == 60
    assert specs[0] == exp552.ScreenSpec(exp552.COORD, 0.25, 11)
    assert specs[-1] == exp552.ScreenSpec(exp552.ORDERED, 1.0, 71)
    for anchor in exp552.ANCHORS:
        final = exp552.final_mode_specs(anchor)
        assert len(final) == 20
        assert len({spec.key for spec in final}) == 20
        assert final[0] == exp552.FinalSpec(anchor, exp552.COORD, 11)
        assert final[-1] == exp552.FinalSpec(anchor, exp552.ORDERED, 71)


def test_expert_bank_and_class_bias_are_frozen() -> None:
    model = _model(exp552.ORDERED)
    assert model.expert_weight.requires_grad is False
    assert model.class_bias.requires_grad is False
    what = torch.randn(3, 7, exp552.WHAT_WIDTH)
    lengths = torch.tensor([7, 6, 5])
    labels = torch.tensor([0, 1, 2])
    logits, _ = model(what, lengths, 64.0, 3.0)
    F.cross_entropy(logits, labels).backward()
    assert model.expert_weight.grad is None
    assert model.class_bias.grad is None
    assert model.residual_head.weight.grad is not None


def test_zero_residual_exactly_reproduces_anchor() -> None:
    what = torch.randn(2, 8, exp552.WHAT_WIDTH)
    lengths = torch.tensor([8, 6])
    for mode in exp552.RESIDUAL_MODES:
        model = _model(mode)
        assert torch.count_nonzero(model.residual_head.weight) == 0
        assert torch.count_nonzero(model.residual_head.bias) == 0
        with torch.no_grad():
            logits, trajectory = model(what, lengths, 64.0, 3.0, return_trajectory=True)
        assert trajectory is not None
        assert torch.count_nonzero(trajectory.delta_z) == 0
        expected_q = torch.softmax(trajectory.anchor_logits, dim=-1)
        expected_q = expected_q * exp552.sequence_mask(lengths, what.shape[1]).to(what.dtype).unsqueeze(-1)
        assert torch.equal(trajectory.q, expected_q)
        assert torch.equal(logits, model.logits_from_q(what, expected_q, lengths))
    assert "log(" not in inspect.getsource(exp552.ProgressAnchoredResidual.anchor_routing_logits)


def test_anchor_rbf_probabilities_match_exp55() -> None:
    what = torch.randn(2, 9, exp552.WHAT_WIDTH)
    lengths = torch.tensor([9, 6])
    fs = 64.0
    max_elapsed = 2.5
    valid = exp552.sequence_mask(lengths, what.shape[1]).to(what.dtype).unsqueeze(-1)

    clock = _model(exp552.COORD, anchor=exp552.CLOCK)
    clock_logits = clock.anchor_routing_logits(what, lengths, fs, max_elapsed)
    centers = torch.linspace(0.0, max_elapsed, exp552.N_STATES)
    sigma = exp552.exp55.RBF_SIGMA_MULTIPLIER * max_elapsed / (exp552.N_STATES - 1)
    elapsed = (torch.arange(what.shape[1], dtype=what.dtype) / fs).clamp(max=max_elapsed)
    expected_clock = exp552.exp55._rbf_probabilities(
        elapsed.view(1, -1).expand(len(what), -1), centers, sigma
    ) * valid
    assert torch.allclose(torch.softmax(clock_logits, dim=-1) * valid, expected_clock, atol=1e-7, rtol=1e-7)

    oracle = _model(exp552.COORD, anchor=exp552.ORACLE)
    oracle_logits = oracle.anchor_routing_logits(what, lengths, fs, max_elapsed)
    centers = torch.linspace(0.0, 1.0, exp552.N_STATES)
    sigma = exp552.exp55.RBF_SIGMA_MULTIPLIER / (exp552.N_STATES - 1)
    time = torch.arange(what.shape[1], dtype=what.dtype).view(1, -1)
    progress = (time / (lengths.to(what.dtype) - 1.0).clamp(min=1.0).unsqueeze(1)).clamp(0.0, 1.0)
    expected_oracle = exp552.exp55._rbf_probabilities(progress, centers, sigma) * valid
    assert torch.allclose(torch.softmax(oracle_logits, dim=-1) * valid, expected_oracle, atol=1e-7, rtol=1e-7)


def test_residual_bound_and_past_only_t0_contract() -> None:
    what = torch.randn(3, 8, exp552.WHAT_WIDTH)
    lengths = torch.tensor([8, 7, 5])
    for mode in exp552.RESIDUAL_MODES:
        model = _model(mode, alpha=0.25)
        _activate_residual_head(model)
        with torch.no_grad():
            delta = model.residual_logits(what, lengths, 64.0, 3.0)
        assert float(delta.abs().max()) <= 0.25 + 1e-7
        if mode in (exp552.LAG1, exp552.ORDERED):
            assert torch.count_nonzero(delta[:, 0]) == 0


def test_ordered_current_routing_cannot_see_current_what() -> None:
    model = _model(exp552.ORDERED).eval()
    _activate_residual_head(model)
    what_a = torch.randn(2, 8, exp552.WHAT_WIDTH)
    what_b = what_a.clone()
    what_b[:, 3, :] += 10.0
    lengths = torch.tensor([8, 7])
    with torch.no_grad():
        _, a = model(what_a, lengths, 64.0, 3.0, return_trajectory=True)
        _, b = model(what_b, lengths, 64.0, 3.0, return_trajectory=True)
    assert a is not None and b is not None
    assert torch.equal(a.delta_z[:, 3], b.delta_z[:, 3])
    assert torch.equal(a.q[:, 3], b.q[:, 3])
    assert not torch.equal(a.delta_z[:, 4], b.delta_z[:, 4])


def test_reset_has_no_cross_timestep_history() -> None:
    model = _model(exp552.RESET).eval()
    _activate_residual_head(model)
    what_a = torch.randn(2, 8, exp552.WHAT_WIDTH)
    what_b = what_a.clone()
    what_b[:, :3, :] += 8.0
    lengths = torch.tensor([8, 8])
    with torch.no_grad():
        delta_a = model.residual_logits(what_a, lengths, 64.0, 3.0)
        delta_b = model.residual_logits(what_b, lengths, 64.0, 3.0)
    assert torch.equal(delta_a[:, 3], delta_b[:, 3])


def test_lag1_uses_previous_what_only() -> None:
    model = _model(exp552.LAG1).eval()
    _activate_residual_head(model)
    base = torch.randn(2, 8, exp552.WHAT_WIDTH)
    current_changed = base.clone(); current_changed[:, 4, :] += 8.0
    previous_changed = base.clone(); previous_changed[:, 3, :] += 8.0
    older_changed = base.clone(); older_changed[:, 2, :] += 8.0
    lengths = torch.tensor([8, 8])
    with torch.no_grad():
        d0 = model.residual_logits(base, lengths, 64.0, 3.0)
        d_current = model.residual_logits(current_changed, lengths, 64.0, 3.0)
        d_previous = model.residual_logits(previous_changed, lengths, 64.0, 3.0)
        d_older = model.residual_logits(older_changed, lengths, 64.0, 3.0)
    assert torch.equal(d0[:, 4], d_current[:, 4])
    assert not torch.equal(d0[:, 4], d_previous[:, 4])
    assert torch.equal(d0[:, 4], d_older[:, 4])


def test_coordinate_residual_never_uses_what_content() -> None:
    model = _model(exp552.COORD).eval()
    _activate_residual_head(model)
    what_a = torch.randn(2, 8, exp552.WHAT_WIDTH)
    what_b = torch.randn_like(what_a) * 20.0
    lengths = torch.tensor([8, 6])
    with torch.no_grad():
        delta_a = model.residual_logits(what_a, lengths, 64.0, 3.0)
        delta_b = model.residual_logits(what_b, lengths, 64.0, 3.0)
    assert torch.equal(delta_a, delta_b)


def test_padding_suffix_cannot_change_logits() -> None:
    model = _model(exp552.ORDERED).eval()
    _activate_residual_head(model)
    what_a = torch.randn(2, 9, exp552.WHAT_WIDTH)
    what_b = what_a.clone()
    lengths = torch.tensor([5, 7])
    what_b[0, 5:] = torch.randn_like(what_b[0, 5:]) * 100.0
    what_b[1, 7:] = torch.randn_like(what_b[1, 7:]) * 100.0
    with torch.no_grad():
        logits_a, _ = model(what_a, lengths, 64.0, 3.0)
        logits_b, _ = model(what_b, lengths, 64.0, 3.0)
    assert torch.equal(logits_a, logits_b)


def test_screen_selection_and_finalizer_contracts() -> None:
    source_gate = inspect.getsource(exp552.validate_source_seed)
    train_source = inspect.getsource(exp552._train_model)
    screen_source = inspect.getsource(exp552.evaluate_screen_one)
    selector = inspect.getsource(exp552.select_screen)
    finalizer = inspect.getsource(exp552.finalize_experiment)
    assert '("val",)' in source_gate
    assert '("train", "val", "test")' not in source_gate
    assert 'F.cross_entropy(logits, yb)' in train_source
    assert '("train", "val")' in train_source
    assert '("train", "val", "test")' not in train_source
    assert '"test_evaluated": False' in screen_source
    assert "CLOCK" in selector and "ORACLE" not in selector
    assert "selected_alpha_by_mode" in selector
    assert "for mode in RESIDUAL_MODES" in selector
    assert "_train_model(" not in finalizer
    assert "_evaluate_final(" not in finalizer
    assert "Finalizer will not regenerate" in finalizer


def test_clock_final_reuses_screen_checkpoint_and_oracle_trains_separately() -> None:
    clock_source = inspect.getsource(exp552.evaluate_clock_final_one)
    oracle_source = inspect.getsource(exp552.run_oracle_final_one)
    assert "screen_checkpoint_path" in clock_source
    assert "_train_model(" not in clock_source
    assert "train_oracle_final_one" in oracle_source
    assert "_evaluate_final" in oracle_source


def test_slurm_multi_cpu_dependency_contract() -> None:
    paths = {name: SLURM_DIR / name for name in RUNNERS + (SUBMITTER,)}
    for name, path in paths.items():
        assert path.is_file(), name
        text = path.read_text(encoding="utf-8")
        assert text.startswith("#!/usr/bin/env bash")
        assert "--wrap" not in text
        assert "/usr/bin/sbatch" not in text

    source = paths[RUNNERS[0]].read_text(encoding="utf-8")
    screen = paths[RUNNERS[1]].read_text(encoding="utf-8")
    select = paths[RUNNERS[2]].read_text(encoding="utf-8")
    clock_final = paths[RUNNERS[3]].read_text(encoding="utf-8")
    oracle_final = paths[RUNNERS[4]].read_text(encoding="utf-8")
    finalize = paths[RUNNERS[5]].read_text(encoding="utf-8")
    submit = paths[SUBMITTER].read_text(encoding="utf-8")

    assert "#SBATCH --array=0-4%5" in source
    assert "#SBATCH --array=0-59%50" in screen
    assert "#SBATCH --array=0-19%20" in clock_final
    assert "#SBATCH --array=0-19%20" in oracle_final
    for text in (source, screen, select, clock_final, oracle_final, finalize):
        assert "--cpus-per-task=1" in text
        assert "module load conda/latest" in text
        assert 'eval "$(conda shell.bash hook)"' in text
        assert "conda activate writingring-gpu" in text
        assert "conda activate writingring-viz" in text
        for variable in ("OMP_NUM_THREADS=1", "MKL_NUM_THREADS=1", "OPENBLAS_NUM_THREADS=1", "NUMEXPR_NUM_THREADS=1"):
            assert variable in text
    assert "command -v sbatch" in submit
    assert 'dependency="afterok:${SOURCE_JOB}"' in submit
    assert 'dependency="afterok:${SCREEN_JOB}"' in submit
    assert submit.count('dependency="afterok:${SELECT_JOB}"') == 2
    assert 'dependency="afterok:${CLOCK_FINAL_JOB}:${ORACLE_FINAL_JOB}"' in submit


def test_readme_and_notebook_are_analysis_artifact_driven() -> None:
    assert README.is_file()
    readme = README.read_text(encoding="utf-8")
    assert "60 Clock screen tasks" in readme
    assert "20 Clock final evaluation tasks" in readme
    assert "20 Oracle train -> evaluate tasks" in readme
    assert "analysis-only" in readme

    assert NOTEBOOK.is_file()
    notebook_text = NOTEBOOK.read_text(encoding="utf-8")
    notebook = json.loads(notebook_text)
    assert notebook["nbformat"] == 4
    for artifact in (
        "selection.json", "screen_summary.csv", "final_summary.csv", "final_paired_deltas.csv",
        "residual_diagnostics.csv", "residual_alignment.csv", "residual_only_probes.csv",
        "generalization_gap.csv", "clock_progress_alignment.csv", "same_what_same_coordinate.csv",
        "manifest.json",
    ):
        assert artifact in notebook_text
    assert "analysis-only" in notebook_text.lower()
    assert "train_screen_one(" not in notebook_text
    assert "train_oracle_final_one(" not in notebook_text
    assert "run_screen_one(" not in notebook_text
    assert "select_screen(" not in notebook_text
    assert "sbatch" not in notebook_text
