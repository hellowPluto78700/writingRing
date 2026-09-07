from __future__ import annotations

import json
from pathlib import Path

import torch

from scripts import experiment_5_3_2_3_ff_before_rsnn as exp5323


REPO_ROOT = Path(__file__).resolve().parents[1]
PREP = REPO_ROOT / "scripts/bash_script/SNN_Bash/prepare_exp_5_3_2_3_local_cpu_array.bash"
RUNNER = REPO_ROOT / "scripts/bash_script/SNN_Bash/run_exp_5_3_2_3_cpu_array.bash"
FINALIZER = REPO_ROOT / "scripts/bash_script/SNN_Bash/finalize_exp_5_3_2_3_cpu.bash"
SUBMIT = REPO_ROOT / "scripts/bash_script/SNN_Bash/submit_exp_5_3_2_3_cpu.bash"
README = REPO_ROOT / "scripts/experiment_5_3_2_3/README.md"
NOTEBOOK = REPO_ROOT / "notebooks/experiment_5_3_2_3_ff_before_rsnn.ipynb"


def test_condition_matrix_and_direct_reuse_contract() -> None:
    assert exp5323.PROTOCOL_VERSION == "ff_before_rsnn_v1"
    assert exp5323.SEEDS == (11, 23, 37, 53, 71)
    assert exp5323.RSNN_WIDTH == 64
    assert exp5323.ALL_CONDITIONS == (
        "direct_rsnn64",
        "linear64_rsnn64",
        "ffsnn64_rsnn64",
        "ffsnn128_rsnn64",
    )
    assert exp5323.TRAINABLE_CONDITIONS == (
        "linear64_rsnn64",
        "ffsnn64_rsnn64",
        "ffsnn128_rsnn64",
    )
    assert exp5323.OBJECTIVE.name == "joint"
    assert (exp5323.OBJECTIVE.phase_weight, exp5323.OBJECTIVE.progress_weight) == (1.0, 1.0)

    train_specs = exp5323.run_specs()
    assert len(train_specs) == 15
    assert len({spec.key for spec in train_specs}) == 15
    assert [(spec.condition, spec.seed) for spec in train_specs[:5]] == [
        (exp5323.LINEAR_CONTROL_CONDITION, seed) for seed in exp5323.SEEDS
    ]
    assert [(spec.condition, spec.seed) for spec in train_specs[5:10]] == [
        (exp5323.FF64_CONDITION, seed) for seed in exp5323.SEEDS
    ]
    assert [(spec.condition, spec.seed) for spec in train_specs[10:15]] == [
        (exp5323.FF128_CONDITION, seed) for seed in exp5323.SEEDS
    ]
    assert all(spec.condition != exp5323.DIRECT_CONDITION for spec in train_specs)
    assert len(exp5323.all_specs()) == 20


def test_front_end_architectures_keep_rsnn64_fixed() -> None:
    expected_decay = exp5323.parent.decay_from_shift(exp5323.SHORT_SHIFT)
    for condition, ff_width in (
        (exp5323.LINEAR_CONTROL_CONDITION, 64),
        (exp5323.FF64_CONDITION, 64),
        (exp5323.FF128_CONDITION, 128),
    ):
        model = exp5323.FrontEndWhenNet(condition, fs=64.0)
        assert model.front_projection.in_features == 128
        assert model.front_projection.out_features == ff_width
        assert model.front_projection.bias is None
        assert model.input_projection.in_features == ff_width
        assert model.input_projection.out_features == 64
        assert model.input_projection.bias is None
        assert model.recurrent.in_features == model.recurrent.out_features == 64
        assert model.recurrent.bias is None
        assert model.phase_head.in_features == 64
        assert model.progress_head.in_features == 64
        assert torch.allclose(model.alpha_vector, torch.full((64,), expected_decay))
        assert torch.allclose(model.beta_vector, torch.full((64,), expected_decay))

        if condition == exp5323.LINEAR_CONTROL_CONDITION:
            assert model.front_end == "linear"
            assert not hasattr(model, "front_alpha_vector")
            assert not hasattr(model, "front_beta_vector")
        else:
            assert model.front_end == "ff_snn"
            assert torch.allclose(
                model.front_alpha_vector,
                torch.full((ff_width,), expected_decay),
            )
            assert torch.allclose(
                model.front_beta_vector,
                torch.full((ff_width,), expected_decay),
            )


def test_linear64_and_ffsnn64_have_paired_initial_weights() -> None:
    linear = exp5323._initialize_model(
        exp5323.LINEAR_CONTROL_CONDITION,
        23,
        64.0,
        torch.device("cpu"),
    )
    ffsnn = exp5323._initialize_model(
        exp5323.FF64_CONDITION,
        23,
        64.0,
        torch.device("cpu"),
    )
    for name in (
        "front_projection",
        "input_projection",
        "recurrent",
        "phase_head",
        "progress_head",
    ):
        left = getattr(linear, name).state_dict()
        right = getattr(ffsnn, name).state_dict()
        assert left.keys() == right.keys()
        for key in left:
            assert torch.equal(left[key], right[key])


def test_parameter_count_keeps_linear64_and_ffsnn64_exactly_matched() -> None:
    linear = exp5323.FrontEndWhenNet(exp5323.LINEAR_CONTROL_CONDITION, fs=64.0)
    ffsnn64 = exp5323.FrontEndWhenNet(exp5323.FF64_CONDITION, fs=64.0)
    ffsnn128 = exp5323.FrontEndWhenNet(exp5323.FF128_CONDITION, fs=64.0)
    linear_counts = exp5323.parameter_counts(linear)
    ff64_counts = exp5323.parameter_counts(ffsnn64)
    ff128_counts = exp5323.parameter_counts(ffsnn128)

    assert linear_counts == ff64_counts
    assert linear_counts["front_projection"] == 128 * 64
    assert linear_counts["input_projection"] == 64 * 64
    assert linear_counts["recurrent"] == 64 * 64
    assert linear_counts["phase_head"] == 10 * 64 + 10
    assert linear_counts["progress_head"] == 64 + 1
    assert linear_counts["trainable_total"] == 17099

    assert ff128_counts["front_projection"] == 128 * 128
    assert ff128_counts["input_projection"] == 128 * 64
    assert ff128_counts["recurrent"] == 64 * 64
    assert ff128_counts["trainable_total"] == 29387


def test_joint_loss_is_padding_invariant_for_ff_snn() -> None:
    model = exp5323._initialize_model(
        exp5323.FF64_CONDITION,
        11,
        64.0,
        torch.device("cpu"),
    )
    x = torch.randn(2, 7, 128)
    lengths = torch.tensor([3, 6])
    total, phase, progress, logits, progress_pred, trajectory = model.loss_components(x, lengths)
    assert torch.allclose(total, phase + progress)
    assert logits.shape == (2, 7, 10)
    assert progress_pred.shape == (2, 7)
    assert trajectory.membranes.shape == (2, 7, 64)

    changed = x.clone()
    changed[0, 3:] = torch.randn_like(changed[0, 3:]) * 100.0
    changed[1, 6:] = torch.randn_like(changed[1, 6:]) * 100.0
    changed_total, changed_phase, changed_progress, *_ = model.loss_components(changed, lengths)
    assert torch.allclose(total, changed_total)
    assert torch.allclose(phase, changed_phase)
    assert torch.allclose(progress, changed_progress)


def test_state_reset_removes_both_ff_and_recurrent_history_but_keeps_current_input() -> None:
    model = exp5323._initialize_model(
        exp5323.FF64_CONDITION,
        37,
        64.0,
        torch.device("cpu"),
    )
    x = torch.randn(2, 6, 128)
    lengths = torch.tensor([6, 6])
    ordered = model.forward_trajectory(x, lengths, reset_state_each_step=False)
    reset = model.forward_trajectory(x, lengths, reset_state_each_step=True)
    assert ordered.membranes.shape == reset.membranes.shape == (2, 6, 64)
    assert not torch.equal(ordered.membranes[:, 1:], reset.membranes[:, 1:])

    one_step = model.forward_trajectory(
        x[:, :1],
        torch.ones(2, dtype=torch.long),
        reset_state_each_step=True,
    )
    assert torch.allclose(reset.membranes[:, :1], one_step.membranes)


def test_loader_seed_remains_architecture_independent() -> None:
    for seed in exp5323.SEEDS:
        assert exp5323._loader_seed(seed, "train") == exp5323.base.dseed(
            seed,
            "exp5_3_2",
            "train",
            "loader",
        )


def test_direct_reference_is_exact_exp5322_h64() -> None:
    for seed in exp5323.SEEDS:
        source = exp5323._direct_width_spec(seed)
        assert source.hidden_width == 64
        assert source.seed == seed
        assert source.condition == "rsnn_h64"


def test_multi_cpu_afterok_and_environment_contract() -> None:
    prep = PREP.read_text(encoding="utf-8")
    runner = RUNNER.read_text(encoding="utf-8")
    finalizer = FINALIZER.read_text(encoding="utf-8")
    submit = SUBMIT.read_text(encoding="utf-8")

    assert "#SBATCH --array=0-4%5" in prep
    assert "#SBATCH --array=0-14%15" in runner
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
    assert "15 train/evaluate runs" in submit


def test_notebook_is_analysis_only_and_reads_final_outputs() -> None:
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    assert notebook["nbformat"] == 4
    sources: list[str] = []
    for index, cell in enumerate(notebook["cells"]):
        source = "".join(cell.get("source", []))
        sources.append(source)
        if cell.get("cell_type") == "code":
            compile(source, f"exp5323-notebook-cell-{index}", "exec")
    joined = "\n".join(sources)

    for token in (
        "runs.csv",
        "histories.csv",
        "probe_runs.csv",
        "ablation_runs.csv",
        "history_gain_runs.csv",
        "activity_runs.csv",
        "representation_runs.csv",
        "baseline_runs.csv",
        "local_reference.csv",
        "paired_deltas.csv",
        "manifest.json",
        "Primary architecture comparison",
        "Paired deltas vs direct",
        "Spiking-transform attribution",
        "FF-width sensitivity",
        "Generalization",
        "History contribution",
        "SNN-native accessibility",
        "effective_dimension",
        "pca90_dimension",
        "strict_pareto_vs_direct",
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


def test_readme_freezes_structural_ablation_contract() -> None:
    text = README.read_text(encoding="utf-8")
    for token in (
        "Pre-recurrent FF-SNN vs direct RSNN64",
        "direct_rsnn64",
        "linear64_rsnn64",
        "ffsnn64_rsnn64",
        "ffsnn128_rsnn64",
        "identity-validated reuse",
        "The FF-SNN feeds **spikes** into the recurrent layer",
        "L_{WHEN}=L_{phase}+L_{progress}",
        "`T_i` is used **only to construct supervision targets and the valid mask**",
        "15 independent runs",
        "H_{reset}",
        "H_{shuffle}",
        "D_{eff}",
        "D_{90}",
        "paired_deltas.csv",
        "15/15",
        "afterok",
        "strict mean Pareto improvement",
        "analysis-only",
    ):
        assert token in text
