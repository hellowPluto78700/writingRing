from __future__ import annotations

import inspect
import json
from pathlib import Path

import torch

from scripts import experiment_5_4_snn_native_fusion as exp54


REPO_ROOT = Path(__file__).resolve().parents[1]
PREP = REPO_ROOT / "scripts/bash_script/SNN_Bash/prepare_exp_5_4_fusion_cpu_array.bash"
RUNNER = REPO_ROOT / "scripts/bash_script/SNN_Bash/run_exp_5_4_cpu_array.bash"
FINALIZER = REPO_ROOT / "scripts/bash_script/SNN_Bash/finalize_exp_5_4_cpu.bash"
SUBMIT = REPO_ROOT / "scripts/bash_script/SNN_Bash/submit_exp_5_4_cpu.bash"
README = REPO_ROOT / "scripts/experiment_5_4/README.md"
NOTEBOOK = REPO_ROOT / "notebooks/experiment_5_4_snn_native_fusion.ipynb"


def test_condition_matrix_and_source_contract() -> None:
    assert exp54.PROTOCOL_VERSION == "snn_native_what_when_fusion_v1"
    assert exp54.SEEDS == (11, 23, 37, 53, 71)
    assert exp54.WHAT_WIDTH == 128
    assert exp54.WHEN_WIDTH == 64
    assert exp54.FUSION_WIDTH == 128
    assert exp54.SOURCE_WHEN_CONDITION == exp54.exp5323.FF128_CONDITION
    assert exp54.CONDITIONS == (
        "what_only_lif",
        "when_only_lif",
        "what_elapsed_lif",
        "what_resetwhen_lif",
        "what_when_linear",
        "what_when_fusion_lif",
    )
    specs = exp54.run_specs()
    assert len(specs) == 30
    assert len({spec.key for spec in specs}) == 30
    for condition_index, condition in enumerate(exp54.CONDITIONS):
        block = specs[condition_index * 5 : (condition_index + 1) * 5]
        assert [(spec.condition, spec.seed) for spec in block] == [
            (condition, seed) for seed in exp54.SEEDS
        ]


def test_fusion_lif_is_short_memory_non_recurrent_and_bias_safe() -> None:
    model = exp54.FusionClassifier(n_classes=12, fs=64.0, fusion_mode="lif")
    expected_decay = exp54.parent.decay_from_shift(exp54.SHORT_SHIFT)
    assert model.what_projection.in_features == 128
    assert model.what_projection.out_features == 128
    assert model.what_projection.bias is None
    assert model.when_projection.in_features == 64
    assert model.when_projection.out_features == 128
    assert model.when_projection.bias is None
    assert model.output_projection.in_features == 128
    assert model.output_projection.out_features == 12
    assert model.output_projection.bias is None
    assert model.class_bias.shape == (12,)
    assert not hasattr(model, "recurrent")
    assert torch.allclose(model.alpha_vector, torch.full((128,), expected_decay))
    assert torch.allclose(model.beta_vector, torch.full((128,), expected_decay))


def test_linear_and_lif_controls_are_parameter_matched_and_paired() -> None:
    class DummyData:
        labels = list(range(12))
        fs = 64.0

    lif_spec = exp54.RunSpec(exp54.WHAT_WHEN_FUSION, 23)
    linear_spec = exp54.RunSpec(exp54.WHAT_WHEN_LINEAR, 23)
    lif = exp54._initialize_model(lif_spec, DummyData(), torch.device("cpu"))
    linear = exp54._initialize_model(linear_spec, DummyData(), torch.device("cpu"))
    assert exp54.parameter_counts(lif) == exp54.parameter_counts(linear)
    assert exp54.parameter_counts(lif) == {
        "what_projection": 128 * 128,
        "when_projection": 64 * 128,
        "output_projection": 128 * 12,
        "class_bias": 12,
        "trainable_total": 26124,
    }
    for name in ("what_projection", "when_projection", "output_projection"):
        left = getattr(lif, name).state_dict()
        right = getattr(linear, name).state_dict()
        assert left.keys() == right.keys()
        for key in left:
            assert torch.equal(left[key], right[key])
    assert torch.equal(lif.class_bias, linear.class_bias)


def test_class_bias_is_added_once_after_accumulation() -> None:
    model = exp54.FusionClassifier(n_classes=3, fs=64.0, fusion_mode="lif")
    with torch.no_grad():
        model.what_projection.weight.zero_()
        model.when_projection.weight.zero_()
        model.output_projection.weight.zero_()
        model.class_bias.copy_(torch.tensor([0.25, -0.5, 1.25]))
    what = torch.zeros(2, 6, 128)
    context = torch.zeros(2, 6, 64)
    lengths = torch.tensor([1, 6])
    logits, _ = model(what, context, lengths)
    expected = model.class_bias.unsqueeze(0).expand(2, -1)
    assert torch.allclose(logits, expected)


def test_padding_cannot_change_valid_fusion_evidence() -> None:
    torch.manual_seed(4)
    model = exp54.FusionClassifier(n_classes=4, fs=64.0, fusion_mode="lif")
    what = torch.randn(2, 7, 128)
    context = torch.randn(2, 7, 64)
    lengths = torch.tensor([3, 6])
    logits, _ = model(what, context, lengths)

    changed_what = what.clone()
    changed_context = context.clone()
    changed_what[0, 3:] = torch.randn_like(changed_what[0, 3:]) * 100.0
    changed_context[0, 3:] = torch.randn_like(changed_context[0, 3:]) * 100.0
    changed_what[1, 6:] = torch.randn_like(changed_what[1, 6:]) * 100.0
    changed_context[1, 6:] = torch.randn_like(changed_context[1, 6:]) * 100.0
    changed_logits, _ = model(changed_what, changed_context, lengths)
    assert torch.allclose(logits, changed_logits)


def test_elapsed_context_is_absolute_time_only() -> None:
    signature = inspect.signature(exp54.elapsed_context)
    assert "lengths" not in signature.parameters
    assert "final_duration" not in signature.parameters
    basis = exp54.elapsed_context(
        batch_size=3,
        steps=10,
        fs=64.0,
        train_max_elapsed_seconds=2.0,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    assert basis.shape == (3, 10, 64)
    assert torch.allclose(basis[0], basis[1])
    assert torch.allclose(basis[1], basis[2])


def test_context_conditions_select_expected_frozen_streams() -> None:
    what = torch.randn(2, 5, 128)
    ordered = torch.randn(2, 5, 64)
    reset = torch.randn(2, 5, 64)
    for condition in exp54.CONDITIONS:
        spec = exp54.RunSpec(condition, 11)
        selected_what, context = exp54._select_inputs(
            spec,
            what,
            ordered,
            reset,
            fs=64.0,
            train_max_elapsed_seconds=2.0,
        )
        assert selected_what.shape == what.shape
        assert context.shape == ordered.shape
        if condition == exp54.WHAT_ONLY:
            assert torch.equal(selected_what, what)
            assert torch.count_nonzero(context) == 0
        elif condition == exp54.WHEN_ONLY:
            assert torch.count_nonzero(selected_what) == 0
            assert torch.equal(context, ordered)
        elif condition == exp54.WHAT_RESETWHEN:
            assert torch.equal(context, reset)
        elif condition in (exp54.WHAT_WHEN_LINEAR, exp54.WHAT_WHEN_FUSION):
            assert torch.equal(context, ordered)


def test_multi_cpu_afterok_and_environment_contract() -> None:
    prep = PREP.read_text(encoding="utf-8")
    runner = RUNNER.read_text(encoding="utf-8")
    finalizer = FINALIZER.read_text(encoding="utf-8")
    submit = SUBMIT.read_text(encoding="utf-8")

    assert "#SBATCH --array=0-4%5" in prep
    assert "#SBATCH --array=0-29%30" in runner
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
    assert "30 train/evaluate runs" in submit


def test_notebook_is_analysis_only_and_reads_final_outputs() -> None:
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    assert notebook["nbformat"] == 4
    sources: list[str] = []
    for index, cell in enumerate(notebook["cells"]):
        source = "".join(cell.get("source", []))
        sources.append(source)
        if cell.get("cell_type") == "code":
            compile(source, f"exp54-notebook-cell-{index}", "exec")
    joined = "\n".join(sources)
    for token in (
        "runs.csv",
        "histories.csv",
        "ablation_runs.csv",
        "activity_runs.csv",
        "paired_deltas.csv",
        "manifest.json",
        "Primary architecture comparison",
        "Paired deltas",
        "WHEN alignment attribution",
        "when_circular_shift",
        "train_test_gap",
        "plt.subplots",
    ):
        assert token in joined
    for forbidden in (
        "optimizer.step(",
        ".backward()",
        "subprocess",
        "sbatch",
        "run-one",
        "prepare-fusion",
    ):
        assert forbidden not in joined


def test_readme_freezes_snn_native_fusion_contract() -> None:
    text = README.read_text(encoding="utf-8")
    for token in (
        "SNN-native causal WHAT–WHEN fusion",
        "what_only_lif",
        "when_only_lif",
        "what_elapsed_lif",
        "what_resetwhen_lif",
        "what_when_linear",
        "what_when_fusion_lif",
        "Fusion SNN128",
        "no recurrent weight matrix",
        "W_out is bias-free",
        "added **once after accumulation**",
        "final whole-sequence CE only",
        "absolute causal time only",
        "never receives final gesture duration",
        "when_zero",
        "when_shuffle",
        "when_circular_shift",
        "30 independent train/evaluate runs",
        "analysis-only",
    ):
        assert token in text
