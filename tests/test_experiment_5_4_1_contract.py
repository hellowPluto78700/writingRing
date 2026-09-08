from __future__ import annotations

import inspect
import json
from pathlib import Path

import torch

from scripts import experiment_5_4_snn_native_fusion as exp54
from scripts import experiment_5_4_1_constrained_conjunction_residual as exp541


REPO_ROOT = Path(__file__).resolve().parents[1]
PREP = REPO_ROOT / "scripts/bash_script/SNN_Bash/prepare_exp_5_4_1_source_cpu_array.bash"
RUNNER = REPO_ROOT / "scripts/bash_script/SNN_Bash/run_exp_5_4_1_cpu_array.bash"
FINALIZER = REPO_ROOT / "scripts/bash_script/SNN_Bash/finalize_exp_5_4_1_cpu.bash"
SUBMIT = REPO_ROOT / "scripts/bash_script/SNN_Bash/submit_exp_5_4_1_cpu.bash"
README = REPO_ROOT / "scripts/experiment_5_4_1/README.md"
NOTEBOOK = REPO_ROOT / "notebooks/experiment_5_4_1_constrained_conjunction_residual.ipynb"


def _base_model() -> exp54.FusionClassifier:
    torch.manual_seed(7)
    return exp54.FusionClassifier(n_classes=12, fs=64.0, fusion_mode="lif")


def _residual_model() -> exp541.ResidualConjunctionClassifier:
    model = exp541.ResidualConjunctionClassifier(_base_model(), n_classes=12)
    with torch.no_grad():
        model.base_selector.weight.fill_(0.05)
        model.when_selector.weight.fill_(0.05)
        model.residual_output.weight.zero_()
    return model


def test_protocol_and_five_seed_run_mapping() -> None:
    assert exp541.PROTOCOL_VERSION == "constrained_conjunction_residual_v1"
    assert exp541.SEEDS == (11, 23, 37, 53, 71)
    assert exp541.WHAT_WIDTH == 128
    assert exp541.WHEN_WIDTH == 64
    assert exp541.BASE_FUSION_WIDTH == 128
    assert exp541.CONTEXT_WIDTH == 128
    specs = exp541.run_specs()
    assert len(specs) == 5
    assert len({spec.key for spec in specs}) == 5
    assert [spec.seed for spec in specs] == list(exp541.SEEDS)
    assert all(spec.condition == exp541.CONDITION for spec in specs)


def test_base_is_frozen_and_residual_output_is_zero_initializable() -> None:
    model = _residual_model()
    assert all(not parameter.requires_grad for parameter in model.base_model.parameters())
    assert all(parameter.requires_grad for parameter in model.base_selector.parameters())
    assert all(parameter.requires_grad for parameter in model.when_selector.parameters())
    assert all(parameter.requires_grad for parameter in model.residual_output.parameters())
    assert torch.count_nonzero(model.residual_output.weight) == 0

    counts = exp541.parameter_counts(model)
    assert counts["base_selector"] == 128 * 128
    assert counts["when_selector"] == 64 * 128
    assert counts["residual_output"] == 128 * 12
    assert counts["trainable_total"] == 26112
    assert counts["frozen_base"] == 26124


def test_epoch_zero_logits_exactly_equal_frozen_what_base() -> None:
    model = _residual_model().eval()
    what = torch.ones(2, 6, 128)
    when = torch.ones(2, 6, 64)
    lengths = torch.tensor([6, 4])
    with torch.no_grad():
        expected, _ = model.base_model(
            what,
            torch.zeros_like(when),
            lengths,
            return_fusion_trajectory=True,
        )
        actual, trajectory = model(what, when, lengths, return_trajectory=True)
    assert trajectory is not None
    assert torch.equal(actual, expected)
    assert torch.count_nonzero(trajectory.residual_logits) == 0
    source = inspect.getsource(exp541.train_one)
    assert "best_epoch = 0" in source
    assert "epoch0_val" in source


def test_conjunction_requires_both_sides_and_cannot_be_when_only_classifier() -> None:
    model = _residual_model().eval()
    with torch.no_grad():
        model.residual_output.weight.fill_(1.0)
        model.base_model.what_projection.weight.fill_(0.1)
        model.base_model.when_projection.weight.zero_()
        model.base_model.output_projection.weight.zero_()
        model.base_model.class_bias.zero_()

    what = torch.ones(2, 5, 128)
    when = torch.ones(2, 5, 64)
    lengths = torch.tensor([5, 3])

    with torch.no_grad():
        _, both = model(what, when, lengths, return_trajectory=True)
        _, no_when = model(what, torch.zeros_like(when), lengths, return_trajectory=True)
        _, no_base = model(
            what,
            when,
            lengths,
            zero_base_context=True,
            return_trajectory=True,
        )
    assert both is not None and no_when is not None and no_base is not None
    assert torch.count_nonzero(both.conjunction_spikes) > 0
    assert torch.count_nonzero(no_when.conjunction_spikes) == 0
    assert torch.count_nonzero(no_when.residual_logits) == 0
    assert torch.count_nonzero(no_base.conjunction_spikes) == 0
    assert torch.count_nonzero(no_base.residual_logits) == 0


def test_padding_changes_cannot_change_valid_whole_sequence_logits() -> None:
    model = _residual_model().eval()
    with torch.no_grad():
        model.residual_output.weight.normal_(mean=0.0, std=0.05)
    what = torch.randint(0, 2, (2, 7, 128), dtype=torch.float32)
    when = torch.randint(0, 2, (2, 7, 64), dtype=torch.float32)
    lengths = torch.tensor([3, 6])
    with torch.no_grad():
        original, _ = model(what, when, lengths)
        changed_what = what.clone()
        changed_when = when.clone()
        changed_what[0, 3:] = 1.0 - changed_what[0, 3:]
        changed_when[0, 3:] = 1.0 - changed_when[0, 3:]
        changed_what[1, 6:] = 1.0 - changed_what[1, 6:]
        changed_when[1, 6:] = 1.0 - changed_when[1, 6:]
        changed, _ = model(changed_what, changed_when, lengths)
    assert torch.equal(original, changed)


def test_ablation_contract_includes_history_alignment_and_two_zero_sides() -> None:
    assert exp541.ABLATIONS == (
        "ordered",
        "reset_when",
        "when_zero",
        "when_shuffle",
        "when_circular_shift",
        "base_context_zero",
    )
    assert exp541.SHUFFLE_REPLICATES == 5
    assert exp541.CONJUNCTION_GAIN < exp541.CONJUNCTION_THRESHOLD
    assert 2 * exp541.CONJUNCTION_GAIN > exp541.CONJUNCTION_THRESHOLD


def test_multi_cpu_slurm_and_afterok_contract() -> None:
    prep = PREP.read_text(encoding="utf-8")
    runner = RUNNER.read_text(encoding="utf-8")
    finalizer = FINALIZER.read_text(encoding="utf-8")
    submit = SUBMIT.read_text(encoding="utf-8")
    assert "#SBATCH --array=0-4%5" in prep
    assert "#SBATCH --array=0-4%5" in runner
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
    assert "prepare-source" in prep
    assert "run-one" in runner
    assert "finalize" in finalizer
    assert 'afterok:${PREP_JOB}' in submit
    assert 'afterok:${RUN_JOB}' in submit
    assert "5 tasks" in submit


def test_notebook_is_analysis_only_and_reads_final_outputs() -> None:
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    assert notebook["nbformat"] == 4
    sources: list[str] = []
    for index, cell in enumerate(notebook["cells"]):
        source = "".join(cell.get("source", []))
        sources.append(source)
        if cell.get("cell_type") == "code":
            compile(source, f"exp541-notebook-cell-{index}", "exec")
    joined = "\n".join(sources)
    for token in (
        "runs.csv",
        "histories.csv",
        "ablation_runs.csv",
        "activity_runs.csv",
        "source_runs.csv",
        "paired_deltas.csv",
        "manifest.json",
        "delta_vs_base_mean",
        "epoch0_selected",
        "reset_when",
        "when_circular_shift",
        "base_context_zero",
        "residual_abs_max",
        "plt.subplots",
    ):
        assert token in joined
    for forbidden in (
        "optimizer.step(",
        ".backward()",
        "subprocess",
        "sbatch",
        "run-one",
        "prepare-source",
    ):
        assert forbidden not in joined


def test_readme_freezes_incremental_conjunction_question() -> None:
    text = README.read_text(encoding="utf-8")
    for token in (
        "Constrained WHAT×WHEN residual conjunction",
        "what_only_lif",
        "does not read raw WHAT",
        "C_t \\approx B_t \\land H_t",
        "epoch 0",
        "exactly zero",
        "base_context_zero",
        "when_circular_shift",
        "final whole-sequence CE only",
        "Five independent runs",
        "afterok",
        "analysis-only",
    ):
        assert token in text
