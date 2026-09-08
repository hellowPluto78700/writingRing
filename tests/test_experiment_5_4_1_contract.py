from __future__ import annotations

import inspect
import json
from pathlib import Path

import torch

from scripts import experiment_5_4_1_constrained_conjunction_residual as exp541


REPO_ROOT = Path(__file__).resolve().parents[1]
PREP = REPO_ROOT / "scripts/bash_script/SNN_Bash/prepare_exp_5_4_1_source_cpu_array.bash"
BASE_RUNNER = REPO_ROOT / "scripts/bash_script/SNN_Bash/run_exp_5_4_1_base_cpu_array.bash"
RESIDUAL_RUNNER = REPO_ROOT / "scripts/bash_script/SNN_Bash/run_exp_5_4_1_cpu_array.bash"
FINALIZER = REPO_ROOT / "scripts/bash_script/SNN_Bash/finalize_exp_5_4_1_cpu.bash"
SUBMIT = REPO_ROOT / "scripts/bash_script/SNN_Bash/submit_exp_5_4_1_cpu.bash"
README = REPO_ROOT / "scripts/experiment_5_4_1/README.md"
NOTEBOOK = REPO_ROOT / "notebooks/experiment_5_4_1_constrained_conjunction_residual.ipynb"


def _base_model() -> exp541.DirectWhatBase:
    torch.manual_seed(7)
    return exp541.DirectWhatBase(n_classes=12)


def _residual_model() -> exp541.ResidualConjunctionClassifier:
    model = exp541.ResidualConjunctionClassifier(_base_model(), n_classes=12)
    with torch.no_grad():
        model.what_selector.weight.fill_(0.05)
        model.when_selector.weight.fill_(0.05)
        model.residual_output.weight.zero_()
    return model


def test_protocol_and_five_seed_run_mapping() -> None:
    assert exp541.PROTOCOL_VERSION == "direct_what_conjunction_residual_v2"
    assert exp541.SEEDS == (11, 23, 37, 53, 71)
    assert exp541.WHAT_WIDTH == 128
    assert exp541.WHEN_WIDTH == 64
    assert exp541.CONTEXT_WIDTH == 128
    specs = exp541.run_specs()
    assert len(specs) == 5
    assert len({spec.key for spec in specs}) == 5
    assert [spec.seed for spec in specs] == list(exp541.SEEDS)
    assert all(spec.condition == exp541.CONDITION for spec in specs)


def test_direct_what_base_has_no_extra_fusion_snn() -> None:
    model = _base_model()
    assert model.output_projection.in_features == 128
    assert model.output_projection.out_features == 12
    assert model.output_projection.bias is None
    assert tuple(model.class_bias.shape) == (12,)
    assert exp541.base_parameter_count(model) == 128 * 12 + 12
    source = inspect.getsource(exp541.DirectWhatBase)
    assert "FusionClassifier" not in source
    assert "membrane" not in source
    assert "synaptic" not in source
    assert "output_projection(what)" in source
    assert "evidence.sum(dim=1) + self.class_bias" in source


def test_direct_base_masks_padding_exactly() -> None:
    model = _base_model().eval()
    what = torch.randint(0, 2, (2, 7, 128), dtype=torch.float32)
    lengths = torch.tensor([3, 6])
    with torch.no_grad():
        original, _ = model(what, lengths)
        changed = what.clone()
        changed[0, 3:] = 1.0 - changed[0, 3:]
        changed[1, 6:] = 1.0 - changed[1, 6:]
        altered, _ = model(changed, lengths)
    assert torch.equal(original, altered)


def test_base_is_frozen_and_residual_output_is_zero_initializable() -> None:
    model = _residual_model()
    assert all(not parameter.requires_grad for parameter in model.base_model.parameters())
    assert all(parameter.requires_grad for parameter in model.what_selector.parameters())
    assert all(parameter.requires_grad for parameter in model.when_selector.parameters())
    assert all(parameter.requires_grad for parameter in model.residual_output.parameters())
    assert torch.count_nonzero(model.residual_output.weight) == 0
    counts = exp541.parameter_counts(model)
    assert counts["frozen_base"] == 128 * 12 + 12
    assert counts["what_selector"] == 128 * 128
    assert counts["when_selector"] == 64 * 128
    assert counts["residual_output"] == 128 * 12
    assert counts["trainable_total"] == 26112
    assert counts["stored_total"] == 26112 + 1548


def test_epoch_zero_logits_exactly_equal_direct_what_base() -> None:
    model = _residual_model().eval()
    what = torch.ones(2, 6, 128)
    when = torch.ones(2, 6, 64)
    lengths = torch.tensor([6, 4])
    with torch.no_grad():
        expected, _ = model.base_model(what, lengths)
        actual, trajectory = model(what, when, lengths, return_trajectory=True)
    assert trajectory is not None
    assert torch.equal(actual, expected)
    assert torch.count_nonzero(trajectory.residual_logits) == 0
    source = inspect.getsource(exp541.train_residual)
    assert "best_epoch = 0" in source
    assert "epoch0_val" in source


def test_conjunction_requires_both_direct_what_and_when() -> None:
    model = _residual_model().eval()
    with torch.no_grad():
        model.residual_output.weight.fill_(1.0)
        model.base_model.output_projection.weight.zero_()
        model.base_model.class_bias.zero_()
    what = torch.ones(2, 5, 128)
    when = torch.ones(2, 5, 64)
    lengths = torch.tensor([5, 3])
    with torch.no_grad():
        _, both = model(what, when, lengths, return_trajectory=True)
        _, no_when = model(what, torch.zeros_like(when), lengths, return_trajectory=True)
        _, no_what = model(
            what,
            when,
            lengths,
            zero_residual_what=True,
            return_trajectory=True,
        )
    assert both is not None and no_when is not None and no_what is not None
    assert torch.count_nonzero(both.conjunction_spikes) > 0
    assert torch.count_nonzero(no_when.conjunction_spikes) == 0
    assert torch.count_nonzero(no_when.residual_logits) == 0
    assert torch.count_nonzero(no_what.conjunction_spikes) == 0
    assert torch.count_nonzero(no_what.residual_logits) == 0


def test_residual_padding_changes_cannot_change_valid_logits() -> None:
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
        "residual_what_zero",
    )
    assert exp541.SHUFFLE_REPLICATES == 5
    assert exp541.CONJUNCTION_GAIN < exp541.CONJUNCTION_THRESHOLD
    assert 2 * exp541.CONJUNCTION_GAIN > exp541.CONJUNCTION_THRESHOLD


def test_multi_cpu_slurm_has_prep_base_residual_afterok_chain() -> None:
    prep = PREP.read_text(encoding="utf-8")
    base_runner = BASE_RUNNER.read_text(encoding="utf-8")
    residual_runner = RESIDUAL_RUNNER.read_text(encoding="utf-8")
    finalizer = FINALIZER.read_text(encoding="utf-8")
    submit = SUBMIT.read_text(encoding="utf-8")
    for text in (prep, base_runner, residual_runner):
        assert "#SBATCH --array=0-4%5" in text
    for text in (prep, base_runner, residual_runner, finalizer):
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
    assert "prepare-inputs" in prep
    assert "train-base" in base_runner
    assert "run-one" in residual_runner
    assert "finalize" in finalizer
    assert 'afterok:${PREP_JOB}' in submit
    assert 'afterok:${BASE_JOB}' in submit
    assert 'afterok:${RUN_JOB}' in submit
    assert "direct-WHAT-base array" in submit


def test_notebook_is_analysis_only_and_reads_v2_outputs() -> None:
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
        "base_runs.csv",
        "base_histories.csv",
        "runs.csv",
        "histories.csv",
        "ablation_runs.csv",
        "activity_runs.csv",
        "paired_deltas.csv",
        "manifest.json",
        "direct_what_wholecount",
        "delta_vs_base_mean",
        "epoch0_selected",
        "reset_when",
        "when_circular_shift",
        "residual_what_zero",
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
        "train-base",
        "prepare-inputs",
    ):
        assert forbidden not in joined


def test_readme_freezes_clean_direct_what_question() -> None:
    text = README.read_text(encoding="utf-8")
    for token in (
        "Direct WHAT base + constrained WHAT×WHEN residual conjunction",
        "No Exp5.4 Fusion-LIF checkpoint is used as the base classifier",
        "bias-free Linear(128 -> 12)",
        "There is no additional Fusion-LIF",
        "C_t \\approx Q_t^{WHAT}\\land Q_t^{WHEN}",
        "epoch 0",
        "residual_what_zero",
        "when_circular_shift",
        "final whole-sequence CE",
        "Five independent seeds",
        "afterok",
        "analysis-only",
    ):
        assert token in text
