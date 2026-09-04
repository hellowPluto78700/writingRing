from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn.functional as F

from scripts import experiment_5_2_1_ff_multitau_objectives as exp521
from scripts import experiment_5_2_frozen_local_tauR_sweep as exp52


REPO_ROOT = Path(__file__).resolve().parents[1]
RUNNER = REPO_ROOT / "scripts" / "bash_script" / "SNN_Bash" / "run_exp_5_2_1_cpu_array.bash"
FINALIZER = REPO_ROOT / "scripts" / "bash_script" / "SNN_Bash" / "finalize_exp_5_2_1_cpu.bash"
SUBMIT = REPO_ROOT / "scripts" / "bash_script" / "SNN_Bash" / "submit_exp_5_2_1_cpu.bash"
README = REPO_ROOT / "scripts" / "experiment_5_2_1" / "README.md"
NOTEBOOK = REPO_ROOT / "notebooks" / "experiment_5_2_1_ff_multitau_objectives.ipynb"


def _fake_data() -> SimpleNamespace:
    return SimpleNamespace(labels=list(range(12)), T=64, fs=64.0, bin_steps=16)


def test_factorial_run_matrix_and_source_reuse_contract() -> None:
    specs = exp521.run_specs()
    assert exp521.PROTOCOL_VERSION == "frozen_exp3_l2_ff_multitau_objectives_v1"
    assert exp521.SOURCE_EXPERIMENT_ID == exp52.EXPERIMENT_ID
    assert exp521.SOURCE_PROTOCOL_VERSION == exp52.PROTOCOL_VERSION
    assert exp521.SEEDS == (11, 23, 37, 53, 71)
    assert exp521.MEMORY_MODES == ("single_shift7", "multitau_4567")
    assert exp521.OBJECTIVES == ("endpoint_ce", "endpoint_fixed250_aux_ce")
    assert len(specs) == 20
    assert len({spec.key for spec in specs}) == 20
    assert exp521.source_results_dir(Path("/tmp/repo")) == exp52.results_dir(Path("/tmp/repo"))


def test_single_and_multitau_profiles_are_balanced() -> None:
    single = exp521.beta_vector("single_shift7")
    assert single.shape == (128,)
    assert torch.allclose(single, torch.full_like(single, exp52.beta_from_shift(7)))

    multi = exp521.beta_vector("multitau_4567")
    slices = exp521.memory_group_slices("multitau_4567")
    assert tuple(slices) == (4, 5, 6, 7)
    for shift, group in slices.items():
        assert group.stop - group.start == 32
        assert torch.allclose(
            multi[group], torch.full_like(multi[group], exp52.beta_from_shift(shift))
        )


def test_decoder_is_ff_uend_readout_and_aux_head_is_training_only() -> None:
    old = exp521.FFTemporalDecoder("single_shift7", "endpoint_ce", 12, 64, 64.0, 16)
    new = exp521.FFTemporalDecoder(
        "multitau_4567", "endpoint_fixed250_aux_ce", 12, 64, 64.0, 16
    )
    assert old.input_hidden.in_features == 128
    assert old.input_hidden.out_features == 128
    assert old.endpoint_head.in_features == 128
    assert old.endpoint_head.out_features == 12
    assert old.aux_head is None
    assert new.aux_head is not None
    assert new.aux_head.in_features == 4 * 128
    assert not hasattr(old, "recurrent")
    assert not hasattr(new, "recurrent")
    assert not hasattr(new, "output_lif")


def test_objectives_keep_endpoint_ce_as_primary_deployed_loss() -> None:
    x = torch.zeros(2, 16, 128)
    lengths = torch.tensor([8, 16])
    y = torch.tensor([0, 1])

    old = exp521.FFTemporalDecoder("single_shift7", "endpoint_ce", 3, 16, 64.0, 4)
    old_total, old_endpoint, old_aux, old_logits, _, old_mem = old.loss_logits(x, lengths, y)
    expected_old = F.cross_entropy(
        old.endpoint_head(exp52._valid_endpoint(old_mem, lengths)), y
    )
    assert old_aux is None
    assert torch.allclose(old_endpoint, expected_old)
    assert torch.allclose(old_total, expected_old)
    assert old_logits.shape == (2, 3)

    new = exp521.FFTemporalDecoder(
        "multitau_4567", "endpoint_fixed250_aux_ce", 3, 16, 64.0, 4
    )
    new_total, new_endpoint, new_aux, _, _, _ = new.loss_logits(x, lengths, y)
    assert new_aux is not None
    assert torch.allclose(new_total, new_endpoint + exp521.AUX_LAMBDA * new_aux)


def test_initialization_pairs_deployed_weights_across_factorial_conditions() -> None:
    device = torch.device("cpu")
    data = _fake_data()
    specs = [
        exp521.RunSpec("single_shift7", "endpoint_ce", 11),
        exp521.RunSpec("multitau_4567", "endpoint_ce", 11),
        exp521.RunSpec("single_shift7", "endpoint_fixed250_aux_ce", 11),
        exp521.RunSpec("multitau_4567", "endpoint_fixed250_aux_ce", 11),
    ]
    models = [exp521._initialize_model(spec, data, device) for spec in specs]
    for model in models[1:]:
        assert torch.equal(models[0].input_hidden.weight, model.input_hidden.weight)
        assert torch.equal(models[0].endpoint_head.weight, model.endpoint_head.weight)
        assert torch.equal(models[0].endpoint_head.bias, model.endpoint_head.bias)


def test_probe_contract_matches_exp52() -> None:
    assert exp521.TEMPORAL_PROBES == exp52.TEMPORAL_PROBES
    assert exp521.LOCAL_REFERENCE_PROBES == exp52.LOCAL_REFERENCE_PROBES
    assert exp521.TEMPORAL_PROBES == (
        "hidden_whole_count",
        "hidden_fixed250_ordered",
        "hidden_relative10_ordered",
        "hidden_uend",
    )


def test_multi_cpu_and_afterok_contract() -> None:
    runner = RUNNER.read_text(encoding="utf-8")
    finalizer = FINALIZER.read_text(encoding="utf-8")
    submit = SUBMIT.read_text(encoding="utf-8")
    assert "#SBATCH --array=0-19%20" in runner
    for text in (runner, finalizer):
        assert "#SBATCH --cpus-per-task=1" in text
        assert "module load conda/latest" in text
        assert "conda activate writingring-gpu" in text
        for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
            assert f"export {name}=1" in text
    assert 'afterok:${RUN_JOB}' in submit
    assert "20 runs" in submit
    assert "reuse" in submit.lower()


def test_notebook_is_analysis_only_and_contains_factorial_diagnostics() -> None:
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    assert notebook["nbformat"] == 4
    sources: list[str] = []
    for index, cell in enumerate(notebook["cells"]):
        source = "".join(cell.get("source", []))
        sources.append(source)
        if cell.get("cell_type") == "code":
            compile(source, f"exp521-notebook-cell-{index}", "exec")
    joined = "\n".join(sources)
    for token in (
        "runs.csv",
        "histories.csv",
        "local_reference.csv",
        "manifest.json",
        "memory_effect",
        "objective_effect",
        "interaction_effect",
        "head_utilization_gap",
        "trajectory_endpoint_gap",
        "hidden_uend_test_ba",
        "hidden_relative10_ordered_test_ba",
        "native_val_balanced_accuracy",
        "plt.subplots",
    ):
        assert token in joined
    for forbidden in (
        "optimizer.step(",
        ".backward()",
        "subprocess",
        "sbatch",
        "run-one",
    ):
        assert forbidden not in joined


def test_readme_states_factorial_and_deployment_contract() -> None:
    text = README.read_text(encoding="utf-8")
    for token in (
        "2 × 2 factorial",
        "single_shift7",
        "multitau_4567",
        "endpoint_ce",
        "endpoint_fixed250_aux_ce",
        "lambda_aux = 1.0",
        "hidden Uend -> Linear",
        "20 independent runs",
        "0-19%20",
        "analysis-only",
        "reuses the committed Experiment 5.2 frozen L2 caches",
    ):
        assert token in text
