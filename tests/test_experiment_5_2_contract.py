from __future__ import annotations

import json
import math
from pathlib import Path

import torch

from scripts import experiment_5_2_frozen_local_tauR_sweep as exp52


REPO_ROOT = Path(__file__).resolve().parents[1]
PREP = REPO_ROOT / "scripts" / "bash_script" / "SNN_Bash" / "prepare_exp_5_2_local_cpu_array.bash"
RUNNER = REPO_ROOT / "scripts" / "bash_script" / "SNN_Bash" / "run_exp_5_2_cpu_array.bash"
FINALIZER = REPO_ROOT / "scripts" / "bash_script" / "SNN_Bash" / "finalize_exp_5_2_cpu.bash"
SUBMIT = REPO_ROOT / "scripts" / "bash_script" / "SNN_Bash" / "submit_exp_5_2_cpu.bash"
README = REPO_ROOT / "scripts" / "experiment_5_2" / "README.md"
NOTEBOOK = REPO_ROOT / "notebooks" / "experiment_5_2_frozen_local_tauR_sweep.ipynb"


def test_run_matrix_and_frozen_local_contract() -> None:
    specs = exp52.run_specs()
    assert exp52.PROTOCOL_VERSION == "frozen_exp3_l2_endpoint_tauR_v1"
    assert exp52.LOCAL_SOURCE_CONDITION == exp52.exp501.EXP3_EXACT
    assert exp52.SEEDS == (11, 23, 37, 53, 71)
    assert exp52.ARCHITECTURES == ("ff", "rsnn")
    assert exp52.SHIFT_MEM_R == (3, 4, 5, 6, 7)
    assert exp52.TEMPORAL_WIDTH == 128
    assert exp52.LOCAL_WIDTH == 128
    assert len(specs) == 50
    assert len({spec.key for spec in specs}) == 50


def test_shift_tau_grid_is_repository_style_and_monotonic() -> None:
    expected_beta = {
        3: 0.875,
        4: 0.9375,
        5: 0.96875,
        6: 0.984375,
        7: 0.9921875,
    }
    taus = []
    for shift in exp52.SHIFT_MEM_R:
        assert math.isclose(exp52.beta_from_shift(shift), expected_beta[shift])
        taus.append(exp52.tau_ms_from_shift(shift, 64.0))
    assert all(left < right for left, right in zip(taus, taus[1:]))
    assert math.isclose(taus[0], 117.0, rel_tol=0.02)
    assert math.isclose(taus[-1], 1992.0, rel_tol=0.02)


def test_endpoint_decoder_is_hidden_uend_head_with_optional_dense_recurrence() -> None:
    ff = exp52.EndpointMemoryDecoder("ff", 4, 12, 64.0)
    rsnn = exp52.EndpointMemoryDecoder("rsnn", 4, 12, 64.0)
    assert ff.input_hidden.in_features == 128
    assert ff.input_hidden.out_features == 128
    assert ff.recurrent is None
    assert rsnn.recurrent is not None
    assert rsnn.recurrent.in_features == 128
    assert rsnn.recurrent.out_features == 128
    assert rsnn.endpoint_head.in_features == 128
    assert rsnn.endpoint_head.out_features == 12
    assert not hasattr(rsnn, "output_lif")
    assert not hasattr(rsnn, "output_spike")


def test_valid_endpoint_uses_length_minus_one_not_padding_end() -> None:
    seq = torch.tensor(
        [
            [[1.0], [2.0], [100.0], [200.0]],
            [[3.0], [4.0], [5.0], [300.0]],
        ]
    )
    endpoint = exp52._valid_endpoint(seq, torch.tensor([2, 3]))
    assert torch.equal(endpoint, torch.tensor([[2.0], [5.0]]))


def test_native_loss_is_endpoint_ce() -> None:
    model = exp52.EndpointMemoryDecoder("ff", 4, 3, 64.0)
    x = torch.zeros(2, 5, 128)
    lengths = torch.tensor([2, 4])
    y = torch.tensor([0, 1])
    loss, logits, spikes, membranes = model.loss_logits(x, lengths, y)
    expected_logits = model.endpoint_head(exp52._valid_endpoint(membranes, lengths))
    assert torch.allclose(logits, expected_logits)
    assert loss.ndim == 0
    assert spikes.shape == (2, 5, 128)
    assert membranes.shape == (2, 5, 128)


def test_initialization_is_paired_across_tau_and_architecture() -> None:
    device = torch.device("cpu")
    specs = [
        exp52.RunSpec("ff", 3, 11),
        exp52.RunSpec("ff", 7, 11),
        exp52.RunSpec("rsnn", 5, 11),
    ]
    models = [exp52._initialize_model(spec, 12, 64.0, device) for spec in specs]
    for model in models[1:]:
        assert torch.equal(models[0].input_hidden.weight, model.input_hidden.weight)
        assert torch.equal(models[0].endpoint_head.weight, model.endpoint_head.weight)
        assert torch.equal(models[0].endpoint_head.bias, model.endpoint_head.bias)


def test_probe_and_reference_contract() -> None:
    assert exp52.TEMPORAL_PROBES == (
        "hidden_whole_count",
        "hidden_fixed250_ordered",
        "hidden_relative10_ordered",
        "hidden_uend",
    )
    assert exp52.LOCAL_REFERENCE_PROBES == (
        "local_whole_count",
        "local_fixed250_ordered",
        "local_relative10_ordered",
    )


def test_multi_cpu_and_afterok_contract() -> None:
    prep = PREP.read_text(encoding="utf-8")
    runner = RUNNER.read_text(encoding="utf-8")
    finalizer = FINALIZER.read_text(encoding="utf-8")
    submit = SUBMIT.read_text(encoding="utf-8")
    assert "#SBATCH --array=0-4%5" in prep
    assert "#SBATCH --array=0-49%50" in runner
    for text in (prep, runner, finalizer):
        assert "#SBATCH --cpus-per-task=1" in text
        assert "module load conda/latest" in text
        assert "conda activate writingring-gpu" in text
        for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
            assert f"export {name}=1" in text
    assert 'afterok:${LOCAL_JOB}' in submit
    assert 'afterok:${DECODER_JOB}' in submit
    assert "50 runs" in submit


def test_notebook_is_analysis_only_and_validation_selects_tau() -> None:
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    assert notebook["nbformat"] == 4
    sources: list[str] = []
    for index, cell in enumerate(notebook["cells"]):
        source = "".join(cell.get("source", []))
        sources.append(source)
        if cell.get("cell_type") == "code":
            compile(source, f"exp52-notebook-cell-{index}", "exec")
    joined = "\n".join(sources)
    for token in (
        "runs.csv",
        "histories.csv",
        "local_reference.csv",
        "manifest.json",
        "val_ba_mean",
        "selected_rsnn_shift",
        "rsnn_minus_ff_test",
        "hidden_uend_test_ba",
        "hidden_relative10_ordered_test_ba",
        "timing_recovery",
        "fixed250_gap",
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
        "Frozen-local FF/RSNN",
        "raw L2 spike trajectory at 64 Hz",
        "shift_mem_R = 3, 4, 5, 6, 7",
        "hidden Uend",
        "endpoint CE",
        "50 independent temporal-decoder runs",
        "RSNN(tau) - FF(tau)",
        "hidden_relative10_ordered",
        "0-49%50",
        "analysis-only",
    ):
        assert token in text
