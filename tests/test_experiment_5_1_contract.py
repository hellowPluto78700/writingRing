from __future__ import annotations

import json
import math
from pathlib import Path

import torch

from scripts import experiment_5_1_boundary_free_temporal_decoder as exp51


REPO_ROOT = Path(__file__).resolve().parents[1]
PREP = REPO_ROOT / "scripts" / "bash_script" / "SNN_Bash" / "prepare_exp_5_1_local_cpu_array.bash"
DECODER = REPO_ROOT / "scripts" / "bash_script" / "SNN_Bash" / "run_exp_5_1_decoder_cpu_array.bash"
PROBE = REPO_ROOT / "scripts" / "bash_script" / "SNN_Bash" / "run_exp_5_1_interface_probe_cpu_array.bash"
FINALIZER = REPO_ROOT / "scripts" / "bash_script" / "SNN_Bash" / "finalize_exp_5_1_cpu.bash"
SUBMIT = REPO_ROOT / "scripts" / "bash_script" / "SNN_Bash" / "submit_exp_5_1_cpu.bash"
README = REPO_ROOT / "scripts" / "experiment_5_1" / "README.md"
NOTEBOOK = REPO_ROOT / "notebooks" / "experiment_5_1_boundary_free_temporal_decoder.ipynb"


def test_run_matrix_and_source_contract() -> None:
    specs = exp51.run_specs()
    assert exp51.PROTOCOL_VERSION == "frozen_local_leaky_interface_rsnn_v1"
    assert exp51.LOCAL_SOURCE_CONDITION == exp51.exp501.EXP3_EXACT
    assert exp51.SEEDS == (11, 23, 101)
    assert exp51.INTERFACES == ("raw_l2", "leaky242")
    assert exp51.ARCHITECTURES == ("ff", "rsnn")
    assert exp51.TEMPORAL_TAU_MEM_MS == (125.0, 250.0, 500.0, 1000.0)
    assert len(specs) == 48
    assert len({spec.key for spec in specs}) == 48
    assert len(exp51.interface_probe_specs()) == 6


def test_leaky242_is_identity_preserving_boundary_free_pooling() -> None:
    assert exp51.AGG_SHIFT == 4
    assert exp51.AGG_ALPHA == 15.0 / 16.0
    assert math.isclose(1.0 / (1.0 - exp51.AGG_ALPHA), 16.0)
    assert math.isclose(exp51.aggregator_tau_ms(64.0), 242.103, rel_tol=1e-3)

    spikes = torch.zeros(1, 4, exp51.LOCAL_WIDTH)
    spikes[0, :, 0] = 1.0
    pooled = exp51.interface_sequence(spikes, "leaky242")
    expected = torch.tensor(
        [
            1.0,
            1.0 + exp51.AGG_ALPHA,
            1.0 + exp51.AGG_ALPHA + exp51.AGG_ALPHA**2,
            1.0 + exp51.AGG_ALPHA + exp51.AGG_ALPHA**2 + exp51.AGG_ALPHA**3,
        ]
    )
    assert torch.allclose(pooled[0, :, 0], expected)
    assert torch.count_nonzero(pooled[0, :, 1:]) == 0
    assert torch.equal(exp51.interface_sequence(spikes, "raw_l2"), spikes)


def test_temporal_decoder_has_analog_head_and_no_output_lif() -> None:
    ff = exp51.TemporalAnalogDecoder("ff", 250.0, 12, 64.0)
    rsnn = exp51.TemporalAnalogDecoder("rsnn", 250.0, 12, 64.0)
    assert ff.input_hidden.in_features == 128
    assert ff.input_hidden.out_features == 128
    assert ff.recurrent is None
    assert rsnn.recurrent is not None
    assert rsnn.recurrent.in_features == 128
    assert rsnn.recurrent.out_features == 128
    assert rsnn.head.in_features == 128
    assert rsnn.head.out_features == 12
    assert rsnn.head.bias is not None
    assert not hasattr(rsnn, "output_lif")
    assert not hasattr(rsnn, "output_spike")


def test_sequence_readout_is_valid_mean_not_timestep_ce() -> None:
    logits_t = torch.tensor([[[1.0, 0.0], [3.0, 0.0], [100.0, 0.0]]])
    lengths = torch.tensor([2])
    logits = exp51.TemporalAnalogDecoder.sequence_logits(logits_t, lengths)
    assert torch.allclose(logits, torch.tensor([[2.0, 0.0]]))


def test_decoder_initialization_is_paired_across_interface_tau_and_architecture() -> None:
    device = torch.device("cpu")
    specs = [
        exp51.RunSpec("raw_l2", "ff", 125.0, 11),
        exp51.RunSpec("leaky242", "ff", 1000.0, 11),
        exp51.RunSpec("raw_l2", "rsnn", 250.0, 11),
    ]
    models = [exp51._initialize_temporal_model(spec, 12, 64.0, device) for spec in specs]
    for model in models[1:]:
        assert torch.equal(models[0].input_hidden.weight, model.input_hidden.weight)
        assert torch.equal(models[0].head.weight, model.head.weight)
        assert torch.equal(models[0].head.bias, model.head.bias)


def test_probe_locations_cover_interface_and_temporal_state() -> None:
    assert exp51.INTERFACE_PROBES == (
        "interface_valid_sum",
        "interface_fixed250_ordered",
        "interface_relative10_ordered",
        "interface_endpoint",
    )
    assert exp51.TEMPORAL_PROBES == (
        "hidden_whole_count",
        "hidden_fixed250_ordered",
        "hidden_relative10_ordered",
        "hidden_uend",
    )


def test_multi_cpu_and_afterok_contract() -> None:
    prep = PREP.read_text(encoding="utf-8")
    decoder = DECODER.read_text(encoding="utf-8")
    probe = PROBE.read_text(encoding="utf-8")
    finalizer = FINALIZER.read_text(encoding="utf-8")
    submit = SUBMIT.read_text(encoding="utf-8")
    assert "#SBATCH --array=0-2%3" in prep
    assert "#SBATCH --array=0-47%48" in decoder
    assert "#SBATCH --array=0-5%6" in probe
    for text in (prep, decoder, probe, finalizer):
        assert "#SBATCH --cpus-per-task=1" in text
        assert "module load conda/latest" in text
        assert "conda activate writingring-gpu" in text
        for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
            assert f"export {name}=1" in text
    assert 'afterok:${LOCAL_JOB}' in submit
    assert 'afterok:${DECODER_JOB}:${PROBE_JOB}' in submit
    assert "48 runs" in submit


def test_notebook_is_analysis_only_and_uses_validation_selection() -> None:
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    assert notebook["nbformat"] == 4
    sources: list[str] = []
    for index, cell in enumerate(notebook["cells"]):
        source = "".join(cell.get("source", []))
        sources.append(source)
        if cell.get("cell_type") == "code":
            compile(source, f"exp51-notebook-cell-{index}", "exec")
    joined = "\n".join(sources)
    for token in (
        "runs.csv",
        "interface_probes.csv",
        "histories.csv",
        "local_reference.csv",
        "val_ba_mean",
        "leaky_minus_raw",
        "hidden_uend_test_ba",
        "hidden_relative10_ordered_test_ba",
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
        "boundary-free leaky interface",
        "A_t = alpha * A_(t-1) + S_t^L2",
        "no cross-neuron mixing",
        "no threshold",
        "no reset",
        "no output LIF",
        "CE(E, y)",
        "48 independent decoder runs",
        "validation native BA",
        "hidden_uend",
        "0-47%48",
        "analysis-only",
    ):
        assert token in text
