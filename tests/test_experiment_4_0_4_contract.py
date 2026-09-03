from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from scripts import experiment_4_0_fixed250_temporal_snn as exp40
from scripts import experiment_4_0_1_multispike_macro_lif as exp401
from scripts import experiment_4_0_4_hidden_state_probe as exp404


def test_exp404_run_matrix_is_four_rsnn_cap_variants_by_five_seeds() -> None:
    specs = exp404.run_specs()
    assert len(specs) == 20 == exp404.EXPECTED_RUNS
    assert {spec.seed for spec in specs} == {11, 23, 37, 53, 71}
    assert {
        (spec.variant, spec.hidden_cap, spec.output_cap) for spec in specs
    } == {
        ("binary", 1, 1),
        ("multi_h", 31, 1),
        ("multi_o", 1, 31),
        ("multi_ho", 31, 31),
    }
    assert len({spec.key for spec in specs}) == len(specs)
    assert [spec.variant for spec in specs] == (
        ["binary"] * 5
        + ["multi_h"] * 5
        + ["multi_o"] * 5
        + ["multi_ho"] * 5
    )
    for offset in range(0, 20, 5):
        assert [spec.seed for spec in specs[offset : offset + 5]] == [11, 23, 37, 53, 71]


def test_exp404_source_identity_is_fixed_to_recurrent_h128_tau250() -> None:
    for probe_spec in exp404.run_specs():
        source = exp404.source_spec(probe_spec)
        assert source.architecture == "rsnn"
        assert source.hidden_width == 128
        assert source.tau_mem_ms == 250.0
        assert source.variant == probe_spec.variant
        assert source.hidden_cap == probe_spec.hidden_cap
        assert source.output_cap == probe_spec.output_cap
        assert source.seed == probe_spec.seed


def test_exp404_endpoint_feature_is_hidden_post_reset_state_only() -> None:
    torch.manual_seed(7)
    model = exp401.MacroTemporalDecoder(
        architecture="rsnn",
        hidden_width=128,
        tau_mem_ms=250.0,
        n_classes=12,
        hidden_cap=31,
        output_cap=1,
    )
    X = torch.randn(2, 4, exp40.EVENT_CHANNELS)
    y = torch.tensor([0, 1])
    valid_bins = torch.tensor([2, 3])
    loader = DataLoader(TensorDataset(X, y, valid_bins), batch_size=2, shuffle=False)

    states_before, labels_before = exp404.extract_hidden_endpoint_states(
        model, loader, torch.device("cpu")
    )

    with torch.no_grad():
        model.hidden_output.weight.fill_(1234.0)
    states_after, labels_after = exp404.extract_hidden_endpoint_states(
        model, loader, torch.device("cpu")
    )

    assert states_before.shape == (2, 128)
    assert np.array_equal(labels_before, np.array([0, 1]))
    assert np.array_equal(labels_before, labels_after)
    assert np.array_equal(states_before, states_after)


def test_exp404_endpoint_feature_selects_each_samples_valid_endpoint() -> None:
    torch.manual_seed(13)
    model = exp401.MacroTemporalDecoder(
        architecture="rsnn",
        hidden_width=128,
        tau_mem_ms=250.0,
        n_classes=12,
        hidden_cap=1,
        output_cap=31,
    )
    X = torch.randn(2, 5, exp40.EVENT_CHANNELS)
    y = torch.tensor([2, 3])
    valid_bins = torch.tensor([2, 4])
    loader = DataLoader(TensorDataset(X, y, valid_bins), batch_size=2, shuffle=False)

    states, _ = exp404.extract_hidden_endpoint_states(
        model, loader, torch.device("cpu")
    )

    hidden_mem = torch.zeros(2, 128)
    prev_hidden_spikes = torch.zeros_like(hidden_mem)
    memories: list[torch.Tensor] = []
    with torch.no_grad():
        for b in range(X.shape[1]):
            current = model.input_hidden(X[:, b])
            if model.recurrent is not None:
                current = current + model.recurrent(prev_hidden_spikes)
            hidden_spikes, hidden_mem, _ = model.hidden_lif(current, hidden_mem)
            memories.append(hidden_mem)
            prev_hidden_spikes = hidden_spikes
    expected = exp40.valid_final_membrane(torch.stack(memories, dim=1), valid_bins)
    assert np.allclose(states, expected.numpy())


def test_exp404_probe_contract_has_no_phase_or_output_neuron_access() -> None:
    torch.manual_seed(17)
    model = exp401.MacroTemporalDecoder(
        architecture="rsnn",
        hidden_width=128,
        tau_mem_ms=250.0,
        n_classes=12,
        hidden_cap=1,
        output_cap=1,
    )
    X = torch.randn(8, 3, exp40.EVENT_CHANNELS)
    y = torch.tensor([0, 1, 2, 3, 0, 1, 2, 3])
    valid_bins = torch.tensor([1, 2, 3, 2, 3, 1, 2, 3])
    loader = DataLoader(TensorDataset(X, y, valid_bins), batch_size=4, shuffle=False)
    probe = exp404.fit_endpoint_state_probe(
        model,
        [loader, loader, loader],
        torch.device("cpu"),
    )
    assert probe["feature_dim"] == 128
    assert probe["temporal_phase_access"] is False
    assert probe["output_neuron_access"] is False
    assert probe["source_snn_frozen"] is True


def test_exp404_multi_cpu_scripts_use_one_core_per_probe_and_afterok_finalizer() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    array_script = (
        repo_root / "scripts/bash_script/SNN_Bash/run_exp_4_0_4_cpu_array.bash"
    ).read_text(encoding="utf-8")
    finalizer = (
        repo_root / "scripts/bash_script/SNN_Bash/finalize_exp_4_0_4_cpu.bash"
    ).read_text(encoding="utf-8")
    submit = (
        repo_root / "scripts/bash_script/SNN_Bash/submit_exp_4_0_4_cpu.bash"
    ).read_text(encoding="utf-8")

    assert "#SBATCH --array=0-19%20" in array_script
    assert "#SBATCH --cpus-per-task=1" in array_script
    assert "run-one" in array_script
    for variable in (
        "OMP_NUM_THREADS=1",
        "MKL_NUM_THREADS=1",
        "OPENBLAS_NUM_THREADS=1",
        "NUMEXPR_NUM_THREADS=1",
    ):
        assert variable in array_script
        assert variable in finalizer
    assert "--dependency=afterok:${ARRAY_JOB_ID}" in submit
    assert "hidden_state_probe finalize" in finalizer
