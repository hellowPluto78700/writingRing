from __future__ import annotations

import torch

from scripts import experiment_4_0_1_multispike_macro_lif as exp401


def test_run_matrix_is_complete_and_factorial() -> None:
    specs = exp401.run_specs()
    assert len(specs) == 40 == exp401.EXPECTED_RUNS
    assert {spec.architecture for spec in specs} == {"ff", "rsnn"}
    assert {spec.seed for spec in specs} == {11, 23, 37, 53, 71}
    assert {
        (spec.architecture, spec.hidden_width, spec.tau_mem_ms)
        for spec in specs
    } == {
        ("ff", 128, 2000.0),
        ("rsnn", 128, 250.0),
    }
    assert {
        (spec.variant, spec.hidden_cap, spec.output_cap)
        for spec in specs
    } == {
        ("binary", 1, 1),
        ("multi_h", 31, 1),
        ("multi_o", 1, 31),
        ("multi_ho", 31, 31),
    }
    assert len({spec.key for spec in specs}) == len(specs)


def test_random_streams_are_paired_across_cap_variants() -> None:
    group = [
        spec
        for spec in exp401.run_specs()
        if spec.architecture == "rsnn" and spec.seed == 11
    ]
    assert len(group) == 4
    assert len({exp401.paired_seed(spec, "model_init") for spec in group}) == 1
    assert len({exp401.paired_seed(spec, "train_loader") for spec in group}) == 1
    other_seed = next(
        spec
        for spec in exp401.run_specs()
        if spec.architecture == "rsnn" and spec.seed == 23
    )
    assert exp401.paired_seed(group[0], "model_init") != exp401.paired_seed(
        other_seed, "model_init"
    )


def test_binary_and_multispike_share_update_but_change_event_cap() -> None:
    current = torch.tensor([[2.1]], dtype=torch.float32)
    membrane = torch.zeros_like(current)
    binary = exp401.MacroMultiSpikeLIF(beta=0.0, threshold=0.5, max_spikes_per_dt=1)
    multi = exp401.MacroMultiSpikeLIF(beta=0.0, threshold=0.5, max_spikes_per_dt=31)

    binary_spikes, binary_mem, binary_pre = binary(current, membrane)
    multi_spikes, multi_mem, multi_pre = multi(current, membrane)

    assert torch.equal(binary_pre, multi_pre)
    assert torch.equal(binary_spikes, torch.tensor([[1.0]]))
    assert torch.allclose(binary_mem, torch.tensor([[1.6]]), atol=1e-6)
    assert torch.equal(multi_spikes, torch.tensor([[4.0]]))
    assert torch.allclose(multi_mem, torch.tensor([[0.1]]), atol=1e-6)


def test_multispike_cap_clamps_and_reset_uses_emitted_event_count() -> None:
    current = torch.tensor([[20.0]], dtype=torch.float32)
    membrane = torch.zeros_like(current)
    neuron = exp401.MacroMultiSpikeLIF(beta=0.0, threshold=0.5, max_spikes_per_dt=31)
    spikes, residual, pre_reset = neuron(current, membrane)
    assert torch.equal(spikes, torch.tensor([[31.0]]))
    assert torch.equal(pre_reset, current)
    assert torch.allclose(residual, torch.tensor([[4.5]]), atol=1e-6)


def test_multithreshold_surrogate_has_nonzero_gradient_near_crossing() -> None:
    membrane = torch.tensor([0.51], dtype=torch.float32, requires_grad=True)
    spikes = exp401.multi_threshold_spike(
        membrane,
        threshold=0.5,
        max_spikes_per_dt=31,
    )
    spikes.sum().backward()
    assert membrane.grad is not None
    assert torch.isfinite(membrane.grad).all()
    assert float(membrane.grad.item()) > 0.0


def test_output_normalization_preserves_per_step_evidence_range() -> None:
    binary = torch.tensor([[[0.0, 1.0]]])
    multi = torch.tensor([[[0.0, 31.0]]])
    assert torch.equal(exp401.normalize_output_spikes(binary, 1), binary)
    normalized = exp401.normalize_output_spikes(multi, 31)
    assert torch.equal(normalized, torch.tensor([[[0.0, 1.0]]]))


def test_valid_evidence_uses_normalized_output_events_and_endpoint_mask() -> None:
    spikes = torch.tensor(
        [
            [
                [31.0, 0.0],
                [0.0, 31.0],
                [31.0, 31.0],
            ]
        ]
    )
    valid_bins = torch.tensor([2])
    valid = exp401.valid_evidence(spikes, valid_bins, output_cap=31)
    full = exp401.full_evidence(spikes, output_cap=31)
    assert torch.equal(valid, torch.tensor([[1.0, 1.0]]))
    assert torch.equal(full, torch.tensor([[2.0, 2.0]]))


def test_rsnn_recurrence_consumes_hidden_event_count_and_ff_has_no_recurrence() -> None:
    ff = exp401.MacroTemporalDecoder(
        architecture="ff",
        hidden_width=128,
        tau_mem_ms=2000.0,
        n_classes=12,
        hidden_cap=31,
        output_cap=1,
    )
    rsnn = exp401.MacroTemporalDecoder(
        architecture="rsnn",
        hidden_width=128,
        tau_mem_ms=250.0,
        n_classes=12,
        hidden_cap=31,
        output_cap=1,
    )
    assert ff.recurrent is None
    assert rsnn.recurrent is not None
    assert tuple(rsnn.recurrent.weight.shape) == (128, 128)
    assert ff.hidden_lif.max_spikes_per_dt == 31
    assert rsnn.hidden_lif.max_spikes_per_dt == 31
    assert rsnn.output_lif.max_spikes_per_dt == 1


def test_parameter_counts_match_exp4_0_anchor_topologies() -> None:
    ff = exp401.exp40.parameter_counts(128, False, 12)
    rsnn = exp401.exp40.parameter_counts(128, True, 12)
    assert ff["total"] == 5376
    assert rsnn["total"] == 21760
