from __future__ import annotations

from dataclasses import replace
import inspect

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from core_benchmark_v1.model import BenchmarkNet
from core_benchmark_v1.runner import smoke_protocol
from scripts import experiment_16_prefix_supervised_selective_memory as exp


def _small_protocol():
    return replace(
        smoke_protocol(),
        width=6,
        input_channels=3,
        total_channels=3,
        steps=16,
        labels=("A", "B", "C"),
        batch_size=4,
        max_epochs=3,
        min_epochs=1,
        patience=1,
        relative_bins=4,
        shuffle_seeds=(101,),
    )


def test_manifest_constants_and_task_mapping() -> None:
    assert exp.SEEDS == (11, 23, 37)
    assert exp.PREFIX_PHASES == (0.50, 0.75)
    assert exp.PREFIX_WEIGHTS == (0.5, 0.5)
    assert exp.PREFIX_LAMBDAS == (0.0, 0.01, 0.03, 0.05, 0.10)
    assert exp.LOOKAHEAD_EPOCHS == (10, 20, 30, 50)
    assert len(exp.phase0a_specs()) == 3
    assert len(exp.phase0b_specs()) == 60
    assert len(exp.phase1_specs()) == 12
    assert len(exp.phase15_specs()) == 9
    assert exp.PHASE1_CASES == ("C0", "P0", "G0", "GP_J")
    assert set(exp.PHASE15_CASES) == {"GP_stopR", "GP_gate", "GP_noGate"}


def test_prefix_loss_is_exp14_1_definition() -> None:
    evidence = torch.tensor(
        [
            [[3.0, 0.0], [1.0, 0.0], [0.0, 2.0], [0.0, 2.0]],
            [[0.0, 3.0], [0.0, 1.0], [2.0, 0.0], [2.0, 0.0]],
        ],
        requires_grad=True,
    )
    lengths = torch.tensor([4, 4])
    y = torch.tensor([0, 1])
    expected = (
        0.5 * F.cross_entropy(evidence[:, :2].mean(1), y)
        + 0.5 * F.cross_entropy(evidence[:, :3].mean(1), y)
    )
    actual = exp.prefix_wcce(evidence, lengths, y)
    assert torch.allclose(actual, expected)
    actual.backward()
    assert evidence.grad is not None


def test_phase0_selectors_are_opposite_primary_metrics() -> None:
    best = {"ba": 0.60, "mean_logit_ce": 1.0}
    lower_ce_lower_ba = {"ba": 0.59, "mean_logit_ce": 0.9}
    higher_ba_higher_ce = {"ba": 0.61, "mean_logit_ce": 1.1}
    assert exp._better(lower_ce_lower_ba, best, exp.SELECTOR_CE)
    assert not exp._better(lower_ce_lower_ba, best, exp.SELECTOR_BA)
    assert exp._better(higher_ba_higher_ce, best, exp.SELECTOR_BA)
    assert not exp._better(higher_ba_higher_ce, best, exp.SELECTOR_CE)


def test_phase0a_source_preserves_ba_first_trajectory_stopping() -> None:
    source = inspect.getsource(exp.train_phase0a)
    assert "best_epoch[SELECTOR_BA]" in source
    assert "best_epoch[SELECTOR_CE]" not in source.split("if epoch >= p.min_epochs")[1].split("break")[0]
    assert "shared_wcce" in source


def test_suppressive_gate_never_amplifies_and_common_init_is_paired() -> None:
    p = _small_protocol()
    ungated = BenchmarkNet(exp._run(exp.ExpSpec("C0", 11)), p)
    gated = exp.SuppressiveContextWriteGateNet(exp._run(exp.ExpSpec("G0", 11)), p)

    for key, value in ungated.state_dict().items():
        if key in gated.state_dict():
            assert torch.equal(value, gated.state_dict()[key]), key

    generator = torch.Generator().manual_seed(19)
    x = torch.rand(4, p.steps, p.input_channels, generator=generator)
    lengths = torch.tensor([16, 14, 11, 9])
    with torch.no_grad():
        out = gated(x, lengths)
    valid = torch.arange(p.steps)[None, :] < lengths[:, None]
    values = out["gate_g"][valid]
    assert torch.all(values > 0)
    assert torch.all(values < 1)
    assert float(values.max()) <= 1.0


def test_prefix_routing_parameter_sets_are_symmetric() -> None:
    p = _small_protocol()
    model = exp.SuppressiveContextWriteGateNet(exp._run(exp.ExpSpec("GP_J", 11)), p)
    names = {id(parameter): name for name, parameter in model.named_parameters()}

    joint = {names[id(p)] for p in exp._prefix_target_params(model, "joint")}
    stop_r = {names[id(p)] for p in exp._prefix_target_params(model, "stopR")}
    gate_only = {names[id(p)] for p in exp._prefix_target_params(model, "gate_only")}
    no_gate = {names[id(p)] for p in exp._prefix_target_params(model, "no_gate")}

    gate_names = {"gate_input.weight", "gate_history.weight", "gate_bias"}
    assert gate_only == gate_names
    assert "head.weight" in joint
    assert "head.weight" not in stop_r
    assert no_gate.isdisjoint(gate_names)
    assert joint == gate_only | no_gate


def test_gate_only_prefix_receives_full_recurrent_total_derivative() -> None:
    p = _small_protocol()
    model = exp.SuppressiveContextWriteGateNet(exp._run(exp.ExpSpec("GP_gate", 11)), p)
    with torch.no_grad():
        model.gate_history.weight.fill_(0.1)
        model.gate_input.weight.fill_(0.1)
    generator = torch.Generator().manual_seed(5)
    x = torch.rand(4, p.steps, p.input_channels, generator=generator)
    y = torch.tensor([0, 1, 2, 0])
    lengths = torch.tensor([16, 14, 12, 10])
    evidence = model(x, lengths)["evidence"]
    wcce = F.cross_entropy(evidence.mean(1), y)
    prefix = exp.prefix_wcce(evidence, lengths, y)
    model.zero_grad(set_to_none=True)
    exp._backward_routed(model, wcce, prefix, 0.03, "gate_only")
    assert model.gate_input.weight.grad is not None
    assert model.gate_history.weight.grad is not None
    assert torch.isfinite(model.gate_history.weight.grad).all()


def test_functional_gate_ablation_preserves_or_destroys_expected_structure() -> None:
    p = _small_protocol()
    model = exp.SuppressiveContextWriteGateNet(exp._run(exp.ExpSpec("GP_J", 11)), p)
    with torch.no_grad():
        model.gate_input.weight.fill_(0.2)
        model.gate_history.weight.fill_(0.1)
    generator = torch.Generator().manual_seed(29)
    x = torch.rand(2, p.steps, p.input_channels, generator=generator)
    lengths = torch.tensor([16, 11])
    ids = np.asarray(["sample-a", "sample-b"])

    with torch.no_grad():
        learned = model(x, lengths)["gate_g"]
        one = exp._forced_gate(model, x, lengths, "gate_one", ids, -1)
        mean = exp._forced_gate(model, x, lengths, "gate_time_mean", ids, -1)
        shuffled = exp._forced_gate(model, x, lengths, "gate_time_shuffle", ids, 101)

    assert one is not None and torch.allclose(one, torch.ones_like(one))
    assert mean is not None
    assert shuffled is not None
    for i, length in enumerate(lengths.tolist()):
        assert torch.allclose(mean[i, :length].mean(), learned[i, :length].mean())
        assert torch.allclose(
            torch.sort(shuffled[i, :length]).values,
            torch.sort(learned[i, :length]).values,
        )


def test_validation_selection_functions_do_not_reference_test_metrics() -> None:
    for function in (exp.finalize_phase0a, exp.select_phase0b, exp.finalize_phase1):
        source = inspect.getsource(function)
        decision_section = source[source.find("decision ="):] if "decision =" in source else source
        assert "test_ba" not in decision_section
        assert '["test"]' not in decision_section
        assert "test_metrics_not_used" in source or function is exp.select_phase0b


def test_phase1_5_decision_is_positive_interaction_only() -> None:
    source = inspect.getsource(exp.finalize_phase1)
    assert "interaction > 0 and prefix_gain > 0" in source
    assert "validation native BA interaction" in source
