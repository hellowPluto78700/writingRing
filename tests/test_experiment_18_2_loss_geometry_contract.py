from __future__ import annotations

from dataclasses import replace

import torch

from core_benchmark_v1.model import BenchmarkNet, mean_logits, valid_sum
from core_benchmark_v1.protocol import Run
from core_benchmark_v1.runner import smoke_protocol
from scripts import experiment_18_2_loss_geometry as exp


def _p():
    return replace(
        smoke_protocol(),
        width=12,
        input_channels=3,
        total_channels=3,
        steps=16,
        labels=("A", "B", "C"),
        batch_size=4,
        max_epochs=2,
        min_epochs=1,
        patience=1,
    )


def test_axes_and_references_are_locked():
    assert exp.SEEDS == (11, 23, 37)
    assert exp.FORMAL_CASES == ("I_NWCCE", "I_MWCCE", "U_NWCCE", "U_MWCCE")
    assert exp.REFERENCE_CASES == ("I_WCCE_REF", "U_WCCE_REF")
    assert exp.MARGIN_TARGET == 1.0
    assert len(exp.specs()) == 12


def test_formal_initialization_is_paired_with_core():
    p = _p()
    core = BenchmarkNet(Run("O0", 11, "01_objective", shifts=exp.SHIFTS, objective="wcce"), p)
    for case in exp.FORMAL_CASES:
        model = exp._model(exp.ExpSpec(case, 11), p)
        assert torch.equal(model.layers[0].weight, core.layers[0].weight)
        assert torch.equal(model.layers[1].weight, core.layers[1].weight)
        assert torch.equal(model.head.weight, core.head.weight)


def test_normalized_counts_are_scale_invariant_and_zero_safe():
    counts = torch.tensor([[1.0, 2.0, 3.0], [0.0, 0.0, 0.0]], requires_grad=True)
    a = exp.normalized_counts(counts)
    b = exp.normalized_counts(7.0 * counts)
    assert torch.allclose(a, b)
    assert torch.equal(a[1], torch.zeros(3))
    a[0, 0].backward()
    assert counts.grad is not None
    assert torch.isfinite(counts.grad).all()
    assert counts.grad[0].abs().sum() > 0


def test_normalized_wcce_removes_uniform_count_scale():
    p = _p()
    model = exp._model(exp.ExpSpec("I_NWCCE", 11), p)
    counts = torch.rand(4, p.width) + 0.1
    logits_a = model.head(exp.normalized_counts(counts))
    logits_b = model.head(exp.normalized_counts(5.0 * counts))
    assert torch.allclose(logits_a, logits_b, atol=1e-7, rtol=1e-6)


def test_margin_loss_is_zero_once_target_margin_is_met():
    logits = torch.tensor([[2.0, 0.5, -1.0], [-1.0, 2.5, 0.0]])
    y = torch.tensor([0, 1])
    assert exp.margin_wcce_loss(logits, y).item() == 0.0


def test_margin_loss_penalizes_insufficient_margin():
    logits = torch.tensor([[0.2, 0.0, -1.0]])
    y = torch.tensor([0])
    loss = exp.margin_wcce_loss(logits, y)
    assert torch.allclose(loss, torch.tensor(0.8))


def test_native_checkpoint_selection_metric_remains_raw_wcce_logits():
    p = _p()
    model = exp._model(exp.ExpSpec("I_NWCCE", 11), p)
    x = torch.randn(3, p.steps, p.input_channels)
    lengths = torch.tensor([16, 14, 11])
    out = model(x, lengths)
    native = mean_logits(out["evidence"], lengths)
    counts = valid_sum(out["spike"][-1], lengths)
    normalized = model.head(exp.normalized_counts(counts))
    assert native.shape == normalized.shape
    # The formal selection path must not silently replace native logits
    # with the normalized training-only geometry.
    assert not torch.equal(native, normalized)


def test_i_and_u_cases_use_expected_carriers():
    p = _p()
    i_model = exp._model(exp.ExpSpec("I_NWCCE", 11), p)
    u_model = exp._model(exp.ExpSpec("U_NWCCE", 11), p)
    assert i_model.carriers == ("I", "I")
    assert u_model.carriers == ("U", "U")
