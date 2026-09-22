from __future__ import annotations

import numpy as np
import torch

from scripts import experiment_12_0_local_primitive_bottleneck as exp12


def test_exp12_contract_dimensions_and_counts() -> None:
    assert exp12.ARCHITECTURE == "234x234"
    assert exp12.SHIFTS == ((2, 3, 4), (2, 3, 4))
    assert exp12.SEEDS == (11, 23, 37)
    assert exp12.PRIMITIVE_WIDTH == 16
    assert exp12.TEMPORAL_MODES == ("t0", "raw", "ema")
    assert len(exp12.dense_specs()) == 36
    assert len(exp12.posthoc_specs()) == 27


def test_raw_and_ema_have_identical_kernel_shape_up_to_scale() -> None:
    torch.manual_seed(7)
    z = torch.rand(2, 12, exp12.HIDDEN_WIDTH)
    raw = object.__new__(exp12.PrimitiveBottleneckNet)
    raw.temporal_mode = "raw"
    ema = object.__new__(exp12.PrimitiveBottleneckNet)
    ema.temporal_mode = "ema"

    raw_h = exp12.PrimitiveBottleneckNet._temporal_integrate(raw, z)
    ema_h = exp12.PrimitiveBottleneckNet._temporal_integrate(ema, z)
    scale = 1.0 / (1.0 - exp12.BETA)
    torch.testing.assert_close(raw_h, scale * ema_h)


def test_normalized_primitive_what_is_scale_and_offset_invariant() -> None:
    torch.manual_seed(11)
    h = torch.rand(2, 5, exp12.HIDDEN_WIDTH)
    projection = torch.nn.Linear(
        exp12.HIDDEN_WIDTH, exp12.PRIMITIVE_WIDTH, bias=False
    )
    base = exp12.PrimitiveBottleneckNet.primitive_components(h, projection)

    with torch.no_grad():
        projection.weight.mul_(7.0)
    scaled = exp12.PrimitiveBottleneckNet.primitive_components(h, projection)

    torch.testing.assert_close(base["q"], scaled["q"], atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(
        base["confidence"], scaled["confidence"], atol=2e-6, rtol=2e-5
    )


def test_same_primitive_peak_uses_candidate_primitive_on_neighbors() -> None:
    normalized = np.array(
        [[
            [0.0, 0.1, -0.1],
            [0.2, 0.1, -0.2],
            [0.9, 1.0, -0.5],
            [0.8, 1.2, -0.4],
            [0.7, 0.3, -0.2],
        ]],
        dtype=np.float64,
    )
    winner = normalized.argmax(axis=-1)
    lengths = np.array([5], dtype=np.int64)

    mask = exp12._peak_mask(normalized, winner, lengths, threshold=0.05)

    assert mask.shape == (1, 5)
    assert mask[0, 3]
    assert not mask[0, 2]


def test_zero_primitive_spread_produces_uniform_what_and_zero_confidence() -> None:
    h = torch.zeros(2, 4, exp12.HIDDEN_WIDTH)
    projection = torch.nn.Linear(
        exp12.HIDDEN_WIDTH, exp12.PRIMITIVE_WIDTH, bias=False
    )
    components = exp12.PrimitiveBottleneckNet.primitive_components(h, projection)

    expected = torch.full_like(components["q"], 1.0 / exp12.PRIMITIVE_WIDTH)
    torch.testing.assert_close(components["q"], expected)
    torch.testing.assert_close(
        components["confidence"], torch.zeros_like(components["confidence"])
    )
    torch.testing.assert_close(
        components["activity"], torch.zeros_like(components["activity"])
    )


def test_zero_preserving_rms_has_finite_zero_gradient() -> None:
    values = torch.zeros(2, 3, exp12.HIDDEN_WIDTH, requires_grad=True)
    rms = exp12._zero_preserving_rms(values, dim=-1)
    torch.testing.assert_close(rms, torch.zeros_like(rms))
    rms.sum().backward()
    assert values.grad is not None
    assert torch.isfinite(values.grad).all()
    torch.testing.assert_close(values.grad, torch.zeros_like(values.grad))


def test_zero_spread_primitive_components_have_finite_backward() -> None:
    h = torch.zeros(2, 4, exp12.HIDDEN_WIDTH, requires_grad=True)
    projection = torch.nn.Linear(
        exp12.HIDDEN_WIDTH, exp12.PRIMITIVE_WIDTH, bias=False
    )
    components = exp12.PrimitiveBottleneckNet.primitive_components(h, projection)
    objective = components["q"].sum() + components["activity"].sum()
    objective.backward()
    assert h.grad is not None
    assert torch.isfinite(h.grad).all()
    assert projection.weight.grad is not None
    assert torch.isfinite(projection.weight.grad).all()


def test_checkpoint_finite_state_detection(tmp_path) -> None:
    good = {"weight": torch.tensor([1.0, 2.0])}
    bad = {"weight": torch.tensor([1.0, float("nan")])}

    assert exp12._model_state_is_finite(good)
    assert not exp12._model_state_is_finite(bad)

    good_path = tmp_path / "good.pt"
    bad_path = tmp_path / "bad.pt"
    torch.save({"model_state_dict": good}, good_path)
    torch.save({"model_state_dict": bad}, bad_path)

    assert exp12._checkpoint_model_state_is_finite(good_path)
    assert not exp12._checkpoint_model_state_is_finite(bad_path)
