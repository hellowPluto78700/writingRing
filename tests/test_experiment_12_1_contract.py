from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch

from scripts import experiment_12_1_primitive_capacity_fragmentation as exp121


def test_exp12_1_contract_dimensions_and_run_count() -> None:
    assert exp121.ARCHITECTURE == "234x234"
    assert exp121.SHIFTS == ((2, 3, 4), (2, 3, 4))
    assert exp121.SEEDS == (11, 23, 37)
    assert exp121.PRIMITIVE_WIDTHS == (16, 32, 64, 128)
    assert exp121.TEMPORAL_MODES == ("t0", "ema")
    assert len(exp121.run_specs()) == 24
    assert len({spec.key for spec in exp121.run_specs()}) == 24


def test_exp12_1_model_uses_requested_primitive_width() -> None:
    for width in exp121.PRIMITIVE_WIDTHS:
        model = exp121.PrimitiveCapacityNet(
            n_classes=12,
            fs=64.0,
            temporal_mode="t0",
            primitive_width=width,
        )
        assert model.primitive_projection.in_features == exp121.HIDDEN_WIDTH
        assert model.primitive_projection.out_features == width
        assert model.classifier.in_features == width
        assert model.classifier.out_features == 12


def test_exp12_1_ema_matches_exp12_temporal_definition() -> None:
    torch.manual_seed(5)
    z = torch.rand(2, 9, exp121.HIDDEN_WIDTH)
    model = object.__new__(exp121.PrimitiveCapacityNet)
    model.temporal_mode = "ema"
    actual = exp121.PrimitiveCapacityNet._temporal_integrate(model, z)

    state = torch.zeros_like(z[:, 0])
    expected = []
    for t in range(z.shape[1]):
        state = exp121.BETA * state + (1.0 - exp121.BETA) * z[:, t]
        expected.append(state)
    torch.testing.assert_close(actual, torch.stack(expected, dim=1))


def test_stroke_fragmentation_uses_true_stroke_boundaries() -> None:
    winner = np.array(
        [[0, 0, 1, 1, 2, 2, 2, 2]],
        dtype=np.int64,
    )
    q = np.eye(3, dtype=np.float64)[winner]
    lengths = np.array([8], dtype=np.int64)
    frame = pd.DataFrame(
        [
            {
                "user": "user_0",
                "action": "0",
                "source_segment_index": 7,
            }
        ]
    )
    stroke_index = {
        ("user_0", "0", 7): (
            (0, 0, 3),
            (1, 4, 7),
        )
    }

    metrics, transitions = exp121._stroke_fragmentation_metrics(
        winner,
        q,
        lengths,
        frame,
        stroke_index,
    )

    # Stroke 0 contains 0->1 and is fragmented; stroke 1 is pure slot 2.
    assert metrics["stroke_count"] == 2.0
    assert metrics["mean_slots_per_stroke"] == 1.5
    assert metrics["mean_dominant_slot_purity"] == 0.75
    assert metrics["mean_switches_per_stroke"] == 0.5
    assert metrics["fragmented_stroke_fraction"] == 0.5
    # The 1->2 change lies exactly at the true stroke boundary and therefore
    # must not be counted as within-stroke fragmentation.
    assert transitions[0, 1] == 1
    assert transitions[1, 2] == 0


def test_output_to_source_segment_uses_padding_manifest_mapping(tmp_path) -> None:
    manifest = pd.DataFrame(
        [
            {"segment_index": 3, "output_segment_index": 0, "exported": True},
            {"segment_index": 4, "output_segment_index": np.nan, "exported": False},
            {"segment_index": 5, "output_segment_index": 1, "exported": True},
        ]
    )
    path = tmp_path / "padding_manifest.csv"
    manifest.to_csv(path, index=False)
    package = SimpleNamespace(padding_manifest_path=path, segment_count=2)

    assert exp121._output_to_source_segment(package) == {0: 3, 1: 5}


def test_raw_and_normalized_softmax_have_same_winner() -> None:
    torch.manual_seed(17)
    h = torch.rand(3, 7, exp121.HIDDEN_WIDTH)
    projection = torch.nn.Linear(exp121.HIDDEN_WIDTH, 64, bias=False)
    components = exp121.PrimitiveCapacityNet.primitive_components(h, projection)
    torch.testing.assert_close(
        components["winner_raw"],
        components["winner_norm"],
    )
