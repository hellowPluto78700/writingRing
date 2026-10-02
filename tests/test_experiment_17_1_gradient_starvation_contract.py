from __future__ import annotations

from dataclasses import replace
import inspect

import numpy as np
import torch

from core_benchmark_v1.runner import smoke_protocol
from scripts import experiment_16_3_run_reward as exp16_3
from scripts import experiment_17_1_gradient_starvation as exp


def _p():
    return replace(
        smoke_protocol(),
        width=12,
        input_channels=3,
        total_channels=3,
        steps=16,
        labels=("A", "B", "C"),
        batch_size=4,
        max_epochs=25,
        min_epochs=1,
        patience=2,
    )


def test_axes_are_locked():
    assert exp.FORMAL_SEEDS == (11, 23, 37)
    assert exp.CALIBRATION_SEED == 101
    assert exp.BRANCH_EPOCH == 20
    assert exp.GROUP_FRACTION == 0.25
    assert exp.CASES == ("C0", "C1", "C2", "C3")
    assert len(exp.formal_specs()) == 12


def test_model_initialization_matches_exp16_3_lin():
    p = _p()
    for seed in exp.FORMAL_SEEDS:
        a = exp._make_model(seed, p).state_dict()
        b = exp16_3._make_model(exp16_3.ExpSpec("LIN", seed, "linear", None), p).state_dict()
        assert a.keys() == b.keys()
        for name in a:
            assert torch.equal(a[name], b[name]), (seed, name)


def test_group_partition_is_dimension_matched_and_deterministic():
    occupancy = np.linspace(0.0, 1.0, 12)
    a = exp._group_indices(occupancy, 11)
    b = exp._group_indices(occupancy, 11)
    assert exp._group_size(12) == 3
    assert np.array_equal(a["low"], np.array([0, 1, 2]))
    assert np.array_equal(a["high"], np.array([9, 10, 11]))
    assert set(a["low"]).isdisjoint(set(a["high"]))
    for name in ("low", "high", "random"):
        assert len(a[name]) == 3
        assert len(np.unique(a[name])) == 3
        assert np.array_equal(a[name], b[name])


def test_case_mapping_is_fixed():
    groups = {
        "low": np.array([0, 1]),
        "high": np.array([6, 7]),
        "random": np.array([2, 5]),
    }
    assert exp._group_for_case(groups, "C0") is None
    assert np.array_equal(exp._group_for_case(groups, "C1"), groups["low"])
    assert np.array_equal(exp._group_for_case(groups, "C2"), groups["high"])
    assert np.array_equal(exp._group_for_case(groups, "C3"), groups["random"])


def test_aux_head_is_bias_free_and_deterministic():
    p = _p()
    a = exp._aux_head(11, p, "C1")
    b = exp._aux_head(11, p, "C1")
    assert a.bias is None
    assert torch.equal(a.weight, b.weight)


def test_formal_training_does_not_use_test_for_optimization():
    source = inspect.getsource(exp._run_formal)
    prefix = source.split('"test": _split_eval', 1)[0]
    assert '"test"' not in prefix
    assert "validation BA, then validation CE" in source


def test_calibration_never_uses_performance_metric():
    source = inspect.getsource(exp._calibrate)
    assert "val_ba" not in source
    assert "test_ba" not in source
    assert "selected_lambda" in source
    assert "GRADIENT_RATIO_TARGET" in source
