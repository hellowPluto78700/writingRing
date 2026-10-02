from __future__ import annotations

import inspect

from scripts import analyze_experiment_17_1_group_probes as probe
from scripts import experiment_17_1_gradient_starvation as exp17_1


def test_probe_axes_are_locked():
    assert probe.GROUPS == ("low", "high")
    specs = probe.probe_specs()
    assert len(specs) == 12
    assert {(spec.case, spec.seed) for spec in specs} == {
        (case, seed)
        for seed in exp17_1.FORMAL_SEEDS
        for case in exp17_1.CASES
    }


def test_probe_is_artifact_only():
    source = inspect.getsource(probe.run_probe_task)
    assert ".backward(" not in source
    assert "optimizer" not in source
    assert "_load_selected_model_and_groups" in source
    assert "_whole_count_features" in source


def test_probe_selection_is_validation_only_and_bias_free():
    source = inspect.getsource(probe._fit_group_probe)
    fit_call = source.split("fit_probe(", 1)[1].split(")", 1)[0]
    assert 'features["train"]' in fit_call
    assert 'arrays["train_y"]' in fit_call
    assert 'features["val"]' in fit_call
    assert 'arrays["val_y"]' in fit_call
    assert '"no_bias"' in fit_call
    assert "test" not in fit_call


def test_whole_count_geometry_is_temporal_sum():
    source = inspect.getsource(probe._whole_count_features)
    assert 'out["spike"][1].sum(1)' in source
    assert "indices" in source


def test_fixed_epoch20_groups_are_cross_checked():
    source = inspect.getsource(probe._load_selected_model_and_groups)
    assert "_bootstrap_groups_path" in source
    assert "np.array_equal" in source
    assert "epoch-20 bootstrap group" in source


def test_aggregate_contains_primary_c1_low_contrast():
    source = inspect.getsource(probe.aggregate)
    assert "paired_vs_C0" in source
    assert "test_ba" in source
    assert "C0" in source
    assert "primary_question" in source
