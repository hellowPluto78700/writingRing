from __future__ import annotations

import numpy as np

from scripts import experiment_18_3_discriminative_coordinate as exp


def test_axes_are_locked():
    assert exp.SEEDS == (11, 23, 37)
    assert exp.CASES == ("I_WCCE", "I_MWCCE", "U_WCCE", "U_MWCCE")
    assert len(exp.specs()) == 12
    assert exp.GROUP_FRACTION == 0.30
    assert exp.RANDOM_REPLICATES == 20


def test_centered_alignment_is_scale_invariant():
    profile = np.array([
        [1.0, 2.0],
        [2.0, 0.0],
        [3.0, 1.0],
    ])
    weight = np.array([
        [0.0, 4.0],
        [1.0, 2.0],
        [2.0, 3.0],
    ])
    a = exp._centered_column_cosine(profile, weight)
    b = exp._centered_column_cosine(7.0 * profile, 3.0 * weight)
    assert np.allclose(a, b, equal_nan=True)


def test_aligned_signal_changes_sign_with_head_reversal():
    profile = np.array([
        [0.0, 2.0],
        [1.0, 1.0],
        [2.0, 0.0],
    ])
    weight = profile.copy()
    positive = exp._aligned_signal(profile, weight)
    negative = exp._aligned_signal(profile, -weight)
    assert np.all(positive > 0)
    assert np.allclose(negative, -positive)


def test_class_profiles_have_class_by_neuron_shape():
    values = np.array([
        [1.0, 0.0],
        [3.0, 2.0],
        [2.0, 4.0],
        [4.0, 6.0],
    ])
    y = np.array([0, 0, 1, 1])
    result = exp._class_profiles(values, y, 2)
    assert result.shape == (2, 2)
    assert np.allclose(result[0], [2.0, 1.0])
    assert np.allclose(result[1], [3.0, 5.0])


def test_margin_contribution_prefers_true_class_aligned_feature():
    occupancy = np.array([
        [2.0, 0.0],
        [0.0, 2.0],
    ])
    weight = np.array([
        [1.0, -1.0],
        [-1.0, 1.0],
    ])
    y = np.array([0, 1])
    scores = occupancy @ weight.T
    contribution = exp._margin_contribution(occupancy, scores, y, weight)
    assert np.all(contribution > 0)


def test_single_neuron_mean_replacement_removes_class_variation():
    occupancy = {
        "train": np.array([[2.0, 0.0], [0.0, 2.0]]),
        "val": np.array([[2.0, 0.0], [0.0, 2.0]]),
        "test": np.array([[2.0, 0.0], [0.0, 2.0]]),
    }
    weight = np.array([[1.0, -1.0], [-1.0, 1.0]])
    arrays = {
        "train_y": np.array([0, 1]),
        "val_y": np.array([0, 1]),
        "test_y": np.array([0, 1]),
    }
    baseline, result = exp._single_neuron_ablation(occupancy, weight, arrays)
    assert baseline["test"]["ba"] == 1.0
    assert result["test"]["mean_replacement_delta_ce"][0] > 0
    assert result["test"]["mean_replacement_drop_ba_pp"][0] >= 0


def test_train_only_ranking_is_stable_and_dimension_matched():
    values = np.array([1.0, 4.0, 3.0, 2.0, 0.0])
    high = exp._ranking_indices(values, 0.4)
    low = exp._ranking_indices(values, 0.4, descending=False)
    assert np.array_equal(high, np.array([1, 2]))
    assert np.array_equal(low, np.array([0, 4]))
    assert len(high) == len(low)


def test_standardized_ols_exposes_requested_coefficients():
    x = np.arange(10, dtype=float)
    result = exp._standardized_ols(
        {"rate": x, "selectivity": x[::-1]},
        2.0 * x + 1.0,
    )
    assert result["n"] == 10
    assert "beta_rate" in result
    assert "beta_selectivity" in result
    assert np.isfinite(result["r2"])
