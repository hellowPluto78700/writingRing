from __future__ import annotations

import numpy as np

from scripts import experiment_3_0_1_single_tau_objectives as base
from scripts import experiment_3_0_5_frozen_representation_accessibility as exp305
from scripts import experiment_3_0_6_causal_temporal_decoding as exp306
from scripts import experiment_3_0_7_online_early_decision as exp307
from scripts.experiment_3_0_6.streaming_prefix import streaming_prefix_logits


def test_experiment_3_0_5_contract() -> None:
    assert exp305.SOURCE_WIDTH == 128
    assert exp305.SOURCE_SHIFTS == (2, 3, 4)
    assert exp305.OBJECTIVES == base.OBJECTIVES
    assert exp305.SEEDS == (11, 23, 101)
    assert exp305.PCA_DIM == 128
    assert exp305.EXPECTED_EVAL_RUNS == 9
    assert len(exp305.eval_specs()) == 9
    assert len(set(exp305.eval_specs())) == 9
    assert exp305.PROBE_TYPES == (
        "full_count",
        "fixed250_ordered",
        "fixed250_shuffled",
        "fixed250_pca128",
        "relative10_ordered",
        "relative10_shuffled",
        "relative10_pca128",
        "duration_only",
    )
    assert exp305.SELECTION_PROBES == (
        "full_count",
        "fixed250_pca128",
        "relative10_pca128",
    )


def test_exp305_fixed_shuffle_preserves_bins_and_invalid_tail() -> None:
    counts = np.arange(2 * 4 * 3, dtype=np.float64).reshape(2, 4, 3)
    lengths = np.asarray([31, 48])
    shuffled = exp305._shuffle_fixed_bins(counts, lengths, 16, 123)

    # Sample 0 has two valid bins; bins 2-3 are invalid padding and must remain fixed.
    assert np.array_equal(shuffled[0, 2:], counts[0, 2:])
    assert sorted(map(tuple, shuffled[0, :2])) == sorted(map(tuple, counts[0, :2]))

    # Sample 1 has three valid bins; the fourth invalid bin must remain fixed.
    assert np.array_equal(shuffled[1, 3:], counts[1, 3:])
    assert sorted(map(tuple, shuffled[1, :3])) == sorted(map(tuple, counts[1, :3]))


def test_experiment_3_0_6_run_mapping_and_causal_features() -> None:
    assert exp306.SEEDS == (11, 23, 101)
    assert exp306.DECODERS == (
        "current250",
        "cumulative250",
        "prefix250_uniform",
        "prefix250_weighted",
    )
    assert exp306.EXPECTED_RUNS == 12
    assert len(exp306.run_specs()) == 12
    assert len(set(exp306.run_specs())) == 12

    counts = np.arange(1 * 4 * 2, dtype=np.float64).reshape(1, 4, 2)
    current = exp306._feature_at_prefix(counts, "current250", 2)
    cumulative = exp306._feature_at_prefix(counts, "cumulative250", 2)
    prefix = exp306._feature_at_prefix(counts, "prefix250_uniform", 2)

    assert np.array_equal(current, counts[:, 2])
    assert np.array_equal(cumulative, counts[:, :3].sum(axis=1))
    expected_prefix = counts.copy()
    expected_prefix[:, 3] = 0
    assert np.array_equal(prefix, expected_prefix.reshape(1, -1))


def test_exp306_samplewise_weights_sum_to_one_per_gesture() -> None:
    partition = {
        "counts": np.ones((2, 4, 2), dtype=np.float64),
        "y": np.asarray([0, 1], dtype=np.int64),
        "lengths": np.asarray([32, 64], dtype=np.int64),
    }
    _, y_uniform, w_uniform = exp306._stack_training_rows(
        partition, "prefix250_uniform", 16
    )
    _, y_weighted, w_weighted = exp306._stack_training_rows(
        partition, "prefix250_weighted", 16
    )
    for labels, weights in ((y_uniform, w_uniform), (y_weighted, w_weighted)):
        for label in (0, 1):
            assert np.isclose(weights[labels == label].sum(), 1.0)


def test_exp306_streaming_prefix_logits_match_full_prefix_linear_model() -> None:
    rng = np.random.default_rng(7)
    n_bins = 4
    width = 3
    n_classes = 2
    feature_dim = n_bins * width
    counts = rng.normal(size=(n_bins, width))
    mean = rng.normal(size=feature_dim)
    scale = rng.uniform(0.3, 2.0, size=feature_dim)
    coef = rng.normal(size=(n_classes, feature_dim))
    intercept = rng.normal(size=n_classes)

    streaming = streaming_prefix_logits(
        counts,
        scaler_mean=mean,
        scaler_scale=scale,
        coef=coef,
        intercept=intercept,
    )
    full_rows = []
    for prefix_index in range(n_bins):
        prefix = np.zeros((n_bins, width), dtype=np.float64)
        prefix[: prefix_index + 1] = counts[: prefix_index + 1]
        x = prefix.reshape(-1)
        full_rows.append(coef @ ((x - mean) / scale) + intercept)
    expected = np.stack(full_rows)
    assert np.allclose(streaming, expected, rtol=1e-12, atol=1e-12)


def test_experiment_3_0_7_policy_grid_contract() -> None:
    assert exp307.SEEDS == (11, 23, 101)
    assert exp307.THRESHOLDS == (0.50, 0.60, 0.70, 0.80, 0.90, 0.95)
    assert exp307.STABILITY_M == (1, 2, 3)
    assert exp307.EXPECTED_RUNS == 3
    assert (exp307.STABILITY_M[1] - 1) * exp307.CHECKPOINT_MS == 250.0
    assert (exp307.STABILITY_M[2] - 1) * exp307.CHECKPOINT_MS == 500.0


def test_exp307_irreversible_policy_and_forced_terminal_decision() -> None:
    # Class 0 is high-confidence at 250 ms, then class 1 dominates. M=1 must
    # preserve the early class-0 commitment; a threshold too high forces the
    # terminal class-1 decision instead.
    logits = np.asarray(
        [
            [5.0, 0.0],
            [0.0, 5.0],
            [0.0, 6.0],
        ]
    )
    early = exp307._policy_prediction(
        logits,
        true_length=48,
        bin_steps=16,
        fs=64.0,
        temperature=1.0,
        threshold=0.90,
        stability=1,
    )
    assert early["prediction"] == 0
    assert early["forced"] is False
    assert early["decision_ms"] == 250.0

    forced = exp307._policy_prediction(
        logits,
        true_length=48,
        bin_steps=16,
        fs=64.0,
        temperature=1.0,
        threshold=0.9999999,
        stability=1,
    )
    assert forced["prediction"] == 1
    assert forced["forced"] is True
    assert forced["decision_ms"] == 750.0


def test_exp307_m2_requires_two_consecutive_checkpoints() -> None:
    logits = np.asarray(
        [
            [5.0, 0.0],
            [5.0, 0.0],
            [0.0, 5.0],
        ]
    )
    decision = exp307._policy_prediction(
        logits,
        true_length=64,
        bin_steps=16,
        fs=64.0,
        temperature=1.0,
        threshold=0.90,
        stability=2,
    )
    assert decision["prediction"] == 0
    assert decision["forced"] is False
    assert decision["decision_ms"] == 500.0
