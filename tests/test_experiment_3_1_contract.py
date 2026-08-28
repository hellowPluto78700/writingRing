from __future__ import annotations

import numpy as np

from scripts import experiment_3_0_1_single_tau_objectives as base
from scripts import experiment_3_1_raw_vs_snn_representation_value as exp31


def test_experiment_3_1_protocol_contract() -> None:
    assert exp31.EXPERIMENT_ID == "experiment_3_1_raw_vs_snn_representation_value"
    assert exp31.PROTOCOL_VERSION == "matched_repr_v1"
    assert exp31.SEEDS == (11, 23, 101)
    assert exp31.run_specs() == [11, 23, 101]
    assert exp31.EXPECTED_RUNS == 3
    assert exp31.SOURCE_WIDTH == 128
    assert exp31.SOURCE_SHIFTS == (2, 3, 4)
    assert exp31.PCA_DIM == 128
    assert base.SPLIT_SEED == 12345
    assert exp31.REPRESENTATIONS == ("raw", "snn_l2", "raw_plus_snn")
    assert exp31.PROBE_TYPES == (
        "full_count",
        "fixed250_ordered",
        "fixed250_shuffled",
        "fixed250_pca128",
        "relative10_ordered",
        "relative10_shuffled",
        "relative10_pca128",
    )
    assert exp31.PRIMARY_PROBES == (
        "fixed250_ordered",
        "fixed250_pca128",
        "relative10_ordered",
        "relative10_pca128",
    )


def test_exp31_fixed_shuffle_pairs_raw_and_snn_and_preserves_invalid_tail() -> None:
    raw = np.arange(2 * 4 * 2, dtype=np.float64).reshape(2, 4, 2)
    snn = (100 + np.arange(2 * 4 * 3, dtype=np.float64)).reshape(2, 4, 3)
    lengths = np.asarray([31, 48])

    raw_shuffled, snn_shuffled = exp31._shared_fixed_shuffle(
        raw, snn, lengths, bin_steps=16, seed=123
    )

    # Sample 0 has two valid bins; bins 2-3 are invalid and must not move.
    assert np.array_equal(raw_shuffled[0, 2:], raw[0, 2:])
    assert np.array_equal(snn_shuffled[0, 2:], snn[0, 2:])

    # The paired permutation is recoverable from raw and must match the SNN order.
    raw_order0 = [int(np.where((raw[0, :2] == row).all(axis=1))[0][0]) for row in raw_shuffled[0, :2]]
    snn_expected0 = snn[0, raw_order0]
    assert np.array_equal(snn_shuffled[0, :2], snn_expected0)

    # Sample 1 has three valid bins; the fourth bin remains fixed.
    assert np.array_equal(raw_shuffled[1, 3:], raw[1, 3:])
    assert np.array_equal(snn_shuffled[1, 3:], snn[1, 3:])


def test_exp31_relative_shuffle_uses_identical_per_sample_bin_order() -> None:
    raw = np.arange(2 * 5 * 2, dtype=np.float64).reshape(2, 5, 2)
    snn = (100 + np.arange(2 * 5 * 3, dtype=np.float64)).reshape(2, 5, 3)
    raw_shuffled, snn_shuffled = exp31._shared_relative_shuffle(raw, snn, seed=99)

    for sample in range(2):
        order = [
            int(np.where((raw[sample] == row).all(axis=1))[0][0])
            for row in raw_shuffled[sample]
        ]
        assert np.array_equal(snn_shuffled[sample], snn[sample, order])


def test_exp31_compose_features_is_matched_and_fusion_is_concatenation() -> None:
    partition = {
        "raw": {
            "full_count": np.asarray([[1.0, 2.0], [3.0, 4.0]]),
            "fixed250_ordered": np.asarray([[1.0, 2.0], [3.0, 4.0]]),
            "fixed250_shuffled": np.asarray([[2.0, 1.0], [4.0, 3.0]]),
            "relative10_ordered": np.asarray([[5.0, 6.0], [7.0, 8.0]]),
            "relative10_shuffled": np.asarray([[6.0, 5.0], [8.0, 7.0]]),
        },
        "snn_l2": {
            "full_count": np.asarray([[10.0], [20.0]]),
            "fixed250_ordered": np.asarray([[10.0], [20.0]]),
            "fixed250_shuffled": np.asarray([[11.0], [21.0]]),
            "relative10_ordered": np.asarray([[30.0], [40.0]]),
            "relative10_shuffled": np.asarray([[31.0], [41.0]]),
        },
    }

    raw = exp31._compose_features(partition, "raw", "relative10_ordered")
    snn = exp31._compose_features(partition, "snn_l2", "relative10_ordered")
    fusion = exp31._compose_features(partition, "raw_plus_snn", "relative10_ordered")

    assert raw.shape == (2, 2)
    assert snn.shape == (2, 1)
    assert fusion.shape == (2, 3)
    assert np.array_equal(fusion, np.concatenate([raw, snn], axis=1))


def test_exp31_pca_probe_types_reuse_ordered_features() -> None:
    assert exp31._source_key("fixed250_pca128") == "fixed250_ordered"
    assert exp31._source_key("relative10_pca128") == "relative10_ordered"
    assert exp31._source_key("full_count") == "full_count"


def test_exp31_probe_randomization_is_independent_of_snn_seed() -> None:
    # Raw is deterministic across the three repeated array tasks, and a matched
    # representation/probe pair must use the same classifier/PCA random seed.
    assert exp31._probe_seed("raw", "relative10_pca128") == exp31._probe_seed(
        "raw", "relative10_pca128"
    )
    assert exp31._probe_seed("raw", "relative10_pca128") != exp31._probe_seed(
        "snn_l2", "relative10_pca128"
    )
