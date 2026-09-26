from __future__ import annotations

import numpy as np
import pandas as pd

from scripts import experiment_13_1_abstraction_generalization as exp


def test_run_mapping_and_primary_controls() -> None:
    assert exp.MODEL_KINDS == ("c1", "c2")
    assert set(exp.ANALYSIS_STATES) == {
        "syn_current",
        "pre_reset",
        "spike50",
    }
    assert len(exp.cache_specs()) == 6
    assert len(exp.metric_specs()) == 18
    assert len(exp.h_feature_specs()) == 18
    assert exp.HISTORY_MS == (50, 100, 250, 500, 750, 1000)
    assert exp.H_PHASES == (0.50, 0.75, 1.00)
    assert exp.H_HISTORY_MS_BY_PHASE == {
        0.50: (50, 100, 250, 500),
        0.75: (50, 100, 250, 500, 750),
        1.00: (50, 100, 250, 500, 750, 1000),
    }


def test_bounded_dtw_identity_symmetry_and_fixed_identity() -> None:
    rng = np.random.default_rng(7)
    a = rng.normal(size=(3, 16, 5)).astype(np.float32)
    b = rng.normal(size=(3, 16, 5)).astype(np.float32)

    identity = exp._batched_bounded_dtw(a, a)
    assert np.allclose(identity, 0.0, atol=1e-6)

    ab = exp._batched_bounded_dtw(a, b)
    ba = exp._batched_bounded_dtw(b, a)
    assert np.allclose(ab, ba, atol=1e-6)

    fixed = exp._fixed_alignment_distance(a, a)
    assert np.allclose(fixed, 0.0, atol=1e-6)


def test_pair_manifest_relations_and_determinism() -> None:
    rows = []
    index = 0
    for user in ("u0", "u1"):
        for label in ("A", "B"):
            for repeat in range(2):
                rows.append(
                    {
                        "split": "train",
                        "sample_index": index,
                        "sample_id": f"{user}_{label}_{repeat}",
                        "user": user,
                        "action": "0",
                        "label": label,
                        "y": 0 if label == "A" else 1,
                        "source_segment_index": index,
                        "valid_length": 80 + repeat,
                    }
                )
                index += 1
    frame = pd.DataFrame(rows)
    first = exp._make_pair_manifest(frame)
    second = exp._make_pair_manifest(frame)
    pd.testing.assert_frame_equal(first, second)

    sc_su = first[first["relation"] == "SC_SU"]
    assert bool((sc_su["user_i"] == sc_su["user_j"]).all())
    assert bool((sc_su["label_i"] == sc_su["label_j"]).all())

    sc_cu = first[first["relation"] == "SC_CU"]
    assert bool((sc_cu["user_i"] != sc_cu["user_j"]).all())
    assert bool((sc_cu["label_i"] == sc_cu["label_j"]).all())

    dc_cu = first[first["relation"] == "DC_CU"]
    assert bool((dc_cu["user_i"] != dc_cu["user_j"]).all())
    assert bool((dc_cu["label_i"] != dc_cu["label_j"]).all())


def test_class_residualization_uses_training_centroids_only() -> None:
    x_train = np.asarray([[1.0], [3.0], [10.0], [14.0]])
    y_train = np.asarray([0, 0, 1, 1])
    x_test = np.asarray([[100.0], [200.0]])
    y_test = np.asarray([0, 1])

    train_residual, test_residual = exp._residualize_by_class(
        x_train,
        y_train,
        x_test,
        y_test,
    )
    assert np.allclose(train_residual[:, 0], [-1.0, 1.0, -2.0, 2.0])
    assert np.allclose(test_residual[:, 0], [98.0, 188.0])


def test_h_fold_assignment_is_user_character_local() -> None:
    rows = []
    for user in ("u0", "u1"):
        for label in ("A", "B"):
            for repeat in range(4):
                rows.append(
                    {
                        "split": "train",
                        "sample_index": len(rows),
                        "sample_id": f"{user}_{label}_{repeat}",
                        "user": user,
                        "label": label,
                    }
                )
    folded = exp._make_h_folds(pd.DataFrame(rows))
    assert bool((folded["h_fold"] >= 0).all())
    for _, part in folded.groupby("user"):
        counts = part["h_fold"].value_counts()
        assert counts.max() - counts.min() <= 1


def test_history_eligibility_counts_complete_history() -> None:
    anchor_steps = np.asarray([0, 15, 31, 47, 63, 64])
    assert exp._history_steps(500, 64.0) == 32
    assert exp._history_steps(750, 64.0) == 48
    assert exp._history_steps(1000, 64.0) == 64

    eligible_32 = exp._history_eligible(anchor_steps, 32)
    assert eligible_32.tolist() == [
        False,
        False,
        True,
        True,
        True,
        True,
    ]
    eligible_64 = exp._history_eligible(anchor_steps, 64)
    assert eligible_64.tolist() == [
        False,
        False,
        False,
        False,
        True,
        True,
    ]


def test_class_coverage_contract_rejects_missing_class() -> None:
    expected = np.asarray([0, 1, 2])
    exp._assert_class_coverage(
        "complete",
        np.asarray([0, 1, 2, 2]),
        expected,
    )
    try:
        exp._assert_class_coverage(
            "missing",
            np.asarray([0, 1, 1]),
            expected,
        )
    except ValueError as exc:
        assert "class coverage mismatch" in str(exc)
    else:
        raise AssertionError("missing class should fail coverage check")


def test_dc_cu_negative_labels_are_balanced() -> None:
    rows = []
    index = 0
    for user in ("u0", "u1", "u2"):
        for label_index, label in enumerate(("A", "B", "C", "D")):
            for repeat in range(2):
                rows.append(
                    {
                        "split": "train",
                        "sample_index": index,
                        "sample_id": f"{user}_{label}_{repeat}",
                        "user": user,
                        "action": "0",
                        "label": label,
                        "y": label_index,
                        "source_segment_index": index,
                        "valid_length": 80 + label_index + repeat,
                    }
                )
                index += 1
    frame = exp._make_pair_manifest(pd.DataFrame(rows))
    negative = frame[frame["relation"] == "DC_CU"]
    counts = negative["label_j"].value_counts()
    assert counts.max() - counts.min() <= 1
