from __future__ import annotations

import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from snn.accel_reconstruction_eval.datasets import (
    automatic_user_split,
    prepare_user_disjoint_splits,
)


def _manifest(users: list[str], *, include_unique_label: bool = False) -> pd.DataFrame:
    rows: list[dict[str, str]] = []
    for user in users:
        rows.extend(
            {
                "sample_id": f"{user}/sample_{label}",
                "user": user,
                "label": label,
            }
            for label in ("a", "b")
        )
    if include_unique_label:
        rows.append(
            {
                "sample_id": "user_99/sample_unique",
                "user": "user_99",
                "label": "unique",
            }
        )
    return pd.DataFrame(rows)


def test_automatic_split_uses_eligible_cohort_for_single_and_multiple_exclusions() -> None:
    users = [f"user_{index}" for index in range(7)]
    excluded = ("user_1", "user_5")
    manifest = _manifest(users)

    result = prepare_user_disjoint_splits(
        manifest,
        train_fraction=0.50,
        val_fraction=0.25,
        seed=17,
        excluded_users=excluded,
    )

    eligible = [user for user in users if user not in excluded]
    expected = automatic_user_split(
        eligible,
        train_fraction=0.50,
        val_fraction=0.25,
        seed=17,
    )
    expected_users = tuple(tuple(split) for split in expected)
    assert (result.train_users, result.val_users, result.test_users) == expected_users
    assert len(result.train_users) == 2
    assert len(result.val_users) == 1
    assert len(result.test_users) == 2
    assert not set(excluded).intersection(result.sample_manifest["user"])


def test_numeric_exclusion_alias_removes_canonical_user() -> None:
    result = prepare_user_disjoint_splits(
        _manifest([f"user_{index}" for index in range(4)]),
        excluded_users=("2",),
        require_all_labels_in_all_splits=False,
    )

    assert "user_2" not in result.train_users + result.val_users + result.test_users
    assert "user_2" not in set(result.sample_manifest["user"])


def test_excluded_alias_duplicates_are_rejected() -> None:
    with pytest.raises(ValueError, match="duplicates"):
        prepare_user_disjoint_splits(
            _manifest([f"user_{index}" for index in range(4)]),
            excluded_users=("17", "user_17"),
        )


def test_explicit_split_overlap_with_exclusion_fails_before_eligibility_checks() -> None:
    with pytest.raises(ValueError, match="Explicit train users include excluded"):
        prepare_user_disjoint_splits(
            _manifest([f"user_{index}" for index in range(4)]),
            explicit_train_users=("2",),
            explicit_val_users=("user_1",),
            explicit_test_users=("user_3",),
            excluded_users=("user_2",),
        )


def test_absent_exclusion_is_allowed() -> None:
    result = prepare_user_disjoint_splits(
        _manifest([f"user_{index}" for index in range(4)]),
        excluded_users=("user_99",),
        require_all_labels_in_all_splits=False,
    )

    assert len(result.sample_manifest) == 8


def test_too_few_eligible_users_fails_after_exclusion() -> None:
    with pytest.raises(ValueError, match="At least three users"):
        prepare_user_disjoint_splits(
            _manifest([f"user_{index}" for index in range(5)]),
            excluded_users=("user_0", "user_1", "user_2"),
        )


def test_label_coverage_is_checked_after_exclusion() -> None:
    manifest = _manifest(["user_0", "user_1", "user_2", "user_3"])
    manifest = manifest.loc[
        ~((manifest["user"] == "user_3") & (manifest["label"] == "b"))
    ].reset_index(drop=True)

    with pytest.raises(ValueError, match="label is absent from a split"):
        prepare_user_disjoint_splits(
            manifest,
            explicit_train_users=("user_0",),
            explicit_val_users=("user_1",),
            explicit_test_users=("user_3",),
            excluded_users=("user_2",),
        )


def test_class_mapping_is_checked_against_eligible_labels() -> None:
    manifest = _manifest(["user_0", "user_1", "user_2", "user_3"], include_unique_label=True)

    with pytest.raises(ValueError, match="class_to_idx label set differs"):
        prepare_user_disjoint_splits(
            manifest,
            excluded_users=("user_99",),
            class_to_idx={"a": 0, "b": 1, "unique": 2},
            require_all_labels_in_all_splits=False,
        )


def test_empty_exclusion_preserves_existing_split_result() -> None:
    manifest = _manifest([f"user_{index}" for index in range(6)])
    kwargs = {
        "explicit_train_users": ("1", "user_2"),
        "explicit_val_users": ("user_3",),
        "explicit_test_users": ("user_0", "user_4", "user_5"),
    }

    default_result = prepare_user_disjoint_splits(manifest, **kwargs)
    empty_result = prepare_user_disjoint_splits(manifest, excluded_users=(), **kwargs)

    assert (default_result.train_users, default_result.val_users, default_result.test_users) == (
        empty_result.train_users,
        empty_result.val_users,
        empty_result.test_users,
    )
    assert default_result.class_to_idx == empty_result.class_to_idx
    assert_frame_equal(default_result.sample_manifest, empty_result.sample_manifest)
    assert_frame_equal(default_result.split_summary, empty_result.split_summary)
    assert_frame_equal(default_result.label_split_counts, empty_result.label_split_counts)
