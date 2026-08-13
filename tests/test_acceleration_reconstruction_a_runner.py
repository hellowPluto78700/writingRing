from __future__ import annotations

from dataclasses import replace

import pandas as pd
from pandas.testing import assert_frame_equal

from scripts.run_experiment_a import _prepare_split
from snn.accel_reconstruction_eval.config import UserSplitConfig, experiment_a_config
from snn.accel_reconstruction_eval.datasets import (
    automatic_user_split,
    prepare_user_disjoint_splits,
)


def _manifest(users: list[str]) -> pd.DataFrame:
    rows: list[dict[str, str]] = []
    for user in users:
        for label in ("a", "b"):
            rows.append(
                {
                    "sample_id": f"{user}/sample_{label}",
                    "user": user,
                    "label": label,
                }
            )
    return pd.DataFrame(rows)


def test_prepare_split_forwards_exclusions_to_eligible_automatic_cohort() -> None:
    users = [f"user_{index}" for index in range(7)]
    manifest = _manifest(users)
    config = replace(
        experiment_a_config(random_seed=17),
        split=UserSplitConfig(
            train_fraction=0.50,
            val_fraction=0.25,
            excluded_users=("1", "user_5"),
        ),
    )

    result = _prepare_split(sample_manifest=manifest, config=config)

    eligible_users = [user for user in users if user not in {"user_1", "user_5"}]
    expected = automatic_user_split(
        eligible_users,
        train_fraction=0.50,
        val_fraction=0.25,
        seed=17,
    )
    assert (result.train_users, result.val_users, result.test_users) == tuple(
        tuple(split) for split in expected
    )
    assert set(result.sample_manifest["user"]) == set(eligible_users)
    assert not set(result.sample_manifest["sample_id"]).intersection(
        {
            "user_1/sample_a",
            "user_1/sample_b",
            "user_5/sample_a",
            "user_5/sample_b",
        }
    )


def test_prepare_split_without_exclusions_matches_existing_split_contract() -> None:
    manifest = _manifest([f"user_{index}" for index in range(6)])
    config = experiment_a_config(random_seed=23)

    result = _prepare_split(sample_manifest=manifest, config=config)
    expected = prepare_user_disjoint_splits(
        manifest,
        train_fraction=config.split.train_fraction,
        val_fraction=config.split.val_fraction,
        seed=config.random_seed,
        explicit_train_users=config.split.explicit_train_users,
        explicit_val_users=config.split.explicit_val_users,
        explicit_test_users=config.split.explicit_test_users,
        require_all_users_assigned=config.split.require_all_users_assigned,
        require_all_labels_in_all_splits=(
            config.split.require_all_labels_in_all_splits
        ),
    )

    assert (result.train_users, result.val_users, result.test_users) == (
        expected.train_users,
        expected.val_users,
        expected.test_users,
    )
    assert result.class_to_idx == expected.class_to_idx
    assert result.idx_to_class == expected.idx_to_class
    assert_frame_equal(result.sample_manifest, expected.sample_manifest)
    assert_frame_equal(result.split_summary, expected.split_summary)
    assert_frame_equal(result.label_split_counts, expected.label_split_counts)
