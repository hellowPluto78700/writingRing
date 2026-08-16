from __future__ import annotations

from dataclasses import replace

import pytest

from snn.accel_reconstruction_eval.config import (
    UserSplitConfig,
    validate_reference_seed,
    experiment_a_config,
    experiment_b_config,
    experiment_c_config,
    experiment_d_config,
)
from snn.accel_reconstruction_eval.io import checkpoint_random_seed


def test_presets_validate_with_empty_exclusions() -> None:
    configs = (
        experiment_a_config(),
        experiment_b_config(),
        experiment_c_config(),
        experiment_d_config(),
    )

    for config in configs:
        config.validate()
        assert config.split.excluded_users == ()


def test_presets_synchronize_evaluation_seed_with_experiment_seed() -> None:
    configs = (
        experiment_a_config(random_seed=17),
        experiment_b_config(random_seed=17),
        experiment_c_config(random_seed=17),
        experiment_d_config(random_seed=17),
    )

    for config in configs:
        assert config.random_seed == 17
        assert config.evaluation.random_seed == 17


def test_config_rejects_divergent_evaluation_seed() -> None:
    config = replace(
        experiment_a_config(random_seed=17),
        evaluation=replace(
            experiment_a_config(random_seed=17).evaluation,
            random_seed=18,
        ),
    )

    with pytest.raises(ValueError, match="evaluation.random_seed must match"):
        config.validate()


def test_checkpoint_random_seed_prefers_consistent_top_level_identity() -> None:
    checkpoint = {
        "random_seed": 23,
        "experiment_config": {"random_seed": 23},
    }

    assert checkpoint_random_seed(checkpoint) == 23


def test_checkpoint_random_seed_supports_nested_legacy_identity() -> None:
    assert checkpoint_random_seed({"experiment_config": {"random_seed": 29}}) == 29


def test_checkpoint_random_seed_rejects_conflicting_identity() -> None:
    with pytest.raises(ValueError, match="random_seed conflicts"):
        checkpoint_random_seed(
            {"random_seed": 23, "experiment_config": {"random_seed": 29}}
        )


def test_checkpoint_random_seed_rejects_missing_identity() -> None:
    with pytest.raises(ValueError, match="missing random_seed"):
        checkpoint_random_seed({})


def test_validate_reference_seed_rejects_mismatch() -> None:
    with pytest.raises(ValueError, match="random_seed mismatch"):
        validate_reference_seed(experiment_b_config(random_seed=17), 23)


def test_excluded_users_are_canonical_immutable_and_serialized() -> None:
    split = UserSplitConfig(
        excluded_users=("17", "user_2", "research participant", "USER_3")
    )

    assert split.excluded_users == (
        "user_17",
        "user_2",
        "research participant",
        "USER_3",
    )
    assert isinstance(split.excluded_users, tuple)

    config = replace(experiment_a_config(), split=split)
    assert config.to_dict()["split"]["excluded_users"] == split.excluded_users


@pytest.mark.parametrize("excluded_users", ["17", 17, None, {"17"}])
def test_excluded_users_reject_scalar_or_non_sequence_input(
    excluded_users: object,
) -> None:
    with pytest.raises(TypeError, match="excluded_users.*sequence"):
        UserSplitConfig(excluded_users=excluded_users)  # type: ignore[arg-type]


def test_excluded_users_reject_canonical_duplicates() -> None:
    with pytest.raises(ValueError, match="duplicate canonical"):
        UserSplitConfig(excluded_users=("17", "user_17"))


def test_included_labels_are_ordered_immutable_and_canonical() -> None:
    split = UserSplitConfig(included_labels=("b", 3, "a"))

    assert split.included_labels == ("b", "3", "a")
    assert isinstance(split.included_labels, tuple)


def test_included_labels_reject_duplicates() -> None:
    with pytest.raises(ValueError, match="duplicate canonical"):
        UserSplitConfig(included_labels=("a", "a"))


@pytest.mark.parametrize("included_labels", ["a", 3])
def test_included_labels_reject_scalar_inputs(included_labels: object) -> None:
    with pytest.raises(TypeError, match="included_labels.*sequence"):
        UserSplitConfig(included_labels=included_labels)  # type: ignore[arg-type]
