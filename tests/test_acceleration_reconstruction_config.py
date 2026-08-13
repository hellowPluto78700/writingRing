from __future__ import annotations

from dataclasses import replace

import pytest

from snn.accel_reconstruction_eval.config import (
    UserSplitConfig,
    experiment_a_config,
    experiment_b_config,
    experiment_c_config,
    experiment_d_config,
)


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
