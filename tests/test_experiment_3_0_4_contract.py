from __future__ import annotations

from scripts import experiment_3_0_1_single_tau_objectives as base
from scripts import experiment_3_0_3_l3_bottleneck_ablation as prev
from scripts.experiment_3_0_4_l2_width_representation_capacity import (
    BASELINE_ARCHITECTURE,
    BASELINE_EXPERIMENT_ID,
    BASELINE_WIDTH,
    EVAL_WIDTHS,
    EXPECTED_EVAL_RUNS,
    EXPECTED_TRAIN_RUNS,
    L1_SHIFTS,
    L1_WIDTH,
    L2_SHIFTS,
    L2_WIDTHS,
    LAYER_PROBE_TYPES,
    OBJECTIVES,
    PCA_DIM,
    SEEDS,
    TRAIN_WIDTHS,
    L2WidthNet,
    balanced_group_slices,
    eval_specs,
    head_input_dim,
    train_specs,
)


def test_experiment_3_0_4_width_contract() -> None:
    assert L1_WIDTH == 128
    assert L1_SHIFTS == (2, 3, 4)
    assert L2_SHIFTS == (2, 3, 4)
    assert L2_WIDTHS == (32, 64, 128, 256)
    assert EVAL_WIDTHS == L2_WIDTHS
    assert TRAIN_WIDTHS == (32, 64, 256)
    assert BASELINE_WIDTH == 128
    assert BASELINE_ARCHITECTURE == "B"
    assert BASELINE_EXPERIMENT_ID == prev.EXPERIMENT_ID
    assert PCA_DIM == 128
    assert LAYER_PROBE_TYPES == (
        "full_count",
        "fixed250",
        "fixed250_pca128",
    )


def test_train_and_eval_specs_are_exact_cartesian_products() -> None:
    train = train_specs()
    evaluation = eval_specs()

    assert EXPECTED_TRAIN_RUNS == 27
    assert EXPECTED_EVAL_RUNS == 36
    assert len(train) == EXPECTED_TRAIN_RUNS
    assert len(evaluation) == EXPECTED_EVAL_RUNS
    assert len(set(train)) == len(train)
    assert len(set(evaluation)) == len(evaluation)

    assert set(train) == {
        (width, objective, seed)
        for width in TRAIN_WIDTHS
        for objective in OBJECTIVES
        for seed in SEEDS
    }
    assert set(evaluation) == {
        (width, objective, seed)
        for width in EVAL_WIDTHS
        for objective in OBJECTIVES
        for seed in SEEDS
    }


def test_tau_groups_cover_each_width_nearly_evenly() -> None:
    for width in L2_WIDTHS:
        slices = balanced_group_slices(width, L2_SHIFTS)
        assert tuple(slices) == L2_SHIFTS
        covered: list[int] = []
        sizes: list[int] = []
        for sl in slices.values():
            covered.extend(range(sl.start, sl.stop))
            sizes.append(sl.stop - sl.start)
        assert covered == list(range(width))
        assert max(sizes) - min(sizes) <= 1


def test_head_dimensions_scale_with_width_and_objective() -> None:
    T = 256
    bin_steps = 16
    for width in L2_WIDTHS:
        assert head_input_dim(width, "timestep_ce", T, bin_steps) == width
        assert head_input_dim(width, "relative10_sequence_ce", T, bin_steps) == width * 10
        assert head_input_dim(width, "fixed250_sequence_ce", T, bin_steps) == width * 16
        assert width * 16 >= PCA_DIM


def test_w128_state_dict_contract_matches_reused_303_no_l3() -> None:
    for objective in OBJECTIVES:
        seed = 11
        base.seed_all(base.dseed(seed, "shared_backbone_init"))
        old = prev.L3AblationNet(
            "B",
            objective,
            12,
            256,
            64.0,
            16,
        )
        base.seed_all(base.dseed(seed, "shared_backbone_init"))
        new = L2WidthNet(
            128,
            objective,
            12,
            256,
            64.0,
            16,
        )
        old_shapes = {
            key: tuple(value.shape)
            for key, value in old.state_dict().items()
        }
        new_shapes = {
            key: tuple(value.shape)
            for key, value in new.state_dict().items()
        }
        assert new_shapes == old_shapes
