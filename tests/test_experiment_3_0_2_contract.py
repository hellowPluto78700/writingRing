from __future__ import annotations

from scripts.experiment_3_0_2_hidden_multitau_architectures import (
    ARCHITECTURES,
    EVAL_ARCHITECTURES,
    EXPECTED_EVAL_RUNS,
    EXPECTED_TRAIN_RUNS,
    OBJECTIVES,
    SEEDS,
    TRAIN_ARCHITECTURES,
    WIDTHS,
    architecture_shifts,
    balanced_group_slices,
    eval_specs,
    train_specs,
)


def test_experiment_3_0_2_architecture_contract() -> None:
    assert ARCHITECTURES == {
        "A": ((3,), (3,), (3,)),
        "B": ((2, 3), (2, 3), (3,)),
        "C": ((2, 3, 4), (2, 3, 4), (3,)),
        "D": ((2, 3), (2, 3, 4, 5), (3,)),
        "E": ((2, 3), (2, 3, 4), (3,)),
    }
    assert TRAIN_ARCHITECTURES == ("B", "C", "D", "E")
    assert EVAL_ARCHITECTURES == ("A", "B", "C", "D", "E")
    assert all(architecture_shifts(name)[2] == (3,) for name in EVAL_ARCHITECTURES)


def test_train_and_eval_spec_counts_are_exact_and_unique() -> None:
    train = train_specs()
    evaluation = eval_specs()

    assert EXPECTED_TRAIN_RUNS == 36
    assert EXPECTED_EVAL_RUNS == 45
    assert len(train) == EXPECTED_TRAIN_RUNS
    assert len(evaluation) == EXPECTED_EVAL_RUNS
    assert len(set(train)) == len(train)
    assert len(set(evaluation)) == len(evaluation)

    assert set(train) == {
        (architecture, objective, seed)
        for architecture in TRAIN_ARCHITECTURES
        for objective in OBJECTIVES
        for seed in SEEDS
    }
    assert set(evaluation) == {
        (architecture, objective, seed)
        for architecture in EVAL_ARCHITECTURES
        for objective in OBJECTIVES
        for seed in SEEDS
    }


def test_multitau_group_slices_cover_each_layer_without_overlap() -> None:
    for architecture in EVAL_ARCHITECTURES:
        for width, shifts in zip(WIDTHS, architecture_shifts(architecture), strict=True):
            slices = balanced_group_slices(width, shifts)
            assert tuple(slices) == shifts
            covered: list[int] = []
            sizes: list[int] = []
            for sl in slices.values():
                covered.extend(range(sl.start, sl.stop))
                sizes.append(sl.stop - sl.start)
            assert covered == list(range(width))
            assert max(sizes) - min(sizes) <= 1
