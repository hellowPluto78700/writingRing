from __future__ import annotations

from scripts.experiment_3_0_3_l3_bottleneck_ablation import (
    ARCHITECTURES,
    EVAL_ARCHITECTURES,
    EXPECTED_EVAL_RUNS,
    EXPECTED_TRAIN_RUNS,
    L1_SHIFTS,
    L1_WIDTH,
    L2_SHIFTS,
    L2_WIDTH,
    OBJECTIVES,
    SEEDS,
    TRAIN_ARCHITECTURES,
    architecture_layers,
    architecture_spec,
    eval_specs,
    last_layer_name,
    last_layer_width,
    train_specs,
)


def test_experiment_3_0_3_architecture_contract() -> None:
    assert L1_WIDTH == 128
    assert L2_WIDTH == 128
    assert L1_SHIFTS == (2, 3, 4)
    assert L2_SHIFTS == (2, 3, 4)

    assert architecture_spec("A").l3_width == 64
    assert architecture_spec("A").l3_shifts == (3,)
    assert architecture_spec("B").l3_width is None
    assert architecture_spec("B").l3_shifts == ()
    assert architecture_spec("C").l3_width == 128
    assert architecture_spec("C").l3_shifts == (3,)
    assert architecture_spec("D").l3_width == 64
    assert architecture_spec("D").l3_shifts == (2, 3, 4)
    assert architecture_spec("E").l3_width == 128
    assert architecture_spec("E").l3_shifts == (2, 3, 4)

    assert TRAIN_ARCHITECTURES == ("B", "C", "D", "E")
    assert EVAL_ARCHITECTURES == ("A", "B", "C", "D", "E")
    assert set(ARCHITECTURES) == set(EVAL_ARCHITECTURES)


def test_no_l3_and_last_layer_contract() -> None:
    assert architecture_layers("B") == (
        ("L1", 128, (2, 3, 4)),
        ("L2", 128, (2, 3, 4)),
    )
    assert last_layer_name("B") == "L2"
    assert last_layer_width("B") == 128

    for architecture in ("A", "C", "D", "E"):
        assert last_layer_name(architecture) == "L3"
        assert len(architecture_layers(architecture)) == 3


def test_causal_ablation_pairs_change_only_intended_l3_factor() -> None:
    a = architecture_spec("A")
    c = architecture_spec("C")
    d = architecture_spec("D")
    e = architecture_spec("E")

    # A vs C: width only.
    assert a.l3_shifts == c.l3_shifts == (3,)
    assert (a.l3_width, c.l3_width) == (64, 128)

    # A vs D: tau family only at width 64.
    assert a.l3_width == d.l3_width == 64
    assert a.l3_shifts == (3,)
    assert d.l3_shifts == (2, 3, 4)

    # C vs E: tau family only at width 128.
    assert c.l3_width == e.l3_width == 128
    assert c.l3_shifts == (3,)
    assert e.l3_shifts == (2, 3, 4)

    # D vs E: width only within multi-tau L3.
    assert d.l3_shifts == e.l3_shifts == (2, 3, 4)
    assert (d.l3_width, e.l3_width) == (64, 128)


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
