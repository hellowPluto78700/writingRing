from __future__ import annotations

import json

import numpy as np
import torch
import torch.nn.functional as F

from core_benchmark_v1.model import BenchmarkNet
from core_benchmark_v1.protocol import Protocol
from scripts import experiment_14_1_dual_loader_accumulation as exp


def test_run_manifest_and_frozen_constants() -> None:
    specs = exp.phase1_specs()
    assert len(specs) == 24
    assert exp.SEEDS == (11, 23, 37)
    assert exp.SHIFTS == ((2, 3, 4), (2, 3, 4))
    assert exp.PHASE_LAMBDAS == (0.01, 0.03, 0.06, 0.10)
    assert exp.PREFIX_LAMBDAS == (0.10, 0.25, 0.50)
    assert exp.PREFIX_PHASES == (0.50, 0.75)
    assert exp.PREFIX_WEIGHTS == (0.5, 0.5)
    assert sum(spec.case == "C0_dual_null" for spec in specs) == 3
    assert sum(spec.case == "P_phase_cu" for spec in specs) == 12
    assert sum(spec.case == "A_prefix_wcce" for spec in specs) == 9


def test_phase2_and_final_eval_specs_are_preregistered() -> None:
    selection = {"phase": {"lambda": 0.03}, "prefix": {"lambda": 0.25}}
    phase2 = exp.phase2_specs(selection)
    assert len(phase2) == 12
    by_case = {case: [spec for spec in phase2 if spec.case == case] for case in {
        "J0_combined", "Jp_half_phase", "Ja_half_prefix", "Jb_half_both"
    }}
    assert all(len(items) == 3 for items in by_case.values())
    assert all(spec.lambda_phase == 0.03 and spec.lambda_prefix == 0.25 for spec in by_case["J0_combined"])
    assert all(spec.lambda_phase == 0.015 and spec.lambda_prefix == 0.25 for spec in by_case["Jp_half_phase"])
    assert all(spec.lambda_phase == 0.03 and spec.lambda_prefix == 0.125 for spec in by_case["Ja_half_prefix"])
    assert all(spec.lambda_phase == 0.015 and spec.lambda_prefix == 0.125 for spec in by_case["Jb_half_both"])
    final = exp.final_eval_specs(selection)
    assert len(final) == 21
    assert len({spec.key for spec in final}) == 21


def test_same_seed_c0_initialization_matches_core_o0() -> None:
    p = Protocol()
    c0 = BenchmarkNet(exp._run(exp.LossSpec("C0_dual_null", 11)), p)
    o0 = BenchmarkNet(exp._core_o0_run(11), p)
    for key, value in c0.state_dict().items():
        assert torch.equal(value, o0.state_dict()[key]), key


def test_prefix_wcce_is_one_ce_per_prefix_after_temporal_mean() -> None:
    evidence = torch.tensor(
        [
            [[3.0, 0.0], [1.0, 0.0], [0.0, 2.0], [0.0, 2.0]],
            [[0.0, 3.0], [0.0, 1.0], [2.0, 0.0], [2.0, 0.0]],
        ],
        requires_grad=True,
    )
    lengths = torch.tensor([4, 4])
    y = torch.tensor([0, 1])
    mean50 = evidence[:, :2].mean(dim=1)
    mean75 = evidence[:, :3].mean(dim=1)
    expected = 0.5 * F.cross_entropy(mean50, y) + 0.5 * F.cross_entropy(mean75, y)
    actual = exp.prefix_wcce(evidence, lengths, y)
    assert torch.allclose(actual, expected)
    actual.backward()
    assert evidence.grad is not None
    assert torch.isfinite(evidence.grad).all()


def test_prefix_means_use_per_sample_ceil_phase_lengths() -> None:
    evidence = torch.arange(2 * 6 * 2, dtype=torch.float32).reshape(2, 6, 2)
    lengths = torch.tensor([5, 6])
    mean50, mean75 = exp._prefix_means(evidence, lengths)
    expected50 = torch.stack((evidence[0, :3].mean(0), evidence[1, :3].mean(0)))
    expected75 = torch.stack((evidence[0, :4].mean(0), evidence[1, :5].mean(0)))
    assert torch.allclose(mean50, expected50)
    assert torch.allclose(mean75, expected75)


def test_auxiliary_schedule_has_warmup_ramp_and_plateau() -> None:
    target = 0.5
    assert exp._aux_weight(target, 1) == 0.0
    assert exp._aux_weight(target, exp.WARMUP_EPOCHS) == 0.0
    assert 0.0 < exp._aux_weight(target, 20) < target
    assert exp._aux_weight(target, exp.RAMP_END_EPOCH) == target
    assert exp._aux_weight(target, 100) == target


def _summary(case: str, seed: int, lp: float = 0.0, la: float = 0.0) -> dict[str, float | int | str]:
    return {
        "case": case,
        "seed": seed,
        "lambda_phase": lp,
        "lambda_prefix": la,
        "native_train_ba": 0.80,
        "native_val_ba": 0.60,
        "native_val_ce": 1.0,
        "val_retrieval_l2_spike": 0.50,
        "val_retrieval_l2_pre_reset": 0.50,
        "val_wholecount_no_bias_ba": 0.55,
        "val_relative10_no_bias_ba": 0.65,
        "val_fixed250_no_bias_ba": 0.57,
        "val_native_prefix50_ba": 0.45,
        "val_native_prefix75_ba": 0.55,
        "val_native_prefix100_ba": 0.60,
        "val_collapse_gap_pp": 10.0,
    }


def test_validation_selection_uses_constraints_and_smaller_phase_tie(tmp_path) -> None:
    config = exp.Config(tmp_path, tmp_path / "results", tmp_path / "core")
    for spec in exp.phase1_specs():
        row = _summary(spec.case, spec.seed, spec.lambda_phase, spec.lambda_prefix)
        if spec.case == "P_phase_cu":
            row["val_retrieval_l2_spike"] = {0.01: 0.55, 0.03: 0.60, 0.06: 0.604, 0.10: 0.62}[spec.lambda_phase]
            if spec.lambda_phase == 0.10:
                row["native_val_ba"] = 0.58
        if spec.case == "A_prefix_wcce":
            row["val_wholecount_no_bias_ba"] = {0.10: 0.56, 0.25: 0.60, 0.50: 0.59}[spec.lambda_prefix]
            row["val_relative10_no_bias_ba"] = {0.10: 0.65, 0.25: 0.645, 0.50: 0.62}[spec.lambda_prefix]
            row["val_collapse_gap_pp"] = 100.0 * (
                row["val_relative10_no_bias_ba"] - row["val_wholecount_no_bias_ba"]
            )
        directory = config.results_dir / "runs" / spec.key
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "validation_summary.json").write_text(json.dumps(row))
    selection = exp.select_phase1(config)
    assert selection["phase"]["lambda"] == 0.03
    assert selection["prefix"]["lambda"] == 0.25


def test_phase1_selection_source_does_not_reference_test_arrays() -> None:
    names = exp.select_phase1.__code__.co_names
    constants = exp.select_phase1.__code__.co_consts
    text = " ".join(str(item) for item in (*names, *constants))
    assert "test_" not in text


def _balanced_aux_arrays() -> dict[str, np.ndarray]:
    labels, users, ids = [], [], []
    for label in range(12):
        for user_index in range(4):
            user = f"user_{user_index}"
            for sample in range(4):
                labels.append(label)
                users.append(user)
                ids.append(f"{label}:{user}:{sample}")
    n = len(labels)
    return {
        "train_y": np.asarray(labels, dtype=np.int64),
        "train_users": np.asarray(users),
        "train_ids": np.asarray(ids),
        "train_x": np.zeros((n, 4, 2), dtype=np.float32),
        "train_lengths": np.full(n, 4, dtype=np.int64),
    }


def test_auxiliary_batches_are_unique_and_have_cross_user_positives() -> None:
    arrays = _balanced_aux_arrays()
    batches = exp._validated_aux_batches(arrays, seed=11, epoch=1)
    assert batches
    for indices in batches:
        assert len(indices) == len(np.unique(indices))
        y = arrays["train_y"][indices]
        users = arrays["train_users"][indices]
        for i in range(len(indices)):
            assert np.any((y == y[i]) & (users != users[i]))


def test_auxiliary_batches_reject_replacement_requirement() -> None:
    arrays = _balanced_aux_arrays()
    keep = np.ones(len(arrays["train_y"]), dtype=bool)
    target = np.flatnonzero((arrays["train_y"] == 0) & (arrays["train_users"] == "user_0"))
    keep[target[-1]] = False
    reduced = {key: value[keep] for key, value in arrays.items()}
    try:
        exp._validated_aux_batches(reduced, seed=11, epoch=1)
    except ValueError as error:
        assert "without replacement" in str(error)
    else:
        raise AssertionError("Expected undersized class-user cell to fail")


def test_c0_bypasses_auxiliary_batches() -> None:
    assert exp._needs_aux_batch(exp.LossSpec("C0_dual_null", 11)) is False
    assert exp._needs_aux_batch(exp.LossSpec("A_prefix_wcce", 11, lambda_prefix=0.25)) is False
    assert exp._needs_aux_batch(exp.LossSpec("P_phase_cu", 11, lambda_phase=0.03)) is True


def test_native_prefix_diagnostics_report_all_horizons() -> None:
    evidence = np.array([
        [[2.0, 0.0], [1.0, 0.0], [1.0, 0.5], [1.0, 0.0]],
        [[0.0, 2.0], [0.0, 1.0], [0.5, 1.0], [0.0, 1.0]],
    ], dtype=np.float32)
    payload = exp._native_prefix_diagnostics(
        evidence,
        np.array([4, 4], dtype=np.int64),
        np.array([0, 1], dtype=np.int64),
        "test",
    )
    assert [row["phase"] for row in payload["rows"]] == [0.5, 0.75, 1.0]
    assert all(row["ba"] == 1.0 for row in payload["rows"])
    assert 0.0 <= payload["summary"]["monotonic_margin_fraction"] <= 1.0
    assert 0.0 <= payload["summary"]["support_sign_reversal_rate"] <= 1.0
    assert payload["summary"]["support_cancellation_ratio_mean"] >= 0.0
