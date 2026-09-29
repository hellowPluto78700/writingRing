from __future__ import annotations

from dataclasses import asdict, replace
import json

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from core_benchmark_v1.model import BenchmarkNet, mean_logits
from core_benchmark_v1.protocol import Protocol
from core_benchmark_v1.runner import smoke_protocol
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
    assert exp.AUX_CLASSES_PER_BATCH == 8
    assert exp.AUX_USERS_PER_CLASS == 8
    assert exp.AUX_SAMPLES_PER_USER == 1
    assert exp.AUX_BATCH_SIZE == 64
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
    spec = exp.LossSpec("C0_dual_null", 11)
    c0 = BenchmarkNet(exp._run(spec), p)
    o0 = BenchmarkNet(exp._core_o0_run(11), p)
    for key, value in c0.state_dict().items():
        assert torch.equal(value, o0.state_dict()[key]), key
    contract = exp._verify_c0_current_core_contract(
        spec,
        p,
        {key: value.detach().cpu().clone() for key, value in c0.state_dict().items()},
    )
    assert contract is not None
    assert contract["initialization_bitwise_match"] is True
    assert contract["auxiliary_path_bypassed"] is True


def test_c0_and_current_core_o0_optimizer_steps_match() -> None:
    p = replace(smoke_protocol(), max_epochs=2, min_epochs=1, patience=1)
    spec = exp.LossSpec("C0_dual_null", 11)
    c0 = BenchmarkNet(exp._run(spec), p)
    core = BenchmarkNet(exp._core_o0_run(11), p)
    c0_optimizer = torch.optim.Adam(
        c0.parameters(),
        lr=p.learning_rate,
        weight_decay=p.weight_decay,
    )
    core_optimizer = torch.optim.Adam(
        core.parameters(),
        lr=p.learning_rate,
        weight_decay=p.weight_decay,
    )
    generator = torch.Generator().manual_seed(991)
    x = torch.rand(8, p.steps, p.input_channels, generator=generator)
    y = torch.tensor([0, 1, 2, 0, 1, 2, 0, 1], dtype=torch.long)
    lengths = torch.tensor([32, 29, 27, 25, 23, 21, 19, 17], dtype=torch.long)

    for start in (0, 4):
        batch = slice(start, start + 4)
        for model, optimizer in ((c0, c0_optimizer), (core, core_optimizer)):
            optimizer.zero_grad(set_to_none=True)
            trajectory = model(x[batch], lengths[batch])
            loss = F.cross_entropy(mean_logits(trajectory["evidence"], lengths[batch]), y[batch])
            loss.backward()
            optimizer.step()
        for key, value in c0.state_dict().items():
            assert torch.equal(value, core.state_dict()[key]), (start, key)


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


def _balanced_aux_arrays(
    users_per_class: int = 10,
    samples_pattern: tuple[int, ...] = (1, 2, 2, 3),
) -> dict[str, np.ndarray]:
    labels, users, ids = [], [], []
    for label in range(12):
        for user_index in range(users_per_class):
            user = f"user_{user_index}"
            count = samples_pattern[(label + user_index) % len(samples_pattern)]
            for sample in range(count):
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


def test_auxiliary_batches_match_user_diverse_geometry() -> None:
    arrays = _balanced_aux_arrays()
    batches = exp._validated_aux_batches(arrays, seed=11, epoch=1, task_batch_size=128)
    assert batches
    for indices in batches:
        assert len(indices) == exp.AUX_BATCH_SIZE == 64
        assert len(indices) == len(np.unique(indices))
        y = arrays["train_y"][indices]
        users = arrays["train_users"][indices]
        unique_classes, counts = np.unique(y, return_counts=True)
        assert len(unique_classes) == exp.AUX_CLASSES_PER_BATCH
        assert np.all(counts == exp.AUX_USERS_PER_CLASS)
        for label in unique_classes:
            class_users = users[y == label]
            assert len(np.unique(class_users)) == exp.AUX_USERS_PER_CLASS
        for i in range(len(indices)):
            positive = (y == y[i]) & (users != users[i])
            assert int(positive.sum()) == exp.AUX_USERS_PER_CLASS - 1


def test_auxiliary_sampler_allows_one_sample_class_user_cells() -> None:
    arrays = _balanced_aux_arrays(samples_pattern=(1,))
    batches = exp._validated_aux_batches(arrays, seed=23, epoch=5, task_batch_size=128)
    assert batches
    assert all(len(batch) == 64 for batch in batches)


def test_auxiliary_sampler_ignores_zero_sample_user_for_that_class() -> None:
    arrays = _balanced_aux_arrays(users_per_class=9, samples_pattern=(1,))
    mask = ~((arrays["train_y"] == 0) & (arrays["train_users"] == "user_0"))
    reduced = {key: value[mask] for key, value in arrays.items()}
    report = exp._aux_sampler_report(reduced)
    class0 = next(row for row in report if row["class_id"] == 0)
    assert class0["eligible_train_users"] == 8
    batches = exp._validated_aux_batches(reduced, seed=37, epoch=2, task_batch_size=128)
    assert batches


def test_auxiliary_sampler_rejects_class_with_fewer_than_eight_users() -> None:
    arrays = _balanced_aux_arrays(users_per_class=8, samples_pattern=(1,))
    mask = ~((arrays["train_y"] == 0) & (arrays["train_users"] == "user_0"))
    reduced = {key: value[mask] for key, value in arrays.items()}
    try:
        exp._validated_aux_batches(reduced, seed=11, epoch=1, task_batch_size=128)
    except ValueError as error:
        assert "need at least 8" in str(error)
    else:
        raise AssertionError("Expected class with <8 eligible users to fail")


def test_task_batch_trace_matches_core_dataloader_first_epoch() -> None:
    arrays = _balanced_aux_arrays(users_per_class=8, samples_pattern=(1,))
    p = replace(Protocol(), batch_size=7)
    actual = exp._task_batch_trace(arrays, p, seed=11)

    index_dataset = torch.utils.data.TensorDataset(
        torch.arange(len(arrays["train_y"]), dtype=torch.long)
    )
    replay = torch.utils.data.DataLoader(
        index_dataset,
        batch_size=p.batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(exp.paired_seed(11, "loader:train")),
        num_workers=0,
    )
    expected: list[list[str]] = []
    for batch_index, (indices,) in enumerate(replay):
        if batch_index >= exp.TASK_BATCH_TRACE_COUNT:
            break
        expected.append([str(arrays["train_ids"][int(index)]) for index in indices.tolist()])
    assert actual == expected


def test_c0_historical_reference_mismatch_is_non_blocking_but_provenance_is_hard(
    tmp_path,
) -> None:
    core = tmp_path / "core"
    run_dir = core / "runs" / "O0__seed11"
    run_dir.mkdir(parents=True)
    (core / "protocol.lock.json").write_text(
        json.dumps({"identity": "core-identity"}),
        encoding="utf-8",
    )

    p = Protocol()
    historical_model = BenchmarkNet(exp._core_o0_run(11), p)
    historical_state = {
        key: value.detach().cpu().clone()
        for key, value in historical_model.state_dict().items()
    }
    checkpoint = run_dir / "checkpoint.pt"
    torch.save(
        {
            "identity": "core-identity",
            "run": asdict(exp._core_o0_run(11)),
            "model_state_dict": historical_state,
            "best_epoch": 93,
            "best_val": {"ba": 0.55, "mean_logit_ce": 1.20},
        },
        checkpoint,
    )

    current_state = {key: value.clone() for key, value in historical_state.items()}
    first_key = next(iter(current_state))
    current_state[first_key].view(-1)[0] += 1e-4
    config = exp.Config(tmp_path, tmp_path / "results", core)
    summary = exp._c0_historical_reference_summary(
        config,
        exp.LossSpec("C0_dual_null", 11),
        current_state,
        best_epoch=91,
        best_val={"ba": 0.56, "mean_logit_ce": 1.18},
    )
    assert summary is not None
    assert summary["state_keys_match"] is True
    assert summary["state_hash_match"] is False
    assert summary["best_epoch_match"] is False
    assert first_key in summary["mismatched_parameters"]
    assert summary["val_ba_delta_current_minus_historical"] == pytest.approx(0.01)

    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    payload["identity"] = "wrong-identity"
    torch.save(payload, checkpoint)
    with pytest.raises(ValueError, match="identity mismatch"):
        exp._c0_historical_reference_summary(
            config,
            exp.LossSpec("C0_dual_null", 11),
            current_state,
            best_epoch=91,
            best_val={"ba": 0.56, "mean_logit_ce": 1.18},
        )

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
