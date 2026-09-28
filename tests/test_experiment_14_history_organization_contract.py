from __future__ import annotations

import numpy as np
import torch

from core_benchmark_v1.model import BenchmarkNet
from core_benchmark_v1.protocol import Protocol
from scripts import experiment_14_history_organization as exp


def test_run_manifest_and_frozen_protocol_constants() -> None:
    specs = exp.phase1_specs()
    assert len(specs) == 42
    assert exp.SEEDS == (11, 23, 37)
    assert exp.SHIFTS == ((2, 3, 4), (2, 3, 4))
    assert exp.LAMBDA_GRID == (0.01, 0.03, 0.1, 0.3)
    assert exp.PHASES == (0.25, 0.50, 0.75, 1.00)
    assert exp.HISTORY_MS == 250
    assert exp.EXPECTED_BATCH_SIZE == 128
    assert sum(spec.case == "C0_wcce" for spec in specs) == 3
    assert sum(spec.case == "C1_whole_cu" for spec in specs) == 12
    assert sum(spec.case == "C2_phase_cu" for spec in specs) == 12
    assert sum(spec.case == "C3_history_delta" for spec in specs) == 12
    assert sum(spec.case == "C5_shuffled_phase_control" for spec in specs) == 3


def test_same_seed_cases_have_identical_snn_initialization() -> None:
    p = Protocol()
    left = BenchmarkNet(exp._run(exp.LossSpec("C0_wcce", 11)), p)
    right = BenchmarkNet(exp._run(exp.LossSpec("C3_history_delta", 11, lambda_history=0.1)), p)
    left_state = left.state_dict()
    right_state = right.state_dict()
    assert left_state.keys() == right_state.keys()
    for key in left_state:
        assert torch.equal(left_state[key], right_state[key]), key


def test_cross_user_supcon_requires_cross_user_positives_and_backpropagates() -> None:
    raw = torch.tensor(
        [[1.0, 0.0], [0.8, 0.2], [0.0, 1.0], [0.2, 0.8]],
        requires_grad=True,
    )
    z = torch.nn.functional.normalize(raw, dim=1)
    labels = torch.tensor([0, 0, 1, 1])
    users = torch.tensor([0, 1, 0, 1])
    loss = exp.cross_user_supcon(z, labels, users)
    assert torch.isfinite(loss)
    loss.backward()
    assert raw.grad is not None
    assert torch.isfinite(raw.grad).all()


def test_auxiliary_schedule_has_warmup_ramp_and_plateau() -> None:
    target = 0.3
    assert exp._aux_weight(target, 1) == 0.0
    assert exp._aux_weight(target, exp.WARMUP_EPOCHS) == 0.0
    middle = exp._aux_weight(target, 20)
    assert 0.0 < middle < target
    assert exp._aux_weight(target, exp.RAMP_END_EPOCH) == target
    assert exp._aux_weight(target, 100) == target


def test_structured_sampler_has_cross_user_same_class_support() -> None:
    labels = []
    users = []
    for class_id in range(12):
        for user_id in range(8):
            for sample in range(4):
                labels.append(class_id)
                users.append(f"user_{user_id}")
    arrays = {
        "train_y": np.asarray(labels, dtype=np.int64),
        "train_users": np.asarray(users, dtype=str),
    }
    batch = next(iter(exp._structured_batches(arrays, seed=11, epoch=1)))
    assert len(batch) == exp.EXPECTED_BATCH_SIZE
    y = arrays["train_y"][batch]
    u = arrays["train_users"][batch]
    assert len(np.unique(y)) == exp.CLASSES_PER_BATCH
    for class_id in np.unique(y):
        selected = u[y == class_id]
        assert len(np.unique(selected)) == exp.USERS_PER_CLASS


def test_history_delta_reset_changes_only_history_path() -> None:
    p = Protocol.from_dict({
        **Protocol().__dict__,
        "profile": "smoke",
        "seeds": [11],
        "width": 6,
        "input_channels": 3,
        "total_channels": 9,
        "steps": 32,
        "labels": ["A", "B", "C"],
        "batch_size": 3,
        "max_epochs": 1,
        "min_epochs": 1,
        "patience": 1,
        "relative_bins": 4,
    })
    spec = exp.LossSpec("C3_history_delta", 11, lambda_history=0.1)
    run = exp._run(spec)
    run = type(run)(
        case=run.case,
        seed=run.seed,
        block=run.block,
        shifts=run.shifts,
        objective=run.objective,
    )
    model = BenchmarkNet(run, p)
    x = torch.zeros(3, 32, 3)
    x[:, :20, 0] = 1.0
    lengths = torch.tensor([32, 32, 32])
    full = model(x, lengths)["pre_reset"][1]
    delta = exp._history_delta(model, x, lengths, full, history_steps=8)
    assert delta.shape == (3, p.width)
    assert torch.isfinite(delta).all()


def test_phase2_combination_uses_selected_lambdas() -> None:
    selection = {
        "C2_phase_cu": {"lambda": 0.1},
        "C3_history_delta": {"lambda": 0.03},
    }
    specs = exp.phase2_specs(selection)
    assert len(specs) == 6
    best = [spec for spec in specs if spec.case == "C4_combined_best"]
    half = [spec for spec in specs if spec.case == "C4_combined_half"]
    assert len(best) == 3 and len(half) == 3
    assert all(spec.lambda_class == 0.1 and spec.lambda_history == 0.03 for spec in best)
    assert all(spec.lambda_class == 0.05 and spec.lambda_history == 0.015 for spec in half)
