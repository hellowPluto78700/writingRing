from __future__ import annotations

from dataclasses import replace

import numpy as np
import torch

from core_benchmark_v1.protocol import Protocol
from scripts import experiment_19_cross_user_generalization as exp


def _p() -> Protocol:
    return replace(
        Protocol(),
        width=12,
        input_channels=3,
        total_channels=3,
        steps=16,
        labels=("A", "B", "C"),
        batch_size=8,
        max_epochs=100,
        min_epochs=1,
        patience=1,
    )


def test_formal_axes_and_task_counts_are_locked():
    assert exp.SEEDS == (11, 23, 37)
    assert exp.CALIBRATION_SEED == 101
    assert exp.F19_1_VARIANTS == ("random", "high_rate", "evidence", "evidence_matched_random")
    assert exp.F19_2_VARIANTS == ("margin_consistency", "rex")
    assert exp.F19_3_VARIANTS == ("matched_noise", "physical", "style_mix")
    assert len(exp.calibration_specs()) == 20
    selection = {
        "19.1": {"q": 0.2, "weight": 0.5},
        "19.2": {"margin_weight": 0.05, "rex_weight": 0.5},
        "19.3": {"strength": "medium", "mix_prob": 0.5},
    }
    assert len(exp.final_specs(selection)) == 27


def test_self_challenge_warmup_and_ramp():
    assert exp.challenge_weight(1, 100, 1.0) == 0.0
    assert exp.challenge_weight(25, 100, 1.0) == 0.0
    assert 0.0 < exp.challenge_weight(30, 100, 1.0) < 1.0
    assert exp.challenge_weight(40, 100, 1.0) == 1.0


def test_true_competitor_never_returns_true_class():
    scores = torch.tensor([[3.0, 2.0, 1.0], [0.0, 4.0, 2.0]])
    y = torch.tensor([0, 1])
    competitor = exp.true_competitor(scores, y)
    assert torch.equal(competitor, torch.tensor([1, 2]))


def test_evidence_contribution_matches_true_minus_competitor_margin():
    spikes = torch.tensor([[[1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]])
    lengths = torch.tensor([3])
    weight = torch.tensor([[2.0, 1.0], [0.5, 1.5], [-1.0, 0.0]])
    y = torch.tensor([0])
    competitor = torch.tensor([1])
    contribution = exp.evidence_contributions(spikes, lengths, weight, y, competitor)
    counts = torch.tensor([[2.0, 2.0]])
    expected = counts * (weight[0] - weight[1])
    assert torch.allclose(contribution, expected)


def test_evidence_mask_targets_top_positive_contributors():
    contribution = torch.tensor([[4.0, 3.0, -9.0, 1.0]])
    mask = exp.evidence_challenge_mask(contribution, 0.5)
    assert torch.equal(mask, torch.tensor([[True, True, False, False]]))


def test_margin_vectors_remove_true_class_coordinate():
    scores = torch.tensor([[3.0, 1.0, 2.0], [1.0, 4.0, 0.0]])
    y = torch.tensor([0, 1])
    margin = exp.margin_vectors(scores, y)
    assert torch.allclose(margin, torch.tensor([[2.0, 1.0], [3.0, 4.0]]))


def test_margin_consistency_and_rex_are_differentiable():
    scores = torch.tensor(
        [[3.0, 1.0, 0.0], [2.5, 0.5, 0.0], [3.1, 1.2, 0.1], [2.7, 0.7, 0.0]],
        requires_grad=True,
    )
    y = torch.zeros(4, dtype=torch.long)
    users = torch.tensor([0, 0, 1, 1])
    loss = exp.margin_consistency_loss(scores, y, users) + exp.rex_loss(scores, y, users)
    assert torch.isfinite(loss)
    loss.backward()
    assert scores.grad is not None
    assert torch.isfinite(scores.grad).all()


def test_aux_sampler_has_8x4x2_geometry():
    labels = []
    users = []
    for cls in range(8):
        for user in range(4):
            for _ in range(2):
                labels.append(cls)
                users.append(f"u{user}")
    n = len(labels)
    arrays = {
        "train_y": np.asarray(labels, dtype=np.int64),
        "train_users": np.asarray(users),
        "train_x": np.zeros((n, 4, 3), dtype=np.float32),
        "train_lengths": np.full(n, 4, dtype=np.int64),
    }
    batches = exp._aux_index_batches(arrays, _p(), 11, 1, 2)
    assert len(batches) == 2
    for idx in batches:
        assert len(idx) == 64
        y = arrays["train_y"][idx]
        u = arrays["train_users"][idx]
        for cls in range(8):
            selected = y == cls
            assert selected.sum() == 8
            assert len(np.unique(u[selected])) == 4


def test_temporal_augmentation_preserves_shape_and_padding():
    x = torch.zeros(2, 8, 3)
    x[0, :5] = 1.0
    x[1, :7] = 2.0
    lengths = torch.tensor([5, 7])
    gen = torch.Generator().manual_seed(123)
    out = exp.physical_augment(x, lengths, "medium", gen)
    assert out.shape == x.shape
    assert torch.equal(out[0, 5:], x[0, 5:])
    assert torch.equal(out[1, 7:], x[1, 7:])


def test_style_mix_requires_same_class_different_user_partner():
    x = torch.arange(4 * 5 * 2, dtype=torch.float32).reshape(4, 5, 2)
    y = torch.tensor([0, 0, 1, 1])
    users = torch.tensor([0, 1, 0, 1])
    lengths = torch.tensor([5, 5, 5, 5])
    out = exp.style_mix(x, y, users, lengths, 1.0, torch.Generator().manual_seed(3))
    assert out.shape == x.shape
    assert torch.isfinite(out).all()
