"""Contracts for the read-only O0 baseline evaluator."""
from __future__ import annotations
from dataclasses import replace
import numpy as np
import torch
from core_benchmark_v1.analysis_tools.baseline_evaluation import SEEDS, _representative_index, confusion, select_classes, trace_one
from core_benchmark_v1.model import BenchmarkNet
from core_benchmark_v1.protocol import Protocol, Run


def smoke_protocol() -> Protocol:
    return replace(Protocol(), profile="smoke", seeds=SEEDS, width=12, input_channels=6,
                   total_channels=6, steps=16, batch_size=8, max_epochs=2, min_epochs=1,
                   patience=1, fixed_ms=250.0)


def test_confusion_is_row_normalized() -> None:
    y = np.array([0, 0, 1, 1, 2, 2])
    pred = np.array([0, 1, 1, 1, 0, 2])
    raw, norm = confusion(y, pred, 3)
    assert raw.tolist() == [[1, 1, 0], [0, 2, 0], [1, 0, 1]]
    assert np.allclose(norm.sum(1), 1)
    assert np.allclose(np.diag(norm), [0.5, 1.0, 0.5])


def test_class_selection_uses_mean_test_recall() -> None:
    labels = tuple("ABCDEFGHIJKL")
    matrices = {}
    for seed, offset in zip(SEEDS, (0.0, 0.01, -0.01)):
        cm = np.zeros((12, 12), dtype=float)
        np.fill_diagonal(cm, np.linspace(0.1, 0.9, 12) + offset)
        matrices[seed] = cm
    selected = select_classes(matrices, labels)
    assert selected["labels"] == {"highest": "L", "middle": "F", "lowest": "A"}


def test_representative_selection_prefers_correct_or_error_and_median_length() -> None:
    arrays = {"test_y": np.array([2, 2, 2, 2, 1]), "test_lengths": np.array([10, 20, 30, 100, 15])}
    pred = np.array([2, 0, 2, 0, 1])
    assert _representative_index(arrays, "test", 2, pred, prefer_error=False) == 0
    assert _representative_index(arrays, "test", 2, pred, prefer_error=True) == 1


def test_single_trace_exports_I_U_spikes_and_accumulator_without_training() -> None:
    p = smoke_protocol()
    run = Run("O0", 11, "01_objective", shifts=((2, 3, 4), (2, 3, 4)), objective="wcce")
    model = BenchmarkNet(run, p)
    x = np.zeros((p.steps, p.input_channels), dtype=np.float32)
    x[:7, 0] = 1
    before = {k: v.detach().clone() for k, v in model.state_dict().items()}
    trace = trace_one(model, x, 7)
    assert trace["input"].shape == (7, p.input_channels)
    for layer in ("L1", "L2"):
        assert trace[f"{layer}_spike"].shape == (7, p.width)
        assert trace[f"{layer}_I"].shape == (7, p.width)
        assert trace[f"{layer}_U"].shape == (7, p.width)
    assert trace["output_evidence"].shape == (7, len(p.labels))
    assert trace["output_accumulator"].shape == (7, len(p.labels))
    assert np.allclose(trace["output_accumulator"][-1], trace["output_evidence"].sum(0))
    assert all(torch.equal(before[k], model.state_dict()[k]) for k in before)