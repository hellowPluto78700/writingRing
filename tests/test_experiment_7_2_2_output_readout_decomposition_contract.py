from __future__ import annotations

import ast
import math
from pathlib import Path

import numpy as np

from scripts import experiment_7_2_2_output_readout_decomposition as exp722
from scripts import experiment_7_2_two_layer_tau_training as exp72


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_extension_reuses_exact_exp72_run_grid() -> None:
    specs = exp722.run_specs()
    assert len(specs) == 84
    assert [s.key for s in specs] == [s.key for s in exp72.run_specs()]


def test_extension_never_trains_the_snn_backbone() -> None:
    source = (REPO_ROOT / "scripts/experiment_7_2_2_output_readout_decomposition.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    top_level_names = {
        node.name
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    assert "train_one" not in top_level_names
    assert "torch.optim" not in source
    assert "backward()" not in source
    assert "snn_backbone_retrained" in source
    assert "exp72.load_model" in source


def test_whole_count_masks_padding() -> None:
    l2 = np.zeros((2, 5, 3), dtype=np.float32)
    l2[0, :, 0] = 1.0
    l2[1, :, 1] = 2.0
    lengths = np.array([2, 4])
    got = exp722._whole_count(l2, lengths)
    expected = np.array([[2.0, 0.0, 0.0], [0.0, 8.0, 0.0]], dtype=np.float32)
    assert np.array_equal(got, expected)


def test_analog_streaming_identity() -> None:
    rng = np.random.default_rng(3)
    l2 = rng.integers(0, 2, size=(4, 7, 5)).astype(np.float32)
    lengths = np.array([7, 5, 4, 6])
    W = rng.normal(size=(3, 5))
    b = rng.normal(size=3)
    got = exp722._analog_scores(l2, lengths, W, b)
    expected = exp722._whole_count(l2, lengths) @ W.T + b[None, :]
    assert np.allclose(got, expected)


def test_lif_forward_matches_macro_update_order() -> None:
    # One sample, one hidden feature, one output. Evidence is +0.3 each step.
    l2 = np.ones((1, 3, 1), dtype=np.float32)
    lengths = np.array([3])
    W = np.array([[0.3]], dtype=float)
    b = np.array([0.0])
    scores, diag = exp722._simulate_lif(
        l2, lengths, W, b, beta=0.5, threshold=0.5, cap=1, fs=64.0
    )
    # U: 0.3 -> 0.45 -> 0.525, one spike at the last step.
    assert np.array_equal(scores, np.array([[1.0]]))
    assert math.isclose(diag["mean_total_output_events_per_sample"], 1.0)


def test_if_beta1_removes_leak_but_keeps_threshold_reset() -> None:
    l2 = np.ones((1, 3, 1), dtype=np.float32)
    lengths = np.array([3])
    W = np.array([[0.2]], dtype=float)
    b = np.array([0.0])
    lif, _ = exp722._simulate_lif(l2, lengths, W, b, beta=0.5, threshold=0.5, cap=1, fs=64.0)
    iff, _ = exp722._simulate_lif(l2, lengths, W, b, beta=1.0, threshold=0.5, cap=1, fs=64.0)
    assert np.array_equal(lif, np.array([[0.0]]))
    assert np.array_equal(iff, np.array([[1.0]]))


def test_multispike_cap_preserves_multiple_crossings() -> None:
    l2 = np.ones((1, 1, 1), dtype=np.float32)
    lengths = np.array([1])
    W = np.array([[1.6]], dtype=float)
    b = np.array([0.0])
    cap1, _ = exp722._simulate_lif(l2, lengths, W, b, beta=0.5, threshold=0.5, cap=1, fs=64.0)
    cap31, _ = exp722._simulate_lif(l2, lengths, W, b, beta=0.5, threshold=0.5, cap=31, fs=64.0)
    assert np.array_equal(cap1, np.array([[1.0]]))
    assert np.array_equal(cap31, np.array([[3.0]]))


def test_bipolar_readout_can_emit_negative_class_evidence() -> None:
    l2 = np.ones((1, 1, 1), dtype=np.float32)
    lengths = np.array([1])
    W = np.array([[-0.6]], dtype=float)
    b = np.array([0.0])
    score, diag = exp722._simulate_bipolar_lif(
        l2, lengths, W, b, beta=0.5, threshold=0.5, cap=1, fs=64.0
    )
    assert np.array_equal(score, np.array([[-1.0]]))
    assert math.isclose(diag["negative_event_fraction"], 1.0)


def test_current_output_lif_constants_are_preserved() -> None:
    beta = math.exp(-(1000.0 / 64.0) / exp72.OUTPUT_TAU_MEM_MS)
    assert np.isclose(beta, 0.5, atol=2e-4)
    assert exp722.THRESHOLD == 0.5
    assert exp722.MULTISPIKE_CAP == 31
    assert exp722.SOURCE_NATIVE_LIF == "native_e2e_lif"
    assert exp722.SOURCE_NATIVE_ANALOG == "native_w_analog_sum"


def test_slurm_uses_84_one_cpu_workers_with_50_way_cap() -> None:
    worker = (REPO_ROOT / "scripts/bash_script/SNN_Bash/run_exp_7_2_2_cpu_array.bash").read_text(encoding="utf-8")
    finalizer = (REPO_ROOT / "scripts/bash_script/SNN_Bash/finalize_exp_7_2_2_cpu.bash").read_text(encoding="utf-8")
    submit = (REPO_ROOT / "scripts/bash_script/SNN_Bash/submit_exp_7_2_2_cpu.bash").read_text(encoding="utf-8")
    assert "#SBATCH --array=0-83%50" in worker
    assert "#SBATCH --cpus-per-task=1" in worker
    assert "run-one" in worker
    assert "finalize" in finalizer
    assert 'afterok:${worker_job}' in submit
    for text in (worker, finalizer):
        for token in (
            "OMP_NUM_THREADS=1",
            "MKL_NUM_THREADS=1",
            "OPENBLAS_NUM_THREADS=1",
            "NUMEXPR_NUM_THREADS=1",
        ):
            assert token in text


def test_notebook_is_aggregate_only() -> None:
    text = (REPO_ROOT / "notebooks/experiment_7_2_2_output_readout_decomposition.ipynb").read_text(encoding="utf-8")
    for token in (
        "architecture_table.csv",
        "readout_summary.csv",
        "paired_deltas.csv",
        "diagnostics_summary.csv",
    ):
        assert token in text
    for token in (
        "readout_runs.csv",
        "diagnostics_runs.csv",
        "paired_delta_runs.csv",
        "probe_parameters_runs.csv",
        "runs/",
        "checkpoints/",
        "torch.optim",
        "sbatch",
    ):
        assert token not in text


def test_source_file_parses() -> None:
    source = REPO_ROOT / "scripts/experiment_7_2_2_output_readout_decomposition.py"
    ast.parse(source.read_text(encoding="utf-8"))
