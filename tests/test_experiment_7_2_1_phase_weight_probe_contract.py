from __future__ import annotations

import ast
from pathlib import Path

import numpy as np

from scripts import experiment_7_2_1_phase_weight_probe as exp721
from scripts import experiment_7_2_two_layer_tau_training as exp72

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_extension_reuses_exact_exp72_run_grid() -> None:
    specs = exp721.run_specs()
    assert len(specs) == 84
    assert [s.key for s in specs] == [s.key for s in exp72.run_specs()]


def test_extension_has_no_snn_training_entrypoint() -> None:
    source = (REPO_ROOT / "scripts/experiment_7_2_1_phase_weight_probe.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    names = {node.name for node in tree.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))}
    assert "train_one" not in names
    assert "torch.optim" not in source
    assert "optimizer" not in source.lower()
    assert "load_model" in source
    assert "snn_retrained" in source


def test_wholecount_is_exact_sum_over_fixed250_bins() -> None:
    x = np.arange(2 * 16 * 128, dtype=np.float32).reshape(2, 16, 128)
    got = exp721._whole_count_features(x)
    assert got.shape == (2, 128)
    assert np.array_equal(got, x.sum(axis=1))


def test_phase_shuffle_preserves_per_sample_bin_multiset_and_total_count() -> None:
    x = np.zeros((3, 16, 2), dtype=np.float32)
    for i in range(3):
        for b in range(16):
            x[i, b] = (100 * i + b, 1000 * i + b)
    shuffled = exp721._phase_shuffle(x, seed=123)
    assert shuffled.shape == x.shape
    assert np.array_equal(shuffled.sum(axis=1), x.sum(axis=1))
    for i in range(3):
        before = sorted(map(tuple, x[i].tolist()))
        after = sorted(map(tuple, shuffled[i].tolist()))
        assert before == after
    assert not np.array_equal(shuffled, x)


def test_probe_contract_has_matched_capacity_phase_control() -> None:
    assert exp721.SHUFfLE_REPEATS == 3 if False else True
    assert exp721.SHUFFLE_REPEATS == 3
    assert exp721.SOURCE_WHOLE == "l2_wholecount_shared"
    assert exp721.SOURCE_ORDERED == "l2_fixed250_ordered"
    assert exp721.SOURCE_SHUFFLED == "l2_fixed250_phase_shuffled"


def test_slurm_is_frozen_checkpoint_cpu_array_with_50_way_cap() -> None:
    worker = (REPO_ROOT / "scripts/bash_script/SNN_Bash/run_exp_7_2_1_phase_probe_cpu_array.bash").read_text(encoding="utf-8")
    finalizer = (REPO_ROOT / "scripts/bash_script/SNN_Bash/finalize_exp_7_2_1_phase_probe_cpu.bash").read_text(encoding="utf-8")
    submit = (REPO_ROOT / "scripts/bash_script/SNN_Bash/submit_exp_7_2_1_phase_probe_cpu.bash").read_text(encoding="utf-8")
    assert "#SBATCH --array=0-83%50" in worker
    assert "#SBATCH --cpus-per-task=1" in worker
    assert "run-one" in worker
    assert "finalize" in finalizer
    assert 'afterok:${worker_job}' in submit
    for text in (worker, finalizer):
        for token in ("OMP_NUM_THREADS=1", "MKL_NUM_THREADS=1", "OPENBLAS_NUM_THREADS=1", "NUMEXPR_NUM_THREADS=1"):
            assert token in text


def test_notebook_reads_aggregate_extension_results_only() -> None:
    text = (REPO_ROOT / "notebooks/experiment_7_2_1_phase_weight_probe.ipynb").read_text(encoding="utf-8")
    for token in ("architecture_table.csv", "probe_summary.csv", "paired_deltas.csv"):
        assert token in text
    for token in ("probe_runs.csv", "paired_delta_runs.csv", "runs/", "checkpoints/", "torch.optim", "sbatch"):
        assert token not in text
