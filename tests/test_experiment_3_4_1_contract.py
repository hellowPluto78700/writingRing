from __future__ import annotations

from pathlib import Path

import numpy as np

from scripts import experiment_3_4_1_raw_causal_temporal_readout as exp341


def test_protocol_run_matrix_and_dimensions() -> None:
    assert exp341.SPLIT_SEEDS == (11, 23, 37, 53, 71)
    assert exp341.EVENT_CHANNEL_COUNT == 30
    assert exp341.EXPECTED_SAMPLING_RATE_HZ == 64.0
    assert exp341.EXPECTED_RUNS == 25
    specs = exp341.run_specs()
    assert len(specs) == 25
    assert len({spec.key for spec in specs}) == 25
    assert [spec.key for spec in specs[:5]] == [f"relative10__whole__split{s}" for s in exp341.SPLIT_SEEDS]
    assert [spec.key for spec in specs[5:10]] == [f"fixed250__whole__split{s}" for s in exp341.SPLIT_SEEDS]
    assert [spec.key for spec in specs[10:15]] == [f"fixed500__whole__split{s}" for s in exp341.SPLIT_SEEDS]
    assert [spec.key for spec in specs[15:20]] == [f"fixed250__prefix__split{s}" for s in exp341.SPLIT_SEEDS]
    assert [spec.key for spec in specs[20:25]] == [f"fixed500__prefix__split{s}" for s in exp341.SPLIT_SEEDS]


def test_expected_feature_dimensions() -> None:
    assert exp341.RELATIVE_N_BINS * exp341.EVENT_CHANNEL_COUNT == 300
    assert (exp341.GLOBAL_PADDED_LENGTH // 16) * exp341.EVENT_CHANNEL_COUNT == 480
    assert (exp341.GLOBAL_PADDED_LENGTH // 32) * exp341.EVENT_CHANNEL_COUNT == 240


def test_prefix_future_bins_are_zero_and_final_prefix_equals_whole() -> None:
    counts = np.arange(4 * 3, dtype=np.float64).reshape(4, 3)
    p2 = exp341.raw_prefix_feature(counts, 2).reshape(4, 3)
    assert np.array_equal(p2[:2], counts[:2])
    assert np.array_equal(p2[2:], np.zeros((2, 3)))
    final = exp341.raw_prefix_feature(counts, 4)
    assert np.array_equal(final, counts.reshape(-1))


def test_prefix_causal_invariance() -> None:
    rng = np.random.default_rng(7)
    counts = rng.normal(size=(8, 5))
    changed = counts.copy()
    changed[3:] = rng.normal(size=changed[3:].shape) * 1000.0
    before = exp341.raw_prefix_feature(counts, 3)
    after = exp341.raw_prefix_feature(changed, 3)
    assert np.array_equal(before, after)


def test_prefix_weights_sum_to_one_per_original_sample_and_endpoint_is_half() -> None:
    counts = np.zeros((3, 4, 2), dtype=np.float32)
    mask = np.array([
        [1, 0, 0, 0],
        [1, 1, 1, 0],
        [1, 1, 1, 1],
    ], dtype=bool)
    labels = np.array([0, 1, 2])
    _, _, weights, source_rows, prefix_indices = exp341.build_prefix_examples(counts, mask, labels)
    totals = np.bincount(source_rows, weights=weights, minlength=3)
    assert np.allclose(totals, 1.0, atol=1e-12, rtol=0.0)
    first = weights[source_rows == 0]
    assert np.array_equal(first, np.array([1.0]))
    for row, k_i in ((1, 3), (2, 4)):
        row_weights = weights[source_rows == row]
        row_prefixes = prefix_indices[source_rows == row]
        assert np.isclose(row_weights[row_prefixes == k_i][0], 0.5)
        assert np.isclose(row_weights[row_prefixes < k_i].sum(), 0.5)


def test_scaler_is_fit_on_whole_training_samples_only() -> None:
    whole = np.array([[0.0, 2.0], [2.0, 4.0], [4.0, 6.0]])
    mean, std = exp341.fit_whole_scaler(whole)
    assert np.allclose(mean, [2.0, 4.0])
    assert np.all(std > 0)
    prefix_raw = np.array([[0.0, 0.0]])
    transformed = exp341.apply_scaler(prefix_raw, mean, std)
    assert transformed.shape == (1, 2)
    assert not np.array_equal(transformed, prefix_raw)


def test_partial_final_bin_contract_comes_from_exp32() -> None:
    lengths = np.array([1, 16, 17, 32, 33, 40, 256])
    valid250 = np.ceil(lengths / 16).astype(int)
    valid500 = np.ceil(lengths / 32).astype(int)
    assert valid250.tolist() == [1, 1, 2, 2, 3, 3, 16]
    assert valid500.tolist() == [1, 1, 1, 1, 2, 2, 8]


def test_slurm_array_is_one_run_per_cpu_and_finalizer_is_dependency_only() -> None:
    root = Path(__file__).resolve().parents[1]
    runner = (root / "scripts/bash_script/SNN_Bash/run_exp_3_4_1_cpu_array.bash").read_text()
    finalizer = (root / "scripts/bash_script/SNN_Bash/finalize_exp_3_4_1_cpu.bash").read_text()
    submitter = (root / "scripts/bash_script/SNN_Bash/submit_exp_3_4_1_pipeline.bash").read_text()
    assert "#SBATCH --array=0-24%25" in runner
    assert "#SBATCH --cpus-per-task=1" in runner
    for variable in ("OMP_NUM_THREADS=1", "MKL_NUM_THREADS=1", "OPENBLAS_NUM_THREADS=1", "NUMEXPR_NUM_THREADS=1"):
        assert variable in runner
    assert 'run-one --array-task-id "$TASK_ID"' in runner
    assert "experiment_3_4_1_raw_causal_temporal_readout finalize" in finalizer
    assert "--dependency=afterok:${EXP341_ARRAY}" in submitter
