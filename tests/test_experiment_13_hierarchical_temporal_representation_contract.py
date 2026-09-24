from pathlib import Path

import numpy as np
import torch

from scripts import experiment_13_hierarchical_temporal_representation as exp13


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_exp13_protocol_and_task_counts() -> None:
    assert exp13.PROTOCOL_VERSION == 'hierarchical_temporal_representation_v2'
    assert exp13.SEEDS == (11, 23, 37)
    assert exp13.HISTORY_MS == (50, 100, 250, 500, 750, 1000)
    assert exp13.PHASES == (0.25, 0.50, 0.75, 1.00)
    assert exp13.RESET_CASES == ('intact', 'reset_l1', 'reset_l2', 'reset_both')
    assert exp13.SCRAMBLE_GAP_MS == (0, 50, 100, 250)
    assert exp13.SCRAMBLE_SCALES == ('stroke', 'multi_stroke')
    assert len(exp13.a1_specs()) == 9
    assert len(exp13.a2_specs()) == 36
    assert len(exp13.a3_specs()) == 12
    assert len(exp13.b_specs()) == 36
    assert len(exp13.c_specs()) == 3
    assert len(exp13.d_specs()) == 24
    assert len(exp13.e_specs()) == 9


def test_exp13_subexperiments_have_independent_result_roots(tmp_path) -> None:
    config = exp13.Config(repo_root=tmp_path, results_dir=tmp_path / 'results')
    roots = [exp13._subdir(config, name) for name in exp13.SUBEXPERIMENTS]
    assert len(roots) == len(set(roots)) == 7
    assert all(path.parent == config.results_dir for path in roots)


def test_spike50_is_trailing_four_step_sum() -> None:
    values = torch.arange(1, 7, dtype=torch.float32).reshape(1, 6, 1)
    actual = exp13._rolling_sum(values, 4).reshape(-1).tolist()
    assert actual == [1.0, 3.0, 6.0, 10.0, 14.0, 18.0]


def test_suffix_builder_keeps_identical_recent_history_only() -> None:
    X = np.zeros((2, 8, 2), dtype=np.float32)
    X[0, :, 0] = np.arange(8, dtype=np.float32)
    X[1, :, 0] = 100 + np.arange(8, dtype=np.float32)
    suffix, lengths = exp13._suffix_batch_for_steps(
        X,
        np.asarray([0, 1]),
        np.asarray([4, 2]),
        history_steps=3,
    )
    np.testing.assert_array_equal(lengths, np.asarray([3, 3]))
    np.testing.assert_array_equal(suffix[0, :, 0], np.asarray([2, 3, 4]))
    np.testing.assert_array_equal(suffix[1, :, 0], np.asarray([100, 101, 102]))


def test_reset_case_contract_zeroes_expected_layers() -> None:
    assert exp13._reset_layer_indices('intact') == ()
    assert exp13._reset_layer_indices('reset_l1') == (0,)
    assert exp13._reset_layer_indices('reset_l2') == (1,)
    assert exp13._reset_layer_indices('reset_both') == (0, 1)


def test_scramble_preserves_stroke_waveforms_and_gap() -> None:
    strokes = [
        np.asarray([[1.0], [2.0]], dtype=np.float32),
        np.asarray([[3.0]], dtype=np.float32),
        np.asarray([[4.0], [5.0]], dtype=np.float32),
    ]
    assembled, positions = exp13._assemble_strokes(strokes, [1, 0, 2], gap_steps=2)
    np.testing.assert_array_equal(
        assembled.reshape(-1),
        np.asarray([3.0, 0.0, 0.0, 1.0, 2.0, 0.0, 0.0, 4.0, 5.0]),
    )
    assert positions[1] == (0, 1)
    assert positions[0] == (3, 5)
    assert positions[2] == (7, 9)


def test_no_bias_probe_has_no_centering_or_intercept() -> None:
    train_x = np.asarray([[0.0], [1.0], [3.0], [4.0]], dtype=np.float64)
    train_y = np.asarray([0, 0, 1, 1], dtype=np.int64)
    scaler, classifier, selected_c = exp13._fit_probe_model(
        train_x,
        train_y,
        train_x,
        train_y,
        seed=11,
        bias_mode='no_bias',
    )
    assert scaler.with_mean is False
    assert classifier.fit_intercept is False
    assert selected_c in exp13.PROBE_C_GRID
    np.testing.assert_array_equal(scaler.transform(np.zeros((1, 1))), np.zeros((1, 1)))


def test_submit_chain_serializes_subexperiments_and_caps_concurrency() -> None:
    submit = (
        REPO_ROOT / 'scripts' / 'bash_script' / 'SNN_Bash' / 'submit_exp_13_cpu.bash'
    ).read_text(encoding='utf-8')
    assert 'SLURM_MAX_CONCURRENCY:-20' in submit
    assert 'SLURM_MAX_CONCURRENCY > 50' in submit
    assert '--array="0-8%${a1_conc}"' in submit
    assert '--array="0-35%${a2_conc}"' in submit
    assert '--array="0-11%${a3_conc}"' in submit
    assert '--array="0-35%${b_conc}"' in submit
    assert '--array="0-2%${c_conc}"' in submit
    assert '--array="0-23%${d_conc}"' in submit
    assert '--array="0-8%${e_conc}"' in submit
    assert 'afterok:${a1_job}' in submit
    assert 'afterok:${a2_job}' in submit
    assert 'afterok:${a3_job}' in submit
    assert 'afterok:${b_job}' in submit
    assert 'afterok:${c_job}' in submit
    assert 'afterok:${d_job}' in submit
    assert 'afterok:${e_job}' in submit


def test_notebook_is_analysis_only() -> None:
    text = (
        REPO_ROOT / 'notebooks' / 'experiment_13_hierarchical_temporal_representation.ipynb'
    ).read_text(encoding='utf-8')
    for token in (
        'A1_recurrence_summary.csv',
        'A2_history_truncation_summary.csv',
        'A3_layer_reset_summary.csv',
        'B_context_expression_summary.csv',
        'C_boundary_summary.csv',
        'D_stroke_scrambling_summary.csv',
        'E_cross_tau_decoding_summary.csv',
    ):
        assert token in text
    assert 'import torch' not in text
    assert 'sbatch' not in text
    assert 'subprocess' not in text
