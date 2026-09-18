from __future__ import annotations

from pathlib import Path

import torch

from scripts import experiment_10_2_2_valid_window_probes as exp1022


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_protocol_and_source_contract() -> None:
    assert exp1022.PROTOCOL_VERSION == "exp10_2_1_checkpoints_valid_window_v1"
    assert exp1022.SOURCE_EXPERIMENT_ID == exp1022.exp1021.EXPERIMENT_ID
    assert exp1022.SOURCE_PROTOCOL_VERSION == exp1022.exp1021.PROTOCOL_VERSION
    assert exp1022.EXPECTED_CHECKPOINTS == 18
    assert exp1022.SUPPORTS == ('valid', 'window')
    assert exp1022.LAYERS == ('l1', 'l2')


def test_probe_inventory_has_40_explicit_support_names() -> None:
    names = exp1022.probe_names()
    assert len(names) == 40
    assert len(set(names)) == 40
    assert 'l1__pre_reset__valid__fixed250_ordered_mean' in names
    assert 'l1__pre_reset__window__fixed250_ordered_mean' in names
    assert 'l2__communication__valid__fixed250_count' in names
    assert 'l2__communication__window__fixed250_count' in names
    assert all('__valid__' in name or '__window__' in name for name in names)


def test_valid_vs_window_aggregation_preserves_post_valid_tail() -> None:
    values = torch.tensor([[[1.0], [2.0], [10.0], [20.0]]])
    lengths = torch.tensor([2])

    valid_mean = exp1022._aggregate(
        values, lengths, 'valid', 'whole_mean', bin_steps=2
    )
    window_mean = exp1022._aggregate(
        values, lengths, 'window', 'whole_mean', bin_steps=2
    )
    assert torch.allclose(valid_mean, torch.tensor([[1.5]]))
    assert torch.allclose(window_mean, torch.tensor([[8.25]]))

    valid_fixed_mean = exp1022._aggregate(
        values, lengths, 'valid', 'fixed250_ordered_mean', bin_steps=2
    )
    window_fixed_mean = exp1022._aggregate(
        values, lengths, 'window', 'fixed250_ordered_mean', bin_steps=2
    )
    assert torch.allclose(valid_fixed_mean, torch.tensor([[1.5, 0.0]]))
    assert torch.allclose(window_fixed_mean, torch.tensor([[1.5, 15.0]]))

    valid_count = exp1022._aggregate(
        values, lengths, 'valid', 'whole_count', bin_steps=2
    )
    window_count = exp1022._aggregate(
        values, lengths, 'window', 'whole_count', bin_steps=2
    )
    assert torch.allclose(valid_count, torch.tensor([[3.0]]))
    assert torch.allclose(window_count, torch.tensor([[33.0]]))

    valid_fixed_count = exp1022._aggregate(
        values, lengths, 'valid', 'fixed250_count', bin_steps=2
    )
    window_fixed_count = exp1022._aggregate(
        values, lengths, 'window', 'fixed250_count', bin_steps=2
    )
    assert torch.allclose(valid_fixed_count, torch.tensor([[3.0, 0.0]]))
    assert torch.allclose(window_fixed_count, torch.tensor([[3.0, 30.0]]))


def test_evaluation_path_has_no_snn_training() -> None:
    source = (REPO_ROOT / 'scripts' / 'experiment_10_2_2_valid_window_probes.py').read_text()
    assert 'torch.optim' not in source
    assert '.backward(' not in source
    assert 'optimizer.step' not in source
    assert 'model.load_state_dict' in source
    assert 'checkpoint.get("experiment_id")' in source
    assert 'checkpoint.get("protocol_version")' in source
    assert 'checkpoint.get("split_sample_hashes")' in source


def test_probe_selection_stays_validation_ba() -> None:
    source = (REPO_ROOT / 'scripts' / 'experiment_10_2_2_valid_window_probes.py').read_text()
    assert 'val_metrics["balanced_accuracy"]' in source
    assert 'val_metrics["macro_f1"]' in source
    assert 'test_macro_f1' in source
    assert 'test_balanced_accuracy' in source
    assert 'window_minus_valid' in source


def test_slurm_is_evaluation_only_18_way_array() -> None:
    root = REPO_ROOT / 'scripts' / 'bash_script' / 'SNN_Bash'
    run = (root / 'run_exp_10_2_2_cpu_array.bash').read_text()
    submit = (root / 'submit_exp_10_2_2_cpu.bash').read_text()
    assert '#SBATCH --array=0-17%18' in run
    assert '#SBATCH --cpus-per-task=1' in run
    assert 'evaluate-one' in run
    assert 'run-one' not in run
    assert 'afterok:${prepare_job}' in submit
    assert 'afterok:${array_job}' in submit
    for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        assert f'export {name}=1' in run


def test_notebook_is_aggregation_only() -> None:
    notebook = (REPO_ROOT / 'notebooks' / 'experiment_10_2_2_valid_window_probes.ipynb').read_text()
    assert 'support_contrast_summary.csv' in notebook
    assert 'key_probe_summary.csv' in notebook
    assert 'test_macro_f1_valid' in notebook
    assert 'test_balanced_accuracy_window' in notebook
    assert 'evaluate_one(' not in notebook
    assert 'forward_trajectory' not in notebook
