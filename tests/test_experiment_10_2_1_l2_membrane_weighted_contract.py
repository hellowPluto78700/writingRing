from __future__ import annotations

import math
from pathlib import Path

import torch

from scripts import experiment_10_2_1_l2_membrane_weighted as exp1021


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_factorial_contract() -> None:
    assert exp1021.PROTOCOL_VERSION == "d1_l1mem2_l2mem123_binary_weighted31_v1"
    assert exp1021.VARIANT == "postencode_mask"
    assert exp1021.ROTATION == 0
    assert exp1021.L1_MEM_SHIFT == 2
    assert exp1021.L2_MEM_SHIFTS == (1, 2, 3)
    assert exp1021.CODINGS == ("binary", "weighted31")
    assert exp1021.MODEL_SEEDS == (11, 23, 37)
    assert exp1021.EXPECTED_RUNS == 18
    specs = exp1021.run_specs()
    assert len(specs) == 18
    assert len({spec.key for spec in specs}) == 18


def test_membrane_mapping() -> None:
    assert math.isclose(exp1021.beta_from_mem_shift(1), math.exp(-(1000.0 / 64.0) / exp1021.exp72.TAU_MEM_MS))
    assert math.isclose(exp1021.beta_from_mem_shift(2), 0.75)
    assert math.isclose(exp1021.beta_from_mem_shift(3), 0.875)
    assert math.isclose(exp1021.tau_mem_ms_from_shift(1), exp1021.exp72.TAU_MEM_MS)
    assert math.isclose(exp1021.tau_mem_ms_from_shift(2), 54.31342963722199, rel_tol=1e-9)
    assert math.isclose(exp1021.tau_mem_ms_from_shift(3), 117.0136826471659, rel_tol=1e-9)


def test_binary_and_weighted_change_only_event_cap_and_l2_beta() -> None:
    binary = exp1021.Exp1021Net(exp1021.RunSpec('binary', 1, 11), 12, 64.0)
    weighted = exp1021.Exp1021Net(exp1021.RunSpec('weighted31', 1, 11), 12, 64.0)
    long_l2 = exp1021.Exp1021Net(exp1021.RunSpec('binary', 3, 11), 12, 64.0)

    assert binary.l1_lif.max_spikes_per_dt == 1
    assert binary.l2_lif.max_spikes_per_dt == 1
    assert weighted.l1_lif.max_spikes_per_dt == 31
    assert weighted.l2_lif.max_spikes_per_dt == 31
    assert math.isclose(binary.l1_lif.beta, 0.75)
    assert math.isclose(weighted.l1_lif.beta, 0.75)
    assert math.isclose(binary.l2_lif.beta, exp1021.beta_from_mem_shift(1))
    assert math.isclose(long_l2.l2_lif.beta, 0.875)
    assert torch.equal(binary.alpha_0, weighted.alpha_0)
    assert torch.equal(binary.alpha_1, weighted.alpha_1)


def test_weighted_count_semantics() -> None:
    lif = exp1021.exp401.MacroMultiSpikeLIF(
        beta=0.75,
        threshold=1.0,
        max_spikes_per_dt=31,
        surrogate_slope=exp1021.exp72.SURROGATE_SLOPE,
    )
    current = torch.tensor([[0.8, 1.4, 2.7, 31.9]])
    membrane = torch.zeros_like(current)
    spikes, post, pre = lif(current, membrane)
    assert torch.equal(spikes, torch.tensor([[0.0, 1.0, 2.0, 31.0]]))
    assert torch.allclose(pre, current)
    assert torch.allclose(post, torch.tensor([[0.8, 0.4, 0.7, 0.9]]), atol=1e-6)


def test_binary_l2shift1_exactly_matches_exp102_shift2_baseline() -> None:
    seed = exp1021.exp73._e2e_pair_seed(11, 'model_init')
    exp1021.exp3.seed_all(seed)
    reference = exp1021.exp102.Exp102Net(2, n_classes=12, fs=64.0)
    exp1021.exp3.seed_all(seed)
    candidate = exp1021.Exp1021Net(
        exp1021.RunSpec('binary', 1, 11), n_classes=12, fs=64.0
    )
    assert set(reference.state_dict()) == set(candidate.state_dict())
    for name, value in reference.state_dict().items():
        assert torch.equal(value, candidate.state_dict()[name]), name

    x = torch.randn(2, 9, 30, generator=torch.Generator().manual_seed(123))
    with torch.no_grad():
        ref = reference.forward_trajectory(x)
        got = candidate.forward_trajectory(x)
    assert torch.equal(ref['l2_evidence'], got['l2_evidence'])
    for layer in ('l1', 'l2'):
        for state in ('syn_current', 'pre_reset', 'spike', 'post_reset'):
            assert torch.equal(ref['hidden'][layer][state], got['hidden'][layer][state]), (layer, state)


def test_paired_model_seed_excludes_coding_and_l2_mem() -> None:
    source = (REPO_ROOT / 'scripts' / 'experiment_10_2_1_l2_membrane_weighted.py').read_text()
    block = source[source.index('def run_one'):source.index('def _probe_value')]
    assert 'exp73._e2e_pair_seed(spec.seed, "model_init")' in block
    assert 'spec.coding, "model_init"' not in block
    assert 'spec.l2_mem_shift, "model_init"' not in block


def test_single_split_is_rotation0() -> None:
    roles = exp1021.exp90._rotation_fold_roles(exp1021.ROTATION)
    assert roles[0] == 'test'
    assert roles[1] == 'val'
    assert all(roles[fold] == 'train' for fold in (2, 3, 4))


def test_activity_contract_contains_weighted_cost_metrics() -> None:
    source = (REPO_ROOT / 'scripts' / 'experiment_10_2_1_l2_membrane_weighted.py').read_text()
    for token in (
        'multi_event_fraction_ge_2',
        'large_event_fraction_ge_4',
        'cap_hit_fraction',
        'mean_events_per_neuron_s',
        'dead_neuron_fraction',
    ):
        assert token in source


def test_slurm_contract() -> None:
    root = REPO_ROOT / 'scripts' / 'bash_script' / 'SNN_Bash'
    run = (root / 'run_exp_10_2_1_cpu_array.bash').read_text()
    submit = (root / 'submit_exp_10_2_1_cpu.bash').read_text()
    assert '#SBATCH --array=0-17%18' in run
    assert '#SBATCH --cpus-per-task=1' in run
    for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        assert f'export {name}=1' in run
    assert 'afterok:${prepare_job}' in submit
    assert 'afterok:${array_job}' in submit


def test_notebook_is_aggregation_only() -> None:
    notebook = (REPO_ROOT / 'notebooks' / 'experiment_10_2_1_l2_membrane_weighted.ipynb').read_text()
    assert 'd1_l1mem2_l2mem123_binary_weighted31_v1' in notebook
    assert 'method_summary.csv' in notebook
    assert 'interaction_summary.csv' in notebook
    assert 'activity_summary.csv' in notebook
    assert 'probe_summary.csv' in notebook
    assert 'run_one(' not in notebook
