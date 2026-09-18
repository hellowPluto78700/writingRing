from __future__ import annotations

import math
from pathlib import Path

import torch

from scripts import experiment_10_2_l1_membrane_memory as exp102


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_exp102_factorial_contract() -> None:
    assert exp102.PROTOCOL_VERSION == "d1_bb_l1_mem_shift_sweep_v1"
    assert exp102.VARIANT == "postencode_mask"
    assert exp102.CODING == "bb"
    assert exp102.OBJECTIVE == "l2_wcce"
    assert exp102.ROTATION == 0
    assert exp102.L1_MEM_SHIFTS == (1, 2, 3, 4)
    assert exp102.L2_MEM_SHIFT == 1
    assert exp102.MODEL_SEEDS == (11, 23, 37)
    assert len(exp102.baseline_specs()) == 3
    assert len(exp102.long_e2e_specs()) == 9
    assert len(exp102.e2e_specs()) == 12
    assert len(exp102.replay_specs()) == 12


def test_membrane_shift_mapping() -> None:
    expected_beta = {
        1: math.exp(-(1000.0 / 64.0) / exp102.exp72.TAU_MEM_MS),
        2: 0.75,
        3: 0.875,
        4: 0.9375,
    }
    for shift, expected in expected_beta.items():
        assert math.isclose(exp102.beta_from_mem_shift(shift), expected)
    taus = [exp102.tau_mem_ms_from_shift(s, 64.0) for s in exp102.L1_MEM_SHIFTS]
    assert all(left < right for left, right in zip(taus, taus[1:]))
    assert math.isclose(taus[0], exp102.exp72.TAU_MEM_MS, rel_tol=1e-12)
    assert math.isclose(taus[1], 54.31342963722199, rel_tol=1e-9)
    assert math.isclose(taus[2], 117.0136826471659, rel_tol=1e-9)
    assert math.isclose(taus[3], 242.10347130039656, rel_tol=1e-9)


def test_network_changes_only_l1_membrane_beta() -> None:
    model = exp102.Exp102Net(3, n_classes=12, fs=64.0)
    assert math.isclose(model.l1_lif.beta, 0.875)
    assert math.isclose(
        model.l2_lif.beta,
        math.exp(-(1000.0 / 64.0) / exp102.exp72.TAU_MEM_MS),
    )
    assert model.hidden_linears[0].in_features == 30
    assert model.hidden_linears[0].out_features == 128
    assert model.hidden_linears[1].in_features == 128
    assert model.hidden_linears[1].out_features == 128
    assert model.output_linear.in_features == 128
    assert model.output_linear.out_features == 12



def test_shift1_replays_exp101_bb_baseline_dynamics() -> None:
    spec101 = exp102.exp101.RunSpec(
        exp102.VARIANT,
        exp102.CODING,
        exp102.OBJECTIVE,
        exp102.ROTATION,
        11,
    )
    seed = exp102.exp73._e2e_pair_seed(11, "model_init")

    exp102.exp3.seed_all(seed)
    reference = exp102.exp101.Exp101Net(spec101, n_classes=12, fs=64.0)
    exp102.exp3.seed_all(seed)
    candidate = exp102.Exp102Net(1, n_classes=12, fs=64.0)

    assert set(reference.state_dict()) == set(candidate.state_dict())
    for name, value in reference.state_dict().items():
        assert torch.equal(value, candidate.state_dict()[name]), name

    generator = torch.Generator().manual_seed(123)
    x = torch.randn(2, 9, 30, generator=generator)
    with torch.no_grad():
        ref = reference.forward_trajectory(x)
        got = candidate.forward_trajectory(x)

    assert torch.equal(ref["l2_evidence"], got["l2_evidence"])
    for layer in ("l1", "l2"):
        for state in ("syn_current", "pre_reset", "spike", "post_reset"):
            assert torch.equal(
                ref["hidden"][layer][state],
                got["hidden"][layer][state],
            ), (layer, state)


def test_paired_seed_excludes_membrane_shift() -> None:
    source = (REPO_ROOT / 'scripts' / 'experiment_10_2_l1_membrane_memory.py').read_text()
    assert 'exp73._e2e_pair_seed(spec.seed, "model_init")' in source
    train_block = source[source.index('def run_train'):source.index('def run_replay')]
    assert 'l1_mem_shift, "model_init"' not in train_block
    assert 'spec.l1_mem_shift, "model_init"' not in train_block


def test_frozen_replay_has_no_optimizer_step() -> None:
    source = (REPO_ROOT / 'scripts' / 'experiment_10_2_l1_membrane_memory.py').read_text()
    replay = source[source.index('def run_replay'):source.index('def _probe_value')]
    assert 'torch.optim' not in replay
    assert '.backward(' not in replay
    assert 'optimizer.step' not in replay
    assert 'model.load_state_dict' in replay
    assert 'BASELINE_L1_MEM_SHIFT' in replay
    assert 'learned_weights_frozen' in replay


def test_single_split_is_rotation0() -> None:
    roles = exp102.exp90._rotation_fold_roles(exp102.ROTATION)
    assert roles[0] == 'test'
    assert roles[1] == 'val'
    assert all(roles[fold] == 'train' for fold in (2, 3, 4))


def test_slurm_arrays_and_dependencies() -> None:
    root = REPO_ROOT / 'scripts' / 'bash_script' / 'SNN_Bash'
    baseline = (root / 'run_exp_10_2_baseline_cpu_array.bash').read_text()
    long_e2e = (root / 'run_exp_10_2_long_e2e_cpu_array.bash').read_text()
    replay = (root / 'run_exp_10_2_replay_cpu_array.bash').read_text()
    stage_a = (root / 'submit_exp_10_2_stage_a_cpu.bash').read_text()
    stage_b = (root / 'submit_exp_10_2_stage_b_cpu.bash').read_text()
    full = (root / 'submit_exp_10_2_cpu.bash').read_text()

    assert '#SBATCH --array=0-2%3' in baseline
    assert '#SBATCH --array=0-8%9' in long_e2e
    assert '#SBATCH --array=0-11%12' in replay
    for script in (baseline, long_e2e, replay):
        assert '#SBATCH --cpus-per-task=1' in script
        assert 'OMP_NUM_THREADS=1' in script
        assert 'MKL_NUM_THREADS=1' in script
        assert 'OPENBLAS_NUM_THREADS=1' in script
        assert 'NUMEXPR_NUM_THREADS=1' in script

    assert 'afterok:${baseline_job}' in stage_a
    assert 'afterok:${replay_job}' in stage_a
    assert 'run_exp_10_2_long_e2e_cpu_array.bash' in stage_b
    assert 'afterok:${long_job}' in stage_b
    assert 'stage_a_manifest.json' in stage_b
    assert '"status": "PASS"' in stage_b
    assert '"run_count": 12' in stage_b
    assert '"protocol_version": "d1_bb_l1_mem_shift_sweep_v1"' in stage_b
    assert stage_b.index('stage_a_manifest.json') < stage_b.index('sbatch --parsable')
    assert 'exit 2' in stage_b
    assert 'afterok:${prepare_job}' in full
    assert 'afterok:${baseline_job}' in full
    assert 'afterok:${replay_job}:${long_job}' in full


def test_notebook_is_aggregation_only() -> None:
    notebook = (REPO_ROOT / 'notebooks' / 'experiment_10_2_l1_membrane_memory.ipynb').read_text()
    assert 'd1_bb_l1_mem_shift_sweep_v1' in notebook
    assert 'stage_a_replay_summary.csv' in notebook
    assert 'all_summary.csv' in notebook
    assert 'all_activity_summary.csv' in notebook
    assert 'e2e_minus_replay.csv' in notebook
    assert 'run_train(' not in notebook
    assert 'run_replay(' not in notebook
