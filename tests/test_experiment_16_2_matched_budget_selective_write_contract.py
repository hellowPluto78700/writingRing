from __future__ import annotations

from dataclasses import replace
import inspect

import numpy as np
import torch

from core_benchmark_v1.runner import smoke_protocol
from scripts import experiment_16_2_matched_budget_selective_write as exp


def _p():
    return replace(
        smoke_protocol(),
        width=6,
        input_channels=3,
        total_channels=3,
        steps=16,
        labels=("A", "B", "C"),
        batch_size=4,
        max_epochs=3,
        min_epochs=1,
        patience=1,
        relative_bins=4,
        shuffle_seeds=(101,),
    )


def test_manifest_counts_and_axes():
    assert exp.CALIBRATION_SEED == 101
    assert exp.FORMAL_SEEDS == (11, 23, 37)
    assert exp.RHOS == (0.9, 0.7, 0.5)
    assert exp.LAMBDA_B_CANDIDATES == (0.1, 0.3, 1.0, 3.0)
    assert len(exp.calibration_specs()) == 12
    assert len(exp.formal_specs()) == 24
    assert len(exp.pair_tasks()) == 12
    for seed in exp.FORMAL_SEEDS:
        cases = [s.case for s in exp.formal_specs() if s.seed == seed]
        assert cases == ["C0", "GZ0", "S90", "G90", "S70", "G70", "S50", "G50"]


def test_dynamic_gate_initialization_and_pair_equality():
    p = _p()
    x = torch.rand(4, p.steps, p.input_channels)
    lengths = torch.tensor([16, 14, 12, 10])
    for rho in exp.RHOS:
        tag = int(round(rho * 100))
        s = exp._make_model(exp.ExpSpec(f"S{tag}", 11, "static", rho=rho), p)
        g = exp._make_model(exp.ExpSpec(f"G{tag}", 11, "dynamic", rho=rho), p)
        assert torch.count_nonzero(g.gate_input.weight) == 0
        with torch.no_grad():
            so = s(x, lengths)
            go = g(x, lengths)
        valid = torch.arange(p.steps)[None, :] < lengths[:, None]
        assert torch.allclose(go["gate_g"][valid], torch.full_like(go["gate_g"][valid], rho), atol=1e-7)
        for key, idx in (("evidence", None), ("spike", 1), ("synapse", 1), ("membrane", 1)):
            left = so[key] if idx is None else so[key][idx]
            right = go[key] if idx is None else go[key][idx]
            assert torch.equal(left, right), (rho, key)


def test_c0_equivalent_to_s100():
    p = _p()
    x = torch.rand(3, p.steps, p.input_channels)
    lengths = torch.tensor([16, 12, 9])
    c0 = exp._make_model(exp.ExpSpec("C0", 23, "none"), p)
    s100 = exp.MatchedBudgetNet(exp._run("S100", 23), p, "static", 1.0)
    own = s100.state_dict()
    for name, value in exp._base_shared_state(23, p).items():
        if name in own:
            own[name] = value.clone()
    s100.load_state_dict(own, strict=True)
    with torch.no_grad():
        a, b = c0(x, lengths), s100(x, lengths)
    assert torch.equal(a["evidence"], b["evidence"])
    assert torch.equal(a["spike"][1], b["spike"][1])
    assert torch.equal(a["pre_reset"][1], b["pre_reset"][1])


def test_budget_is_per_sample_and_padding_excluded():
    gates = torch.tensor([
        [0.5, 0.5, 0.5, 9.0],
        [0.2, 0.8, 9.0, 9.0],
    ])
    lengths = torch.tensor([3, 2])
    means = exp._budget_per_sample(gates, lengths)
    assert torch.allclose(means, torch.tensor([0.5, 0.5]))
    assert torch.allclose(exp._budget_loss(gates, lengths, 0.5), torch.tensor(0.0))


def test_epoch_sampler_is_explicit_and_reproducible():
    a = exp._epoch_permutation(31, 11, 4)
    b = exp._epoch_permutation(31, 11, 4)
    c = exp._epoch_permutation(31, 11, 5)
    assert torch.equal(a, b)
    assert exp._permutation_hash(a) == exp._permutation_hash(b)
    assert not torch.equal(a, c)


def test_replay_mean_shuffle_shift_preserve_valid_mean():
    learned = torch.tensor([
        [0.1, 0.3, 0.7, 0.9, 0.0],
        [0.2, 0.8, 0.4, 0.0, 0.0],
    ])
    lengths = torch.tensor([4, 3])
    ids = np.array([101, 202])
    baseline = exp._budget_per_sample(learned, lengths)
    for mode, arg in (("mean", None), ("shuffle", 101), ("shift", 0.25), ("shift", 0.5), ("shift", 0.75)):
        forced = exp._replay_gate(learned, lengths, mode, ids, arg)
        assert torch.allclose(exp._budget_per_sample(forced, lengths), baseline)


def test_constrained_checkpoint_uses_validation_budget_only():
    source = inspect.getsource(exp.train_one)
    assert '"val"' in source
    assert "budget_mae" in source
    assert "BUDGET_MAE_TOL" in source
    assert '"test"' not in source
    assert "test_ba" not in source


def test_calibration_selection_does_not_use_ba():
    source = inspect.getsource(exp.finalize_calibration)
    assert "late_budget_mae" in source
    assert "compliant" in source
    assert "val_ba" not in source
    assert "test_ba" not in source


def test_slurm_contract_files():
    root = exp.find_repo_root()
    bash = root / "scripts" / "bash_script" / "SNN_Bash"
    phase0 = (bash / "run_exp_16_2_phase0_cpu_array.bash").read_text()
    phase1 = (bash / "run_exp_16_2_phase1_pairs_cpu_array.bash").read_text()
    phase2 = (bash / "run_exp_16_2_phase2_replay_cpu_array.bash").read_text()
    submit = (bash / "submit_exp_16_2_cpu.bash").read_text()
    assert "#SBATCH --array=0-11%12" in phase0
    assert "#SBATCH --array=0-11%12" in phase1
    assert "#SBATCH --array=0-11%12" in phase2
    assert "#SBATCH --cpus-per-task=1" in phase0
    assert "#SBATCH --cpus-per-task=1" in phase1
    assert "#SBATCH --cpus-per-task=1" in phase2
    assert "finalize-calibration" in submit
    assert "--dependency=afterok:" in submit