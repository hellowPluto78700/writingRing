from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import torch

from scripts import experiment_4_0_5_temporal_resolution_event_capacity as exp405


def test_run_matrix_has_three_representations_three_variants_five_seeds() -> None:
    specs = exp405.run_specs()
    assert len(specs) == 45 == exp405.EXPECTED_RUNS
    assert {spec.representation for spec in specs} == {
        "fixed250",
        "repeat4",
        "raw64",
    }
    assert {(spec.variant, spec.hidden_cap, spec.output_cap) for spec in specs} == {
        ("binary", 1, 1),
        ("multi_h", 31, 1),
        ("multi_ho", 31, 31),
    }
    assert {spec.seed for spec in specs} == {11, 23, 37, 53, 71}
    assert len({spec.key for spec in specs}) == 45


def test_array_order_is_representation_variant_seed_major() -> None:
    specs = exp405.run_specs()
    assert [(spec.representation, spec.variant, spec.seed) for spec in specs[:5]] == [
        ("fixed250", "binary", seed) for seed in exp405.SEEDS
    ]
    assert [(spec.representation, spec.variant, spec.seed) for spec in specs[15:20]] == [
        ("repeat4", "binary", seed) for seed in exp405.SEEDS
    ]
    assert [(spec.representation, spec.variant, spec.seed) for spec in specs[40:45]] == [
        ("raw64", "multi_ho", seed) for seed in exp405.SEEDS
    ]


def test_repeat4_preserves_information_mass_and_multiplies_valid_steps() -> None:
    X = np.arange(2 * 3 * 30, dtype=np.float32).reshape(2, 3, 30)
    valid = np.array([2, 3], dtype=np.int64)
    expanded, expanded_valid = exp405.repeat4_from_fixed(X, valid)
    assert expanded.shape == (2, 12, 30)
    assert np.array_equal(expanded_valid, np.array([8, 12]))
    assert np.allclose(expanded.sum(axis=1), X.sum(axis=1))
    for b in range(3):
        block = expanded[:, b * 4 : (b + 1) * 4]
        expected = X[:, b : b + 1] / 4.0
        assert np.allclose(block, np.repeat(expected, 4, axis=1))


def test_physical_tau_is_held_constant_across_temporal_resolution() -> None:
    beta_fixed = exp405.state_beta(250.0)
    beta_repeat = exp405.state_beta(62.5)
    beta_raw = exp405.state_beta(15.625)
    assert math.isclose(beta_fixed, math.exp(-1.0))
    assert math.isclose(beta_repeat**4, beta_fixed, rel_tol=1e-12, abs_tol=1e-12)
    assert math.isclose(beta_raw**16, beta_fixed, rel_tol=1e-12, abs_tol=1e-12)
    assert math.isclose(exp405.output_beta(250.0), beta_fixed)


def test_hierarchical_architecture_keeps_local_and_memory_roles_separate() -> None:
    model = exp405.HierarchicalTemporalDecoder(
        n_classes=12,
        dt_ms=62.5,
        hidden_cap=31,
        output_cap=1,
    )
    assert model.local_lif.beta == 0.0
    assert model.local_lif.max_spikes_per_dt == 31
    assert model.state_lif.max_spikes_per_dt == 31
    assert model.output_lif.max_spikes_per_dt == 1
    assert tuple(model.input_local.weight.shape) == (128, 30)
    assert tuple(model.local_state.weight.shape) == (128, 128)
    assert tuple(model.recurrent.weight.shape) == (128, 128)
    assert tuple(model.state_output.weight.shape) == (12, 128)
    assert math.isclose(model.state_lif.beta, exp405.state_beta(62.5))
    assert math.isclose(model.output_lif.beta, exp405.output_beta(62.5))


def test_probe_features_have_no_phase_slot_access() -> None:
    model = exp405.HierarchicalTemporalDecoder(
        n_classes=12,
        dt_ms=250.0,
        hidden_cap=1,
        output_cap=1,
    )
    X = torch.zeros(2, 4, 30)
    trajectories = model.forward_trajectory(X)
    valid = torch.tensor([2, 3])
    hidden_count = exp405.exp40.valid_whole_count(
        trajectories["state_spikes"], valid
    )
    endpoint = exp405.exp40.valid_final_membrane(
        trajectories["state_membranes"], valid
    )
    assert hidden_count.shape == (2, 128)
    assert endpoint.shape == (2, 128)
    assert hidden_count.ndim == 2
    assert endpoint.ndim == 2


def test_output_normalization_depends_only_on_output_cap() -> None:
    spikes = torch.tensor([[[0.0, 31.0], [31.0, 31.0]]])
    valid = torch.tensor([2])
    evidence = exp405.normalized_output_evidence(spikes, valid, 31)
    assert torch.allclose(evidence, torch.tensor([[1.0, 2.0]]))


def test_parameter_count_is_constant_across_all_conditions() -> None:
    counts = exp405.parameter_counts(12)
    assert counts["total"] == 38144
    assert counts == {
        "input_local": 3840,
        "local_state": 16384,
        "recurrent": 16384,
        "state_output": 1536,
        "total": 38144,
    }


def test_multi_cpu_scripts_follow_repository_contract() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    array_script = (
        repo_root / "scripts/bash_script/SNN_Bash/run_exp_4_0_5_cpu_array.bash"
    ).read_text()
    submit_script = (
        repo_root / "scripts/bash_script/SNN_Bash/submit_exp_4_0_5_cpu.bash"
    ).read_text()
    finalizer = (
        repo_root / "scripts/bash_script/SNN_Bash/finalize_exp_4_0_5_cpu.bash"
    ).read_text()
    assert "#SBATCH --array=0-44%45" in array_script
    assert "#SBATCH --cpus-per-task=1" in array_script
    assert '--dependency="afterok:${ARRAY_JOB}"' in submit_script
    for script in (array_script, finalizer):
        assert "OMP_NUM_THREADS=1" in script
        assert "MKL_NUM_THREADS=1" in script
        assert "OPENBLAS_NUM_THREADS=1" in script
        assert "NUMEXPR_NUM_THREADS=1" in script
