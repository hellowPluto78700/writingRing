from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import torch

from scripts import experiment_7_0_hierarchical_context_snn as exp70


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_run_grid_is_24_paired_hierarchical_plus_3_local_runs() -> None:
    specs = exp70.run_specs()
    local = exp70.local_run_specs()
    assert len(specs) == 24
    assert len(local) == 3
    assert {spec.method for spec in specs} == set(exp70.METHODS)
    assert {spec.synapse_mode for spec in specs} == {exp70.LEGACY, exp70.NORMALIZED}
    assert {spec.seed for spec in specs} == {11, 23, 37}
    assert {spec.seed for spec in local} == {11, 23, 37}


def test_hierarchical_timescales_are_s4_s5_s6() -> None:
    assert exp70.HIERARCHICAL_SHIFTS == (4, 5, 6)
    taus = [exp70.tau_ms_from_shift(shift, 64.0) for shift in exp70.HIERARCHICAL_SHIFTS]
    assert np.allclose(taus, [242.10347130039656, 492.14626105923616, 992.1672768984098])
    assert exp70.HIDDEN_WIDTH == 128
    assert exp70.HIDDEN_CAP == 1
    assert exp70.OUTPUT_CAP == 1


def test_training_contract() -> None:
    assert exp70.MAX_EPOCHS == 100
    assert exp70.MIN_EPOCHS == 20
    assert exp70.PATIENCE == 30
    assert exp70.TRAIN_SEEDS == (11, 23, 37)
    assert exp70.EXPECTED_SPLIT_SEED == 12345


def test_synapse_updates_match_legacy_and_unit_dc_definitions() -> None:
    drive = torch.ones(2, 3)
    state = torch.zeros_like(drive)
    alpha = exp70.exp60.decay_from_shift(6)
    legacy = state
    normalized = state
    for _ in range(10000):
        legacy = exp70.synapse_step(legacy, drive, alpha, exp70.LEGACY)
        normalized = exp70.synapse_step(normalized, drive, alpha, exp70.NORMALIZED)
    assert torch.allclose(normalized, torch.ones_like(normalized), atol=1e-5)
    expected_legacy = 1.0 / (1.0 - alpha)
    assert torch.allclose(legacy, torch.full_like(legacy, expected_legacy), atol=1e-3)


def test_context_gain_initializes_to_identity() -> None:
    model = exp70.HierarchicalContextSNN(
        method=exp70.METHOD_CONTEXT_GAIN,
        synapse_mode=exp70.NORMALIZED,
        n_classes=12,
        fs=64.0,
    )
    assert model.gain_head is not None
    assert torch.count_nonzero(model.gain_head.weight).item() == 0
    long_spikes = torch.randint(0, 2, (4, exp70.HIDDEN_WIDTH)).float()
    gain = 1.0 + exp70.GAIN_LAMBDA * torch.tanh(model.gain_head(long_spikes))
    assert torch.allclose(gain, torch.ones_like(gain))


def test_what_only_omits_long_layer_and_context_methods_include_it() -> None:
    what = exp70.HierarchicalContextSNN(
        exp70.METHOD_WHAT_ONLY, exp70.NORMALIZED, 12, 64.0
    )
    assert what.long_linear is None
    assert what.long_lif is None
    for method in (
        exp70.METHOD_ALL_SKIP,
        exp70.METHOD_ADDITIVE_CONTEXT,
        exp70.METHOD_CONTEXT_GAIN,
    ):
        model = exp70.HierarchicalContextSNN(method, exp70.NORMALIZED, 12, 64.0)
        assert model.long_linear is not None
        assert model.long_lif is not None


def test_local_baseline_reuses_exp01_local_234x2_contract() -> None:
    assert exp70.LOCAL_SHIFTS == ((2, 3, 4), (2, 3, 4))
    assert exp70.LOCAL_WIDTH == 128
    assert exp70.LOCAL_HIDDEN_CAP == 1


def test_source_file_parses() -> None:
    source_path = REPO_ROOT / "scripts/experiment_7_0_hierarchical_context_snn.py"
    ast.parse(source_path.read_text(encoding="utf-8"))


def test_slurm_topology_is_parallel_worker_arrays_plus_dependency_finalizer() -> None:
    hierarchical = (
        REPO_ROOT / "scripts/bash_script/SNN_Bash/run_exp_7_0_hierarchical_cpu_array.bash"
    ).read_text(encoding="utf-8")
    local = (
        REPO_ROOT / "scripts/bash_script/SNN_Bash/run_exp_7_0_local_cpu_array.bash"
    ).read_text(encoding="utf-8")
    baseline = (
        REPO_ROOT / "scripts/bash_script/SNN_Bash/run_exp_7_0_baseline_cpu.bash"
    ).read_text(encoding="utf-8")
    final = (
        REPO_ROOT / "scripts/bash_script/SNN_Bash/finalize_exp_7_0_cpu.bash"
    ).read_text(encoding="utf-8")
    submit = (
        REPO_ROOT / "scripts/bash_script/SNN_Bash/submit_exp_7_0_cpu.bash"
    ).read_text(encoding="utf-8")

    assert "#SBATCH --array=0-23%24" in hierarchical
    assert "#SBATCH --array=0-2%3" in local
    for text in (hierarchical, local, baseline, final):
        assert "#SBATCH --cpus-per-task=1" in text
        assert "OMP_NUM_THREADS=1" in text
        assert "MKL_NUM_THREADS=1" in text
        assert "OPENBLAS_NUM_THREADS=1" in text
        assert "NUMEXPR_NUM_THREADS=1" in text
    assert "run-one" in hierarchical
    assert "run-local" in local
    assert " baseline" in baseline
    assert " finalize" in final
    assert 'afterok:${baseline_job}:${local_job}:${hierarchical_job}' in submit


def test_notebook_is_analysis_only_and_method_level() -> None:
    path = REPO_ROOT / "notebooks/experiment_7_0_hierarchical_context_snn.ipynb"
    text = path.read_text(encoding="utf-8")
    for token in (
        "summary.csv",
        "paired_neuron_deltas.csv",
        "method_deltas.csv",
        "firing_summary.csv",
    ):
        assert token in text

    forbidden = (
        "history.csv",
        "runs.csv",
        "run_one(",
        "run_hierarchical_one(",
        "train_hierarchical(",
        "train_local_baseline(",
        "torch.optim",
        ".fit(",
        "subprocess",
        "sbatch",
        "rasters/",
        "evaluations/",
    )
    for token in forbidden:
        assert token not in text
