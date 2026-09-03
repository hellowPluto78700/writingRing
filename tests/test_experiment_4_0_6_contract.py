from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from scripts import experiment_4_0_5_temporal_resolution_event_capacity as exp405
from scripts import experiment_4_0_6_local_mem_raw64 as exp406


def test_run_matrix_has_45_new_runs_and_15_frozen_baselines() -> None:
    specs = exp406.run_specs()
    baselines = exp406.baseline_specs()
    assert len(specs) == 45 == exp406.EXPECTED_NEW_RUNS
    assert len(baselines) == 15 == exp406.EXPECTED_BASELINE_PROBES
    assert {spec.condition for spec in specs} == {"shift4", "shift34", "shift234"}
    assert {spec.condition for spec in baselines} == {"beta0"}
    assert {spec.variant for spec in specs} == {"binary", "multi_h", "multi_ho"}
    assert {spec.seed for spec in specs} == {11, 23, 37, 53, 71}
    assert len({spec.key for spec in [*baselines, *specs]}) == 60


def test_training_array_mapping_is_condition_then_variant_then_seed() -> None:
    specs = exp406.run_specs()
    assert [spec.condition for spec in specs[:15]] == ["shift4"] * 15
    assert [spec.condition for spec in specs[15:30]] == ["shift34"] * 15
    assert [spec.condition for spec in specs[30:45]] == ["shift234"] * 15
    assert [(spec.variant, spec.seed) for spec in specs[:15]] == [
        (variant, seed)
        for variant in ("binary", "multi_h", "multi_ho")
        for seed in (11, 23, 37, 53, 71)
    ]


def test_local_shift_groups_and_width_allocations_are_locked() -> None:
    single = exp406.local_condition_definition("shift4")
    double = exp406.local_condition_definition("shift34")
    triple = exp406.local_condition_definition("shift234")
    baseline = exp406.local_condition_definition("beta0")
    assert single["shifts"] == (4,)
    assert single["group_widths"] == (128,)
    assert double["shifts"] == (3, 4)
    assert double["group_widths"] == (64, 64)
    assert triple["shifts"] == (2, 3, 4)
    assert triple["group_widths"] == (43, 43, 42)
    assert baseline["shifts"] == ()
    assert baseline["betas"] == (0.0,)


def test_local_timescales_use_repository_shift_convention_at_64_hz() -> None:
    triple = exp406.local_condition_definition("shift234", fs=64.0)
    expected_betas = tuple(exp405.exp40.base.alpha(shift) for shift in (2, 3, 4))
    expected_taus = tuple(exp405.exp40.base.tau_ms(shift, 64.0) for shift in (2, 3, 4))
    assert np.allclose(triple["betas"], expected_betas)
    assert np.allclose(triple["tau_mem_ms"], expected_taus)
    assert expected_taus[0] < expected_taus[1] < expected_taus[2]
    assert 50.0 < expected_taus[0] < 60.0
    assert 110.0 < expected_taus[1] < 125.0
    assert 235.0 < expected_taus[2] < 250.0


def test_heterogeneous_beta_vector_matches_group_counts() -> None:
    beta = exp406.local_beta_vector("shift234")
    b2 = exp405.exp40.base.alpha(2)
    b3 = exp405.exp40.base.alpha(3)
    b4 = exp405.exp40.base.alpha(4)
    assert beta.shape == (128,)
    assert torch.allclose(beta[:43], torch.full((43,), b2))
    assert torch.allclose(beta[43:86], torch.full((43,), b3))
    assert torch.allclose(beta[86:], torch.full((42,), b4))


def test_local_memory_changes_only_local_membrane_decay_not_topology() -> None:
    model = exp406.LocalMemoryRaw64Decoder(
        condition="shift234",
        n_classes=12,
        hidden_cap=31,
        output_cap=31,
    )
    assert tuple(model.input_local.weight.shape) == (128, 30)
    assert tuple(model.local_state.weight.shape) == (128, 128)
    assert tuple(model.recurrent.weight.shape) == (128, 128)
    assert tuple(model.state_output.weight.shape) == (12, 128)
    assert model.local_lif.beta.shape == (128,)
    assert np.isclose(model.state_lif.beta, exp405.state_beta(exp406.RAW_DT_MS))
    assert np.isclose(model.output_lif.beta, exp405.output_beta(exp406.RAW_DT_MS))
    assert model.hidden_cap == 31
    assert model.output_cap == 31


def test_local_layer_has_state_but_no_learned_local_recurrent_matrix() -> None:
    model = exp406.LocalMemoryRaw64Decoder("shift4", 12, 1, 1)
    parameter_names = {name for name, _ in model.named_parameters()}
    assert "recurrent.weight" in parameter_names
    assert not any(name.startswith("local_recurrent") for name in parameter_names)
    assert not any(name.startswith("recurrent_local") for name in parameter_names)


def test_forward_consumes_raw64_steps_without_pooling() -> None:
    model = exp406.LocalMemoryRaw64Decoder("shift34", 12, 1, 1)
    X = torch.zeros(2, 7, exp405.exp40.EVENT_CHANNELS)
    traj = model.forward_trajectory(X)
    assert traj["local_spikes"].shape == (2, 7, 128)
    assert traj["state_spikes"].shape == (2, 7, 128)
    assert traj["state_membranes"].shape == (2, 7, 128)
    assert traj["output_spikes"].shape == (2, 7, 12)


def test_new_conditions_reuse_exact_exp405_paired_random_stream() -> None:
    spec = exp406.run_specs()[0]
    source = exp405.RunSpec(
        representation="raw64",
        variant=spec.variant,
        hidden_cap=spec.hidden_cap,
        output_cap=spec.output_cap,
        seed=spec.seed,
    )
    assert exp406.paired_seed(spec, "model_init") == exp405.paired_seed(
        source, "model_init"
    )
    assert exp406.paired_seed(spec, "train_loader") == exp405.paired_seed(
        source, "train_loader"
    )


def test_beta0_source_is_raw64_exp405_and_is_not_in_training_matrix() -> None:
    baseline = exp406.baseline_specs()[0]
    source = exp406.source_exp405_spec(baseline)
    assert source.representation == "raw64"
    assert source.variant == baseline.variant
    assert source.hidden_cap == baseline.hidden_cap
    assert source.output_cap == baseline.output_cap
    assert source.seed == baseline.seed
    assert all(spec.condition != "beta0" for spec in exp406.run_specs())


def test_probe_pipeline_standardizes_train_features_before_linear_classifier() -> None:
    pipeline = exp406._scaled_linear_probe()
    assert isinstance(pipeline.named_steps["scale"], StandardScaler)
    classifier = pipeline.named_steps["classifier"]
    assert isinstance(classifier, LogisticRegression)
    assert classifier.class_weight == "balanced"
    assert classifier.solver == "lbfgs"
    assert classifier.max_iter == exp406.PROBE_MAX_ITER


def test_multi_cpu_scripts_follow_repository_contract() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    train_script = (
        repo_root / "scripts/bash_script/SNN_Bash/run_exp_4_0_6_cpu_array.bash"
    ).read_text()
    baseline_script = (
        repo_root / "scripts/bash_script/SNN_Bash/probe_exp_4_0_6_baseline_cpu_array.bash"
    ).read_text()
    submit_script = (
        repo_root / "scripts/bash_script/SNN_Bash/submit_exp_4_0_6_cpu.bash"
    ).read_text()
    final_script = (
        repo_root / "scripts/bash_script/SNN_Bash/finalize_exp_4_0_6_cpu.bash"
    ).read_text()

    assert "#SBATCH --array=0-44%45" in train_script
    assert "#SBATCH --array=0-14%15" in baseline_script
    assert "#SBATCH --cpus-per-task=1" in train_script
    assert "#SBATCH --cpus-per-task=1" in baseline_script
    assert '--dependency="afterok:${TRAIN_JOB}:${BASE_JOB}"' in submit_script
    assert "experiment_4_0_6_local_mem_raw64 finalize" in final_script
    for script in (train_script, baseline_script, final_script):
        assert "OMP_NUM_THREADS=1" in script
        assert "MKL_NUM_THREADS=1" in script
        assert "OPENBLAS_NUM_THREADS=1" in script
        assert "NUMEXPR_NUM_THREADS=1" in script
        assert "module load conda/latest" in script
