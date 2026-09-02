from __future__ import annotations

from pathlib import Path

import torch

from scripts import experiment_4_0_3_hierarchical_snn as exp403


def test_run_matrix_has_four_new_conditions_and_five_seeds() -> None:
    specs = exp403.run_specs()
    assert len(specs) == 20 == exp403.EXPECTED_NEW_RUNS
    assert {spec.condition for spec in specs} == {
        "rsnn176_capacity",
        "local128_rsnn128",
        "rsnn128_rsnn128",
        "local128_ff128",
    }
    assert {spec.seed for spec in specs} == {11, 23, 37, 53, 71}
    assert len({spec.key for spec in specs}) == len(specs)


def test_data_order_random_stream_is_paired_across_conditions() -> None:
    seed11 = [spec for spec in exp403.run_specs() if spec.seed == 11]
    assert len({exp403.paired_seed(spec, "train_loader") for spec in seed11}) == 1
    seed23 = next(spec for spec in exp403.run_specs() if spec.seed == 23)
    assert exp403.paired_seed(seed11[0], "train_loader") != exp403.paired_seed(
        seed23, "train_loader"
    )


def test_parameter_matched_capacity_control_is_close_to_primary_hierarchy() -> None:
    capacity = exp403.parameter_counts("rsnn176_capacity", 12)
    primary = exp403.parameter_counts("local128_rsnn128", 12)
    deep = exp403.parameter_counts("rsnn128_rsnn128", 12)
    no_rec = exp403.parameter_counts("local128_ff128", 12)
    assert capacity["total"] == 38368
    assert primary["total"] == 38144
    assert abs(capacity["total"] - primary["total"]) == 224
    assert deep["total"] == 54528
    assert no_rec["total"] == 21760


def test_primary_hierarchy_separates_local_coding_from_recurrent_memory() -> None:
    model = exp403.HierarchicalMacroDecoder("local128_rsnn128", 12)
    assert model.layer1_width == 128
    assert model.layer2_width == 128
    assert model.recurrent1 is None
    assert model.layer1_lif.beta == 0.0
    assert model.recurrent2 is not None
    assert tuple(model.recurrent2.weight.shape) == (128, 128)
    assert model.layer2_lif is not None
    assert model.layer2_lif.beta == exp403.state_beta()
    assert model.layer1_lif.max_spikes_per_dt == 31
    assert model.layer2_lif.max_spikes_per_dt == 31
    assert model.output_lif.max_spikes_per_dt == 31


def test_depth_and_no_recurrence_controls_have_expected_paths() -> None:
    deep = exp403.HierarchicalMacroDecoder("rsnn128_rsnn128", 12)
    no_rec = exp403.HierarchicalMacroDecoder("local128_ff128", 12)
    capacity = exp403.HierarchicalMacroDecoder("rsnn176_capacity", 12)
    assert deep.recurrent1 is not None
    assert deep.recurrent2 is not None
    assert no_rec.recurrent1 is None
    assert no_rec.recurrent2 is None
    assert no_rec.layer1_lif.beta == 0.0
    assert capacity.layer1_width == 176
    assert capacity.layer2_width is None
    assert capacity.recurrent1 is not None
    assert capacity.state_width == 176


def test_local_beta_zero_ignores_previous_macro_bin_membrane() -> None:
    model = exp403.HierarchicalMacroDecoder("local128_rsnn128", 12)
    current = torch.full((1, 128), 0.4)
    mem_a = torch.zeros_like(current)
    mem_b = torch.full_like(current, 100.0)
    spikes_a, next_a, pre_a = model.layer1_lif(current, mem_a)
    spikes_b, next_b, pre_b = model.layer1_lif(current, mem_b)
    assert torch.equal(pre_a, pre_b)
    assert torch.equal(spikes_a, spikes_b)
    assert torch.equal(next_a, next_b)


def test_forward_consumes_only_fixed250_channel_vectors() -> None:
    model = exp403.HierarchicalMacroDecoder("local128_rsnn128", 12)
    X = torch.zeros(2, 4, exp403.exp40.EVENT_CHANNELS)
    trajectories = model.forward_trajectory(X)
    assert trajectories["layer1_spikes"].shape == (2, 4, 128)
    assert trajectories["state_spikes"].shape == (2, 4, 128)
    assert trajectories["state_membranes"].shape == (2, 4, 128)
    assert trajectories["output_spikes"].shape == (2, 4, 12)
    assert "no sub-bin timing" in exp403.INPUT_INFORMATION_CONTRACT


def test_endpoint_state_probe_feature_uses_only_causal_endpoint_state() -> None:
    states = torch.tensor(
        [
            [[1.0, 10.0], [2.0, 20.0], [3.0, 30.0], [4.0, 40.0]],
            [[5.0, 50.0], [6.0, 60.0], [7.0, 70.0], [8.0, 80.0]],
        ]
    )
    valid_bins = torch.tensor([2, 3])
    endpoint = exp403.valid_final_state(states, valid_bins)
    assert torch.equal(endpoint, torch.tensor([[2.0, 20.0], [7.0, 70.0]]))


def test_multi_cpu_scripts_follow_repository_array_contract() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    array_script = (
        repo_root / "scripts/bash_script/SNN_Bash/run_exp_4_0_3_cpu_array.bash"
    ).read_text()
    probe_script = (
        repo_root / "scripts/bash_script/SNN_Bash/probe_exp_4_0_3_baseline_cpu.bash"
    ).read_text()
    submit_script = (
        repo_root / "scripts/bash_script/SNN_Bash/submit_exp_4_0_3_cpu.bash"
    ).read_text()
    assert "#SBATCH --array=0-19%20" in array_script
    assert "#SBATCH --cpus-per-task=1" in array_script
    assert "--dependency=\"afterok:${ARRAY_JOB}:${PROBE_JOB}\"" in submit_script
    for script in (array_script, probe_script):
        assert "OMP_NUM_THREADS=1" in script
        assert "MKL_NUM_THREADS=1" in script
        assert "OPENBLAS_NUM_THREADS=1" in script
        assert "NUMEXPR_NUM_THREADS=1" in script
