from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from scripts import experiment_4_0_5_temporal_resolution_event_capacity as exp405
from scripts import experiment_4_3_long_term_memory_validation as exp43


def test_run_matrix_is_five_ff_and_five_frozen_rsnn_seeds() -> None:
    specs = exp43.run_specs()
    ff = exp43.run_specs("ff")
    rsnn = exp43.run_specs("rsnn")
    assert len(specs) == 10
    assert len(ff) == 5
    assert len(rsnn) == 5
    assert [spec.seed for spec in ff] == [11, 23, 37, 53, 71]
    assert [spec.seed for spec in rsnn] == [11, 23, 37, 53, 71]
    assert {spec.model_kind for spec in specs} == {"ff", "rsnn"}
    assert len({spec.key for spec in specs}) == 10


def test_exp43_locks_current_shift34_multi_ho_architecture() -> None:
    assert exp43.LOCAL_CONDITION == "shift34"
    assert exp43.VARIANT == "multi_ho"
    assert exp43.HIDDEN_CAP == 31
    assert exp43.OUTPUT_CAP == 31
    source = exp43.source_spec(11)
    assert source.condition == "shift34"
    assert source.variant == "multi_ho"
    assert source.hidden_cap == 31
    assert source.output_cap == 31
    assert source.seed == 11


def test_memory_intervention_schedules_match_raw64_exactly() -> None:
    dt_ms = 1000.0 / 64.0
    assert exp43.reset_schedule(dt_ms) == {250: 16, 500: 32, 1000: 64}
    assert exp43.delay_schedule(dt_ms) == {
        0: 0,
        250: 16,
        500: 32,
        1000: 64,
        2000: 128,
    }
    with pytest.raises(ValueError):
        exp43._ms_to_steps(333, dt_ms)


def test_ff_and_rsnn_have_same_topology_but_ff_cannot_enable_recurrence() -> None:
    rsnn = exp43.Stage2ValidationDecoder("rsnn", n_classes=12)
    ff = exp43.Stage2ValidationDecoder("ff", n_classes=12)
    assert tuple(rsnn.input_local.weight.shape) == (128, 30)
    assert tuple(rsnn.local_state.weight.shape) == (128, 128)
    assert tuple(rsnn.recurrent.weight.shape) == (128, 128)
    assert tuple(rsnn.state_output.weight.shape) == (12, 128)
    assert set(rsnn.state_dict()) == set(ff.state_dict())

    X = torch.zeros(1, 2, exp405.exp40.EVENT_CHANNELS)
    with pytest.raises(ValueError):
        ff.forward_trajectory(X, recurrence_enabled=True)


def test_recurrence_off_is_functionally_identical_to_ff_with_same_weights() -> None:
    torch.manual_seed(7)
    rsnn = exp43.Stage2ValidationDecoder("rsnn", n_classes=12)
    ff = exp43.Stage2ValidationDecoder("ff", n_classes=12)
    ff.load_state_dict(rsnn.state_dict())
    X = torch.rand(2, 9, exp405.exp40.EVENT_CHANNELS)

    ablated = rsnn.forward_trajectory(X, recurrence_enabled=False)
    ff_traj = ff.forward_trajectory(X)
    for key in (
        "local_spikes",
        "state_spikes",
        "state_membranes",
        "output_spikes",
        "output_membranes",
    ):
        assert torch.equal(ablated[key], ff_traj[key])


def test_periodic_reset_never_changes_local_trajectory() -> None:
    torch.manual_seed(13)
    model = exp43.Stage2ValidationDecoder("rsnn", n_classes=12)
    X = torch.rand(2, 12, exp405.exp40.EVENT_CHANNELS)
    normal = model.forward_trajectory(X)
    reset = model.forward_trajectory(X, state_reset_interval_steps=4)
    assert torch.equal(normal["local_spikes"], reset["local_spikes"])
    assert torch.equal(normal["local_pre_reset"], reset["local_pre_reset"])


def test_silent_input_masks_each_sample_after_its_own_valid_endpoint() -> None:
    X = torch.ones(2, 5, 3)
    valid = torch.tensor([2, 4])
    silent = exp43._masked_silent_input(X, valid, extra_steps=3)
    assert silent.shape == (2, 8, 3)
    assert torch.equal(silent[0, :2], torch.ones(2, 3))
    assert torch.count_nonzero(silent[0, 2:]) == 0
    assert torch.equal(silent[1, :4], torch.ones(4, 3))
    assert torch.count_nonzero(silent[1, 4:]) == 0


def test_gather_at_indices_uses_per_sample_endpoints() -> None:
    sequence = torch.arange(2 * 5 * 3).reshape(2, 5, 3)
    gathered = exp43._gather_at_indices(sequence, torch.tensor([1, 3]))
    assert torch.equal(gathered[0], sequence[0, 1])
    assert torch.equal(gathered[1], sequence[1, 3])


def test_multi_cpu_scripts_follow_repository_one_core_array_contract() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    ff_script = (
        repo_root / "scripts/bash_script/SNN_Bash/run_exp_4_3_ff_cpu_array.bash"
    ).read_text()
    rsnn_script = (
        repo_root / "scripts/bash_script/SNN_Bash/eval_exp_4_3_rsnn_cpu_array.bash"
    ).read_text()
    final_script = (
        repo_root / "scripts/bash_script/SNN_Bash/finalize_exp_4_3_cpu.bash"
    ).read_text()
    submit_script = (
        repo_root / "scripts/bash_script/SNN_Bash/submit_exp_4_3_cpu.bash"
    ).read_text()

    assert "#SBATCH --array=0-4%5" in ff_script
    assert "#SBATCH --array=0-4%5" in rsnn_script
    assert "#SBATCH --cpus-per-task=1" in ff_script
    assert "#SBATCH --cpus-per-task=1" in rsnn_script
    assert '--dependency="afterok:${FF_JOB}:${RSNN_JOB}"' in submit_script
    assert "experiment_4_3_long_term_memory_validation finalize" in final_script
    for script in (ff_script, rsnn_script, final_script):
        assert "OMP_NUM_THREADS=1" in script
        assert "MKL_NUM_THREADS=1" in script
        assert "OPENBLAS_NUM_THREADS=1" in script
        assert "NUMEXPR_NUM_THREADS=1" in script
        assert "module load conda/latest" in script


def test_exp43_reuses_exact_exp406_paired_random_stream() -> None:
    spec = exp43.RunSpec("ff", 37)
    source = exp43.source_spec(37)
    assert exp43.paired_seed(spec, "model_init") == exp43.exp406.paired_seed(
        source, "model_init"
    )
    assert exp43.paired_seed(spec, "train_loader") == exp43.exp406.paired_seed(
        source, "train_loader"
    )


def test_local_shift34_timescales_remain_unchanged() -> None:
    local = exp43.exp406.local_condition_definition("shift34", fs=64.0)
    assert local["shifts"] == (3, 4)
    assert local["group_widths"] == (64, 64)
    assert np.allclose(
        local["tau_mem_ms"],
        tuple(exp405.exp40.base.tau_ms(shift, 64.0) for shift in (3, 4)),
    )
