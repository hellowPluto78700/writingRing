from __future__ import annotations

from dataclasses import replace
import inspect
from pathlib import Path

import numpy as np
import torch

from core_benchmark_v1.model import BenchmarkNet
from core_benchmark_v1.protocol import Run
from core_benchmark_v1.runner import smoke_protocol
from scripts import experiment_17_2_history_input_competition as exp


def _p():
    return replace(
        smoke_protocol(),
        width=12,
        input_channels=3,
        total_channels=3,
        steps=16,
        labels=("A", "B", "C"),
        batch_size=4,
    )


def _model(seed: int = 11):
    p = _p()
    run = Run(
        "LIN",
        seed,
        "exp17_2_test",
        shifts=((2, 3, 4), (2, 3, 4)),
        objective="wcce",
    )
    return BenchmarkNet(run, p), p


def test_protocol_axes_are_locked():
    assert exp.FORMAL_SEEDS == (11, 23, 37)
    assert exp.FIXED_GROUP_EPOCH == 20
    assert exp.PERSISTENT_QUARTILE == 0.25
    assert exp.RELATIVE_BINS == 10
    assert exp.HORIZON_STEPS == (1, 2, 4, 8, 16)
    assert exp.HORIZON_SPLITS == ("train", "test")
    assert exp.HORIZON_ANCHOR_STRIDE == 8


def test_instrumented_factual_replay_matches_benchmark_forward_exactly():
    model, p = _model()
    torch.manual_seed(4)
    x = torch.rand(3, p.steps, p.input_channels)
    lengths = torch.tensor([p.steps, p.steps - 3, 7])

    with torch.no_grad():
        reference = model(x, lengths)
        replay = exp.instrumented_replay(model, x, lengths)

    assert torch.equal(reference["evidence"], replay["evidence"])
    for key in ("spike", "pre_reset", "final_syn", "final_mem"):
        assert len(reference[key]) == len(replay[key])
        for a, b in zip(reference[key], replay[key]):
            assert torch.equal(a, b), key


def test_drive_decomposition_identities_hold_on_valid_timesteps():
    model, p = _model()
    torch.manual_seed(8)
    x = torch.rand(2, p.steps, p.input_channels)
    lengths = torch.tensor([p.steps, 9])

    with torch.no_grad():
        replay = exp.instrumented_replay(model, x, lengths)

    valid = (
        torch.arange(p.steps)[None, :] < lengths[:, None]
    )[:, :, None]
    for layer, comp in enumerate(replay["components"]):
        syn_expected = comp["syn_history"] + comp["new_drive"]
        pre_expected = comp["mem_carry"] + comp["syn_state"]
        assert torch.allclose(comp["syn_state"][valid.expand_as(syn_expected)],
                              syn_expected[valid.expand_as(syn_expected)])
        assert torch.allclose(replay["pre_reset"][layer][valid.expand_as(pre_expected)],
                              pre_expected[valid.expand_as(pre_expected)])


def test_padding_values_do_not_change_valid_diagnostics():
    model, p = _model()
    torch.manual_seed(12)
    lengths = torch.tensor([8, 11])
    x = torch.rand(2, p.steps, p.input_channels)
    y = x.clone()
    for i, length in enumerate(lengths.tolist()):
        y[i, length:] = torch.randn_like(y[i, length:]) * 100.0

    with torch.no_grad():
        a = exp.instrumented_replay(model, x, lengths)
        b = exp.instrumented_replay(model, y, lengths)

    valid = (
        torch.arange(p.steps)[None, :] < lengths[:, None]
    )[:, :, None]
    for layer in range(2):
        mask = valid.expand_as(a["spike"][layer])
        assert torch.equal(a["spike"][layer][mask], b["spike"][layer][mask])
        for name in ("mem_carry", "syn_history", "new_drive", "syn_state"):
            assert torch.equal(
                a["components"][layer][name][mask],
                b["components"][layer][name][mask],
            )


def test_decision_masks_distinguish_history_trigger_and_suppression():
    full = torch.tensor([1, 1, 0, 0], dtype=torch.float32)
    no_new = torch.tensor([1, 0, 1, 0], dtype=torch.bool)
    masks = exp._decision_masks(full, no_new)
    assert masks["history_sufficient"].tolist() == [True, False, False, False]
    assert masks["new_triggered"].tolist() == [False, True, False, False]
    assert masks["new_suppressed"].tolist() == [False, False, True, False]
    assert masks["incoming_flip"].tolist() == [False, True, True, False]


def test_fixed_quartile_groups_are_deterministic_and_dimension_matched():
    values = np.arange(12, dtype=float)
    groups = exp._quartile_groups(values)
    assert sum(v == "high25" for v in groups.values()) == 3
    assert sum(v == "low25" for v in groups.values()) == 3
    assert sum(v == "middle50" for v in groups.values()) == 6
    assert [k for k, v in groups.items() if v == "high25"] == [9, 10, 11]
    assert [k for k, v in groups.items() if v == "low25"] == [0, 1, 2]


def test_tau_shift_labels_match_locked_group_allocation():
    model, _ = _model()
    labels = exp._tau_shift_labels(model, 1)
    assert len(labels) == 12
    assert np.array_equal(labels, np.array([2] * 4 + [3] * 4 + [4] * 4))


def test_horizon_epoch_selection_includes_fixed_and_role_epochs():
    manifest = {
        "snapshots": [
            {"epoch": 0, "roles": ["initial"]},
            {"epoch": 20, "roles": ["periodic"]},
            {"epoch": 40, "roles": ["periodic"]},
            {"epoch": 60, "roles": ["periodic"]},
            {"epoch": 80, "roles": ["periodic"]},
            {"epoch": 95, "roles": ["selected_best"]},
            {"epoch": 100, "roles": ["stopped"]},
        ]
    }
    assert exp._horizon_epochs(manifest) == [0, 20, 40, 60, 80, 95, 100]


def test_runner_is_artifact_only_and_never_contains_training_step():
    source = inspect.getsource(exp)
    assert "optimizer.step(" not in source
    assert ".backward(" not in source
    assert "loss.backward(" not in source


def test_slurm_contract_files_are_present_and_cpu_arrayed():
    root = exp.find_repo_root()
    bash = root / "scripts" / "bash_script" / "SNN_Bash"
    replay = (bash / "run_exp_17_2_replay_cpu_array.bash").read_text()
    horizon = (bash / "run_exp_17_2_horizon_cpu_array.bash").read_text()
    submit = (bash / "submit_exp_17_2_cpu.bash").read_text()

    for source in (replay, horizon):
        assert "#SBATCH --array=0-2%3" in source
        assert "#SBATCH --cpus-per-task=1" in source
        assert "OMP_NUM_THREADS=1" in source
        assert "conda activate writingring-gpu" in source

    assert "--dependency=afterok:" in submit
    assert "jid_replay" in submit
    assert "jid_horizon" in submit
    assert "jid_final" in submit
