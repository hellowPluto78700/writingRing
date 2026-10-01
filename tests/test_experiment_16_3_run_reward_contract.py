from __future__ import annotations

from dataclasses import replace
import inspect
import math

import numpy as np
import torch

from core_benchmark_v1.model import mean_logits
from core_benchmark_v1.runner import smoke_protocol
from scripts import experiment_16_3_run_reward as exp


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


def test_formal_manifest_and_cases():
    assert exp.FORMAL_SEEDS == (11, 23, 37)
    assert [row[0] for row in exp.FORMAL_CASES] == [
        "LIN",
        "CAP4",
        "CAP8",
        "CAP16",
        "CAP24",
        "SUB8",
        "EP1",
    ]
    specs = exp.formal_specs()
    assert len(specs) == 21
    for seed in exp.FORMAL_SEEDS:
        assert [s.case for s in specs if s.seed == seed] == [
            "LIN",
            "CAP4",
            "CAP8",
            "CAP16",
            "CAP24",
            "SUB8",
            "EP1",
        ]
    assert len(exp.objective_diagnostic_specs()) == 6


def test_all_formal_cases_share_exact_initial_state():
    p = _p()
    for seed in exp.FORMAL_SEEDS:
        models = [
            exp._make_model(exp.ExpSpec(case, seed, kind, param), p)
            for case, kind, param in exp.FORMAL_CASES
        ]
        reference = models[0].state_dict()
        for model in models[1:]:
            current = model.state_dict()
            assert reference.keys() == current.keys()
            for name in reference:
                assert torch.equal(reference[name], current[name]), (seed, name)


def test_linear_reward_is_exact_wcce_readout_geometry():
    p = _p()
    spec = exp.ExpSpec("LIN", 11, "linear", None)
    model = exp._make_model(spec, p)
    x = torch.rand(3, p.steps, p.input_channels)
    lengths = torch.tensor([8, 6, 5])
    with torch.no_grad():
        out = model(x, lengths)
        reward = exp.reward_logits(model, out, lengths, spec)
        wcce = mean_logits(out["evidence"], lengths)
    assert torch.allclose(reward, wcce, atol=1e-7, rtol=1e-7)


def test_run_reward_forward_values():
    spikes = torch.tensor(
        [[[1.0], [1.0], [1.0], [0.0], [1.0], [1.0], [1.0], [1.0]]]
    )
    lengths = torch.tensor([8])
    linear = exp.run_reward_feature(spikes, lengths, "linear", None)
    cap2 = exp.run_reward_feature(spikes, lengths, "cap", 2.0)
    episode = exp.run_reward_feature(spikes, lengths, "episode", 1.0)
    sub8 = exp.run_reward_feature(spikes, lengths, "sublinear", 8.0)

    assert torch.allclose(linear, torch.tensor([[7.0 / 8.0]]))
    assert torch.allclose(cap2, torch.tensor([[4.0 / 8.0]]))
    assert torch.allclose(episode, torch.tensor([[2.0 / 8.0]]))

    phi3 = 8.0 * (1.0 - math.exp(-3.0 / 8.0))
    phi4 = 8.0 * (1.0 - math.exp(-4.0 / 8.0))
    assert torch.allclose(
        sub8,
        torch.tensor([[(phi3 + phi4) / 8.0]]),
        atol=1e-6,
    )


def test_cap_gradient_matches_marginal_repeated_reward():
    spikes = torch.tensor(
        [[[1.0], [1.0], [1.0], [0.0], [1.0], [1.0], [1.0], [1.0]]],
        requires_grad=True,
    )
    lengths = torch.tensor([8])
    exp.run_reward_feature(spikes, lengths, "cap", 2.0).sum().backward()
    got = spikes.grad[0, :, 0]
    expected = torch.tensor([1, 1, 0, 0, 1, 1, 0, 0], dtype=torch.float32) / 8.0
    assert torch.allclose(got, expected)


def test_episode_forward_equals_cap1_and_has_onset_marginal_reward():
    spikes = torch.tensor(
        [[[1.0], [1.0], [0.0], [1.0], [1.0], [0.0], [0.0], [1.0]]],
        requires_grad=True,
    )
    lengths = torch.tensor([8])
    episode = exp.run_reward_feature(spikes, lengths, "episode", 1.0)
    cap1 = exp.run_reward_feature(spikes, lengths, "cap", 1.0)
    assert torch.allclose(episode, cap1)

    episode.sum().backward()
    # A hypothetical spike immediately after an existing spike is a continuation
    # and gets zero marginal episode reward; after a gap it starts a new episode.
    expected = torch.tensor([1, 0, 0, 1, 0, 0, 1, 1], dtype=torch.float32) / 8.0
    assert torch.allclose(spikes.grad[0, :, 0], expected)


def test_numpy_run_features_match_torch_forward():
    z = np.asarray(
        [[[1], [1], [1], [0], [1], [1], [1], [1]]],
        dtype=np.uint8,
    )
    lengths = np.asarray([8])
    torch_z = torch.from_numpy(z.astype(np.float32))
    for kind, param in (
        ("linear", None),
        ("cap", 2.0),
        ("episode", 1.0),
        ("sublinear", 8.0),
    ):
        a = exp._run_position_features_np(z, lengths, kind, param)
        b = exp.run_reward_feature(
            torch_z,
            torch.from_numpy(lengths),
            kind,
            param,
        ).numpy()
        assert np.allclose(a, b, atol=1e-7)


def test_sampler_is_epoch_specific_and_reproducible():
    a = exp._epoch_permutation(37, 11, 5)
    b = exp._epoch_permutation(37, 11, 5)
    c = exp._epoch_permutation(37, 11, 6)
    assert torch.equal(a, b)
    assert exp._permutation_hash(a) == exp._permutation_hash(b)
    assert not torch.equal(a, c)


def test_formal_training_selection_never_reads_test():
    source = inspect.getsource(exp.train_one)
    assert '_split_eval(model, arrays, p, spec, "val")' in source
    assert '"test"' not in source
    assert "test_ba" not in source


def test_slurm_contract():
    root = exp.find_repo_root()
    bash = root / "scripts" / "bash_script" / "SNN_Bash"
    scale = (bash / "run_exp_16_3_scale_cpu_array.bash").read_text()
    objective = (bash / "run_exp_16_3_objective_cpu_array.bash").read_text()
    train = (bash / "run_exp_16_3_train_cpu_array.bash").read_text()
    submit = (bash / "submit_exp_16_3_cpu.bash").read_text()

    assert "#SBATCH --array=0-23%24" in scale
    assert "#SBATCH --array=0-5%6" in objective
    assert "#SBATCH --array=0-20%21" in train
    for source in (scale, objective, train):
        assert "#SBATCH --cpus-per-task=1" in source
        assert "source /etc/profile" in source
    assert "--dependency=afterok:" in submit
    assert "jid_scale" in submit
    assert "jid_obj" in submit
    assert "jid_train" in submit
