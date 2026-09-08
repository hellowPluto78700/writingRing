from __future__ import annotations

import inspect
import json
from pathlib import Path

import numpy as np
import torch

from scripts import experiment_5_5_3_teacher_weight_routing_decomposition as exp553


REPO_ROOT = Path(__file__).resolve().parents[1]
SLURM_DIR = REPO_ROOT / "scripts/bash_script/SNN_Bash"
README = REPO_ROOT / "scripts/experiment_5_5_3/README.md"
NOTEBOOK = REPO_ROOT / "notebooks/experiment_5_5_3_teacher_weight_routing_decomposition.ipynb"
RUNNERS = (
    "prepare_exp_5_5_3_teachers_cpu_array.bash",
    "run_exp_5_5_3_train_cpu_array.bash",
    "run_exp_5_5_3_linear_cpu_array.bash",
    "finalize_exp_5_5_3_cpu.bash",
)
SUBMITTER = "submit_exp_5_5_3_cpu.bash"


def _model(family: str, routing: str, n_states: int, bin_steps: int) -> exp553.RoutedWeightBank:
    torch.manual_seed(123)
    model = exp553.RoutedWeightBank(
        family=family,
        routing=routing,
        n_states=n_states,
        bin_steps=bin_steps,
        sampling_rate_hz=8.0,
    )
    model.reset_random()
    return model.eval()


def test_protocol_and_run_mapping() -> None:
    assert exp553.PROTOCOL_VERSION == "teacher_weight_routing_decomposition_v1"
    assert exp553.SEEDS == (11, 23, 37, 53, 71)
    assert exp553.WHAT_WIDTH == 128
    assert exp553.RELATIVE_STATES == 10
    assert exp553.FAMILIES == ("relative10", "fixed250")
    assert exp553.TRAIN_CONDITIONS == (
        "hard_random_train",
        "soft_random_train",
        "soft_teacher_init_retrain",
    )
    train = exp553.train_specs()
    linear = exp553.linear_specs()
    assert len(train) == 30
    assert len({spec.key for spec in train}) == 30
    assert len(linear) == 10
    assert len({spec.key for spec in linear}) == 10
    assert train[0] == exp553.TrainSpec(exp553.RELATIVE10, exp553.HARD_RANDOM, 11)
    assert train[-1] == exp553.TrainSpec(exp553.FIXED250, exp553.SOFT_TEACHER_INIT, 71)


def test_relative_feature_definition_matches_hard_routing() -> None:
    rng = np.random.default_rng(1)
    what = rng.normal(size=(3, 13, exp553.WHAT_WIDTH)).astype(np.float32)
    lengths = np.asarray([13, 10, 7], dtype=np.int64)
    direct = exp553._relative_features(what, lengths, n_states=10).reshape(3, 10, exp553.WHAT_WIDTH)
    routed = exp553._routed_features_numpy(
        what,
        lengths,
        exp553.RELATIVE10,
        "hard",
        10,
        8.0,
        0,
    )
    assert np.allclose(direct, routed, atol=0.0, rtol=0.0)


def test_fixed_hard_routing_has_full_batch_shape_and_expected_bins() -> None:
    lengths = torch.tensor([8, 5, 3], dtype=torch.long)
    model = _model(exp553.FIXED250, "hard", n_states=4, bin_steps=2)
    q = model.routing_probabilities(lengths, steps=8, dtype=torch.float32)
    assert q.shape == (3, 8, 4)
    assert torch.equal(q[0].argmax(dim=1), torch.tensor([0, 0, 1, 1, 2, 2, 3, 3]))
    assert torch.equal(q[1, :5].argmax(dim=1), torch.tensor([0, 0, 1, 1, 2]))
    assert torch.equal(q[2, :3].argmax(dim=1), torch.tensor([0, 0, 1]))
    assert torch.count_nonzero(q[1, 5:]) == 0
    assert torch.count_nonzero(q[2, 3:]) == 0


def test_soft_routing_is_normalized_only_on_valid_prefix() -> None:
    lengths = torch.tensor([8, 5], dtype=torch.long)
    for family, n_states, bin_steps in (
        (exp553.RELATIVE10, 10, 0),
        (exp553.FIXED250, 4, 2),
    ):
        model = _model(family, "soft", n_states=n_states, bin_steps=bin_steps)
        q = model.routing_probabilities(lengths, steps=8, dtype=torch.float32)
        assert torch.allclose(q[0, :8].sum(dim=1), torch.ones(8), atol=1e-6, rtol=1e-6)
        assert torch.allclose(q[1, :5].sum(dim=1), torch.ones(5), atol=1e-6, rtol=1e-6)
        assert torch.count_nonzero(q[1, 5:]) == 0


def test_padding_suffix_cannot_change_logits() -> None:
    lengths = torch.tensor([5, 7], dtype=torch.long)
    base = torch.randn(2, 9, exp553.WHAT_WIDTH)
    changed = base.clone()
    changed[0, 5:] = torch.randn_like(changed[0, 5:]) * 100.0
    changed[1, 7:] = torch.randn_like(changed[1, 7:]) * 100.0
    for family, n_states, bin_steps in (
        (exp553.RELATIVE10, 10, 0),
        (exp553.FIXED250, 5, 2),
    ):
        for routing in ("hard", "soft"):
            model = _model(family, routing, n_states=n_states, bin_steps=bin_steps)
            with torch.no_grad():
                logits_a = model(base, lengths)
                logits_b = model(changed, lengths)
            assert torch.equal(logits_a, logits_b)


def test_teacher_initialization_is_exact() -> None:
    rng = np.random.default_rng(3)
    weight = rng.normal(size=(10, exp553.N_CLASSES, exp553.WHAT_WIDTH)).astype(np.float32)
    bias = rng.normal(size=(exp553.N_CLASSES,)).astype(np.float32)
    model = _model(exp553.RELATIVE10, "soft", n_states=10, bin_steps=0)
    model.initialize_teacher(weight, bias)
    assert np.array_equal(model.weight_bank.detach().cpu().numpy(), weight)
    assert np.array_equal(model.class_bias.detach().cpu().numpy(), bias)


def test_random_hard_and_soft_conditions_share_constructor_seed_contract() -> None:
    source = inspect.getsource(exp553._initialize_train_model)
    assert 'base.dseed(spec.seed, EXPERIMENT_ID, spec.family, "paired_random_init")' in source
    assert "spec.condition" not in source.split("paired_random_init")[0].split("base.dseed(")[-1]


def test_epoch_zero_is_eligible_and_test_does_not_select_checkpoint() -> None:
    source = inspect.getsource(exp553.run_train_one)
    assert "epoch0_val" in source
    assert "best_epoch = 0" in source
    assert "best_state = copy.deepcopy(model.state_dict())" in source
    assert '"test_used_for_checkpoint_selection": False' in source
    assert '("train", "val", "test")' in source


def test_finalizer_is_artifact_only() -> None:
    source = inspect.getsource(exp553.finalize_experiment)
    assert "Finalizer will not regenerate" in source
    assert "Finalizer will not retrain" in source
    assert "run_train_one(" not in source
    assert "run_linear_one(" not in source
    assert "prepare_teacher_seed(" not in source


def test_slurm_multi_cpu_dependency_contract() -> None:
    paths = {name: SLURM_DIR / name for name in RUNNERS + (SUBMITTER,)}
    for name, path in paths.items():
        assert path.is_file(), name
        text = path.read_text(encoding="utf-8")
        assert text.startswith("#!/usr/bin/env bash")
        assert "--wrap" not in text
        assert "/usr/bin/sbatch" not in text

    teacher = paths[RUNNERS[0]].read_text(encoding="utf-8")
    train = paths[RUNNERS[1]].read_text(encoding="utf-8")
    linear = paths[RUNNERS[2]].read_text(encoding="utf-8")
    finalize = paths[RUNNERS[3]].read_text(encoding="utf-8")
    submit = paths[SUBMITTER].read_text(encoding="utf-8")

    assert "#SBATCH --array=0-4%5" in teacher
    assert "#SBATCH --array=0-29%30" in train
    assert "#SBATCH --array=0-9%10" in linear
    for text in (teacher, train, linear, finalize):
        assert "--cpus-per-task=1" in text
        assert "module load conda/latest" in text
        assert 'eval "$(conda shell.bash hook)"' in text
        assert "conda activate writingring-gpu" in text
        assert "conda activate writingring-viz" in text
        for variable in (
            "OMP_NUM_THREADS=1",
            "MKL_NUM_THREADS=1",
            "OPENBLAS_NUM_THREADS=1",
            "NUMEXPR_NUM_THREADS=1",
        ):
            assert variable in text

    assert "command -v sbatch" in submit
    assert submit.count('dependency="afterok:${TEACHER_JOB}"') == 2
    assert 'dependency="afterok:${TRAIN_JOB}:${LINEAR_JOB}"' in submit


def test_readme_and_notebook_are_analysis_artifact_driven() -> None:
    assert README.is_file()
    readme = README.read_text(encoding="utf-8")
    assert "30 train->evaluate" in readme
    assert "10 soft-linear-refit" in readme
    assert "artifact-only finalizer" in readme
    assert "analysis-only" in readme

    assert NOTEBOOK.is_file()
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    assert notebook["nbformat"] == 4
    text = NOTEBOOK.read_text(encoding="utf-8")
    for artifact in (
        "teacher_equivalence.csv",
        "runs.csv",
        "summary.csv",
        "paired_deltas.csv",
        "recovery.csv",
        "logit_distortion.csv",
        "weight_drift.csv",
    ):
        assert artifact in text
    for forbidden in (
        "run_train_one",
        "run_linear_one",
        "prepare_teacher_seed",
        "subprocess",
        "sbatch",
        "LogisticRegression",
        "torch.optim",
    ):
        assert forbidden not in text
