from pathlib import Path

import pandas as pd

from scripts import experiment_9_0_within_user_generalization as exp90


REPO_ROOT = Path(__file__).resolve().parents[1]


def _synthetic_manifest(n_users: int = 10) -> pd.DataFrame:
    rows = []
    for user_index in range(n_users):
        user = f"user_{user_index}"
        labels = ["A", "A", "B", "B", "B", "C", "C", "C", "C", "C"]
        for segment, label in enumerate(labels):
            rows.append(
                {
                    "user": user,
                    "label": label,
                    "sample_id": f"{user}/segment_{segment}",
                    "source_trial": f"{user}/session_0",
                }
            )
    return pd.DataFrame(rows)


def test_experiment_matrix_contract() -> None:
    assert exp90.PROTOCOL_VERSION == "rotating_grouped_cv_v2"
    assert exp90.CV_MODES == ("within_user", "cross_user")
    assert exp90.METHODS == ("raw250_linear", "a2_234x234")
    assert exp90.N_FOLDS == 5
    assert exp90.ROTATIONS == (0, 1, 2, 3, 4)
    assert exp90.MODEL_SEEDS == (11, 23, 37, 53, 71)
    assert exp90.ARCHITECTURE_SHIFTS == ((2, 3, 4), (2, 3, 4))
    assert exp90.EXPECTED_RUNS == 20
    specs = exp90.run_specs()
    assert len(specs) == 20
    assert len({spec.key for spec in specs}) == 20


def test_within_user_folds_keep_every_user_in_every_fold() -> None:
    manifest = _synthetic_manifest()
    assignment = exp90._assign_within_user_folds(manifest)
    assert len(assignment) == len(manifest)
    assert assignment.sample_id.nunique() == len(manifest)
    counts = assignment.groupby(["user", "cv_fold"]).size().unstack(fill_value=0)
    assert list(counts.columns) == list(range(exp90.N_FOLDS))
    assert (counts > 0).all().all()


def test_cross_user_folds_keep_each_user_in_one_fold() -> None:
    manifest = _synthetic_manifest()
    assignment = exp90._assign_cross_user_folds(manifest)
    assert assignment.groupby("user").cv_fold.nunique().max() == 1
    users_per_fold = (
        assignment[["user", "cv_fold"]].drop_duplicates().groupby("cv_fold").size()
    )
    assert set(users_per_fold.index) == set(range(exp90.N_FOLDS))
    assert users_per_fold.sum() == manifest.user.nunique()


def test_rotations_make_every_sample_oof_test_exactly_once() -> None:
    manifest = _synthetic_manifest()
    for mode in exp90.CV_MODES:
        assignment = exp90._build_fold_assignment(manifest, mode)
        held_out = []
        for rotation in exp90.ROTATIONS:
            split = exp90._apply_rotation(assignment, rotation)
            held_out.append(split[split.split == "test"]["sample_id"])
            if mode == exp90.CV_WITHIN:
                assert set(split[split.split == "train"].user) == set(manifest.user)
                assert set(split[split.split == "test"].user) == set(manifest.user)
            else:
                train_users = set(split[split.split == "train"].user)
                val_users = set(split[split.split == "val"].user)
                test_users = set(split[split.split == "test"].user)
                assert not (train_users & val_users)
                assert not (train_users & test_users)
                assert not (val_users & test_users)
        test_ids = pd.concat(held_out, ignore_index=True)
        assert len(test_ids) == len(manifest)
        assert test_ids.nunique() == len(manifest)


def test_rotation_roles_are_602020_by_fold_count() -> None:
    for rotation in exp90.ROTATIONS:
        roles = exp90._rotation_fold_roles(rotation)
        assert list(roles.values()).count("train") == 3
        assert list(roles.values()).count("val") == 1
        assert list(roles.values()).count("test") == 1
        assert roles[rotation] == "test"
        assert roles[(rotation + 1) % exp90.N_FOLDS] == "val"


def test_cpu_array_dependency_and_thread_contract() -> None:
    root = REPO_ROOT / "scripts" / "bash_script" / "SNN_Bash"
    run = (root / "run_exp_9_0_cpu_array.bash").read_text()
    assert "#SBATCH --array=0-19%20" in run
    assert "#SBATCH --cpus-per-task=1" in run
    assert "OMP_NUM_THREADS=1" in run
    assert "MKL_NUM_THREADS=1" in run
    assert "OPENBLAS_NUM_THREADS=1" in run
    assert "NUMEXPR_NUM_THREADS=1" in run
    assert "--array-task-id" in run
    assert "--insufficient-policy" not in run

    submit = (root / "submit_exp_9_0_cpu.bash").read_text()
    assert "prepare_exp_9_0_cpu.bash" in submit
    assert 'afterok:${prepare_job}' in submit
    assert 'afterok:${array_job}' in submit
    assert "finalize_exp_9_0_cpu.bash" in submit
    assert "EXP9_INSUFFICIENT_POLICY" not in submit


def test_readme_and_notebook_are_rotating_cv_and_aggregation_only() -> None:
    readme = (
        REPO_ROOT
        / "scripts"
        / "experiment_9_0_within_user_generalization"
        / "README.md"
    ).read_text()
    assert "5-fold" in readme
    assert "within-user" in readme
    assert "cross-user" in readme
    assert "Raw250" in readme
    assert "a2_234x234" in readme
    assert "20-way CPU array" in readme

    notebook = (
        REPO_ROOT / "notebooks" / "experiment_9_0_within_user_generalization.ipynb"
    ).read_text()
    assert "rotating_grouped_cv_v2" in notebook
    assert "oof_test_metrics.csv" in notebook
    assert "oof_per_user_test.csv" in notebook
    assert "within_vs_cross_user.csv" in notebook
    assert "fit(" not in notebook
