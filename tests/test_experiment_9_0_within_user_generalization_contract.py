from pathlib import Path

import pandas as pd

from scripts import experiment_9_0_within_user_generalization as exp90


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_experiment_matrix_contract() -> None:
    assert exp90.PROTOCOL_VERSION == "within_user_segment_generalization_v1"
    assert exp90.METHODS == ("raw250_linear", "a2_234x234")
    assert exp90.SPLIT_SEEDS == (11, 23, 37, 53, 71)
    assert exp90.ARCHITECTURE_SHIFTS == ((2, 3, 4), (2, 3, 4))
    assert exp90.EXPECTED_RUNS == 10
    specs = exp90.run_specs()
    assert len(specs) == 10
    assert len({spec.key for spec in specs}) == 10


def test_602020_allocator_has_minimum_val_and_test() -> None:
    expected = {
        3: (1, 1, 1),
        4: (2, 1, 1),
        5: (3, 1, 1),
        6: (4, 1, 1),
        9: (5, 2, 2),
        10: (6, 2, 2),
    }
    for n, target in expected.items():
        counts = exp90._allocate_counts(n, seed=7)
        assert sum(counts.values()) == n
        assert counts["val"] >= 1
        assert counts["test"] >= 1
        assert counts["train"] >= 1
        assert (counts["train"], counts["val"], counts["test"]) == target
    assert exp90._allocate_counts(2) == {"train": 2, "val": 0, "test": 0}


def test_source_trial_assignment_is_disjoint_and_class_complete() -> None:
    rows = []
    for trial in range(5):
        for label in ("A", "B"):
            rows.append(
                {
                    "user": "user_0",
                    "label": label,
                    "source_trial": f"user_0/action_0/trial_{trial}",
                    "sample_id": f"{trial}/{label}",
                }
            )
    frame = pd.DataFrame(rows)
    assignment = exp90._assign_user_trials(frame, seed=11)
    assert set(assignment.values()) == {"train", "val", "test"}
    split = frame.assign(split=frame.source_trial.map(assignment))
    assert split.groupby("source_trial").split.nunique().max() == 1
    counts = split.groupby(["label", "split"]).size()
    for label in ("A", "B"):
        for part in exp90.SPLITS:
            assert counts[(label, part)] >= 1


def test_sparse_pair_default_keeps_data_and_exclusions_remain_explicit() -> None:
    manifest = pd.DataFrame(
        [
            {"user": "u0", "label": "A"},
            {"user": "u0", "label": "A"},
            {"user": "u0", "label": "B"},
            {"user": "u0", "label": "B"},
            {"user": "u0", "label": "B"},
        ]
    )
    summary = exp90._initial_pair_summary(manifest)
    bad = summary[~summary.strict_three_way_eligible]
    assert {(row.user, row.label) for row in bad.itertuples()} == {("u0", "A")}
    kept, kept_meta = exp90._apply_insufficient_policy(manifest, summary, "keep_all")
    assert len(kept) == len(manifest)
    assert kept_meta["pairs"] == ["u0:A"]
    filtered, excluded = exp90._apply_insufficient_policy(manifest, summary, "exclude_pair")
    assert set(filtered.label) == {"B"}
    assert excluded["pairs"] == ["u0:A"]


def test_cpu_array_dependency_and_thread_contract() -> None:
    root = REPO_ROOT / "scripts" / "bash_script" / "SNN_Bash"
    run = (root / "run_exp_9_0_cpu_array.bash").read_text()
    assert "#SBATCH --array=0-9%10" in run
    assert "#SBATCH --cpus-per-task=1" in run
    assert "OMP_NUM_THREADS=1" in run
    assert "MKL_NUM_THREADS=1" in run
    assert "OPENBLAS_NUM_THREADS=1" in run
    assert "NUMEXPR_NUM_THREADS=1" in run
    assert "--array-task-id" in run
    assert "experiment_9_0_within_user_generalization" in run

    submit = (root / "submit_exp_9_0_cpu.bash").read_text()
    assert "prepare_exp_9_0_cpu.bash" in submit
    assert 'afterok:${prepare_job}' in submit
    assert 'afterok:${array_job}' in submit
    assert "finalize_exp_9_0_cpu.bash" in submit
    assert "EXP9_INSUFFICIENT_POLICY" in submit
    assert "keep_all" in submit


def test_readme_and_notebook_are_aggregation_only() -> None:
    readme = (
        REPO_ROOT
        / "scripts"
        / "experiment_9_0_within_user_generalization"
        / "README.md"
    ).read_text()
    assert "60/20/20" in readme
    assert "Raw250" in readme
    assert "a2_234x234" in readme
    assert "exclude_pair" in readme
    assert "10-way CPU array" in readme

    notebook = (
        REPO_ROOT / "notebooks" / "experiment_9_0_within_user_generalization.ipynb"
    ).read_text()
    assert "within_user_segment_generalization_v1" in notebook
    assert "metric_summary.csv" in notebook
    assert "per_user_summary.csv" in notebook
    assert "within_vs_cross_user.csv" in notebook
    assert "fit(" not in notebook
