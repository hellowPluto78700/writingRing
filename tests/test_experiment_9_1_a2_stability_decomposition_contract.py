from pathlib import Path

from scripts import experiment_9_1_a2_stability_decomposition as exp91


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_exp91_matrix_contract() -> None:
    assert exp91.PROTOCOL_VERSION == "a2_stability_decomposition_v1"
    assert exp91.ARCHITECTURE == "234x234"
    assert exp91.ARCHITECTURE_SHIFTS == ((2, 3, 4), (2, 3, 4))
    assert exp91.REPRO_SEEDS == (11, 23, 37)
    assert exp91.MODEL_SEEDS == (11, 23, 37, 53, 71)
    assert exp91.FOLDS == (0, 1, 2, 3, 4)
    assert exp91.EXPECTED_REPRO_RUNS == 3
    assert exp91.EXPECTED_FACTORIAL_RUNS == 25
    assert len(exp91.repro_specs()) == 3
    assert len(exp91.factorial_specs()) == 25
    assert len({spec.key for spec in exp91.factorial_specs()}) == 25


def test_fold_and_seed_are_fully_crossed() -> None:
    observed = {(spec.fold, spec.seed) for spec in exp91.factorial_specs()}
    expected = {
        (fold, seed)
        for fold in exp91.FOLDS
        for seed in exp91.MODEL_SEEDS
    }
    assert observed == expected


def test_seed_contract_reuses_old_a2_helpers() -> None:
    source = (
        REPO_ROOT / "scripts" / "experiment_9_1_a2_stability_decomposition.py"
    ).read_text()
    assert 'exp73._e2e_pair_seed(seed, "model_init")' in source
    assert "exp73._raw_loaders(data, seed, batch_size, shuffle_train)" in source
    train_block = source[source.index("def _train_a2"):source.index("def run_reproduction")]
    assert "_stable_seed(" not in train_block


def test_collapse_and_diagnostic_contract() -> None:
    assert exp91.COLLAPSE_TRAIN_BA == 0.70
    assert exp91.DIAGNOSTIC_EPOCHS == (1, 5, 10, 20, 40, 60, 80, 100)
    source = (
        REPO_ROOT / "scripts" / "experiment_9_1_a2_stability_decomposition.py"
    ).read_text()
    for token in (
        "grad_l1",
        "grad_l2",
        "grad_out",
        "dead_neuron_fraction",
        "spike_occupancy",
        "l1_whole_count_test_ba",
        "l1_fixed250_test_ba",
        "l2_whole_count_test_ba",
        "l2_fixed250_test_ba",
        "final_optimizer_state_dict",
    ):
        assert token in source


def test_multi_cpu_slurm_contract() -> None:
    root = REPO_ROOT / "scripts" / "bash_script" / "SNN_Bash"
    repro = (root / "run_exp_9_1_reproduction_cpu_array.bash").read_text()
    factorial = (root / "run_exp_9_1_factorial_cpu_array.bash").read_text()
    submit = (root / "submit_exp_9_1_cpu.bash").read_text()

    assert "#SBATCH --array=0-2%3" in repro
    assert "#SBATCH --array=0-24%25" in factorial
    for script in (repro, factorial):
        assert "#SBATCH --cpus-per-task=1" in script
        assert "OMP_NUM_THREADS=1" in script
        assert "MKL_NUM_THREADS=1" in script
        assert "OPENBLAS_NUM_THREADS=1" in script
        assert "NUMEXPR_NUM_THREADS=1" in script

    assert 'afterok:${prepare_job}' in submit
    assert 'afterok:${repro_job}:${factorial_job}' in submit
    assert "run_exp_9_1_reproduction_cpu_array.bash" in submit
    assert "run_exp_9_1_factorial_cpu_array.bash" in submit
    assert "finalize_exp_9_1_cpu.bash" in submit


def test_notebook_is_aggregation_only() -> None:
    notebook = (
        REPO_ROOT / "notebooks" / "experiment_9_1_a2_stability_decomposition.ipynb"
    ).read_text()
    assert "a2_stability_decomposition_v1" in notebook
    assert "reproduction_runs.csv" in notebook
    assert "test_ba_matrix.csv" in notebook
    assert "native_vs_probe.csv" in notebook
    assert "training_diagnostics.csv" in notebook
    assert "fit(" not in notebook
