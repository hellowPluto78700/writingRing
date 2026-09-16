from pathlib import Path

import torch

from scripts import experiment_7_3_5_hidden_state_information_loss as exp735


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_protocol_and_run_mapping() -> None:
    assert exp735.ARCHITECTURE == "234x234"
    assert exp735.SOURCE_METHOD == "A2_e2e_linear_wcce"
    assert exp735.SEEDS == (11, 23, 37)
    assert exp735.LAYERS == ("l1", "l2")
    assert exp735.PRIMARY_STATES == ("syn_current", "pre_reset", "spike")
    assert exp735.SECONDARY_STATES == ("post_reset",)
    assert exp735.AGGREGATIONS == ("whole_mean", "fixed250_ordered_mean")
    assert exp735.C_GRID == (1e-3, 1e-2, 1e-1, 1.0, 10.0)
    assert len(exp735.extraction_specs()) == 3
    assert len(exp735.probe_specs()) == 48
    assert exp735.EXPECTED_EXTRACTION_TASKS == 3
    assert exp735.EXPECTED_PROBE_TASKS == 48


def test_fixed250_ordered_mean_respects_valid_length() -> None:
    values = torch.tensor(
        [
            [[1.0], [3.0], [5.0], [7.0]],
            [[2.0], [4.0], [100.0], [200.0]],
        ]
    )
    lengths = torch.tensor([4, 2])
    result = exp735._fixed250_ordered_mean(values, lengths, bin_steps=2)
    expected = torch.tensor([[2.0, 6.0], [3.0, 0.0]])
    assert torch.allclose(result, expected)


def test_hidden_state_replay_preserves_lif_update_identity() -> None:
    model = exp735.exp73.Exp73Net("linear", n_classes=12, fs=64.0)
    x = torch.randn(2, 5, exp735.exp72.EXPECTED_CHANNELS)
    trajectory = exp735._hidden_state_trajectory(model, x)

    assert set(trajectory) == {"l1", "l2"}
    for layer_index, layer in enumerate(exp735.LAYERS):
        for state in exp735.STATES:
            assert trajectory[layer][state].shape == (2, 5, exp735.HIDDEN_WIDTH)
        pre = trajectory[layer]["pre_reset"]
        spike = trajectory[layer]["spike"]
        post = trajectory[layer]["post_reset"]
        syn = trajectory[layer]["syn_current"]
        assert torch.allclose(post, pre - spike * exp735.exp73.THRESHOLD)
        assert torch.allclose(pre[:, 0], syn[:, 0])
        assert torch.all((spike == 0) | (spike == 1))
        assert model.hidden_lifs[layer_index].max_spikes_per_dt == 1


def test_probe_specs_cover_every_state_once_per_seed_layer_aggregation() -> None:
    specs = exp735.probe_specs()
    keys = {(spec.seed, spec.layer, spec.state, spec.aggregation) for spec in specs}
    assert len(keys) == len(specs)
    for seed in exp735.SEEDS:
        for layer in exp735.LAYERS:
            for state in exp735.STATES:
                for aggregation in exp735.AGGREGATIONS:
                    assert (seed, layer, state, aggregation) in keys


def test_slurm_layout_and_dependencies() -> None:
    root = REPO_ROOT / "scripts" / "bash_script" / "SNN_Bash"
    extract = (root / "extract_exp_7_3_5_cpu_array.bash").read_text()
    assert "#SBATCH --array=0-2%3" in extract
    assert "#SBATCH --cpus-per-task=1" in extract
    assert "OMP_NUM_THREADS=1" in extract
    assert "python -m scripts.experiment_7_3_5_hidden_state_information_loss" in extract
    assert "extract --array-task-id" in extract

    probe = (root / "run_exp_7_3_5_probe_cpu_array.bash").read_text()
    assert "#SBATCH --array=0-47%48" in probe
    assert "#SBATCH --cpus-per-task=1" in probe
    assert "OMP_NUM_THREADS=1" in probe
    assert "probe --array-task-id" in probe

    submit = (root / "submit_exp_7_3_5_cpu.bash").read_text()
    assert 'afterok:${extract_job}' in submit
    assert 'afterok:${probe_job}' in submit


def test_notebook_is_analysis_only() -> None:
    text = (
        REPO_ROOT
        / "notebooks"
        / "experiment_7_3_5_hidden_state_information_loss.ipynb"
    ).read_text()
    for name in (
        "manifest.json",
        "method_runs.csv",
        "method_summary.csv",
        "contrast_runs.csv",
        "contrast_summary.csv",
        "source_reproduction_checks.csv",
    ):
        assert name in text
    assert "import torch" not in text
    assert "subprocess" not in text
    assert "sbatch" not in text
    assert "LogisticRegression" not in text
