from pathlib import Path

import torch

from scripts import experiment_7_3_training_strategy_decomposition as exp73


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_core_protocol_and_task_counts() -> None:
    assert exp73.ARCHITECTURE == "234x234"
    assert exp73.SHIFTS == ((2, 3, 4), (2, 3, 4))
    assert exp73.REGULARIZATION == "task_only"
    assert exp73.SEEDS == (11, 23, 37)
    assert exp73.CE_GAIN == 1.0
    assert exp73.LIF_BETA == 0.5
    assert exp73.OUTPUT_CAP == 1
    assert len(exp73.E2E_CASES) == 4
    assert len(exp73.STAGE2_CASES) == 8
    assert len(exp73.e2e_specs()) == 12
    assert len(exp73.backbone_specs()) == 6
    assert len(exp73.stage2_specs()) == 24
    assert len({spec.method for spec in exp73.e2e_specs()}) == 4
    assert len({spec.method for spec in exp73.stage2_specs()}) == 8


def test_stage1_sources_are_linear_tsce_or_wcce() -> None:
    for spec in exp73.backbone_specs():
        assert spec.source.readout == "linear"
        assert spec.source.objective == spec.objective
        assert spec.source.method in {"A1_e2e_linear_tsce", "A2_e2e_linear_wcce"}


def test_pairing_excludes_training_condition() -> None:
    roles = ("model_init", "train_loader", "val_loader")
    for role in roles:
        assert exp73._e2e_pair_seed(11, role) == exp73._e2e_pair_seed(11, role)
        assert exp73._stage2_pair_seed(11, role) == exp73._stage2_pair_seed(11, role)
    assert exp73._e2e_pair_seed(11, "model_init") != exp73._stage2_pair_seed(11, "model_init")


def test_tsce_and_wcce_have_distinct_temporal_semantics() -> None:
    y = torch.tensor([0])
    lengths = torch.tensor([2])
    uneven = torch.tensor([[[4.0, 0.0], [0.0, 4.0]]])
    even = torch.tensor([[[2.0, 2.0], [2.0, 2.0]]])

    wc_uneven, wc_scores_uneven = exp73._objective_loss_scores(uneven, lengths, y, "wcce")
    wc_even, wc_scores_even = exp73._objective_loss_scores(even, lengths, y, "wcce")
    ts_uneven, _ = exp73._objective_loss_scores(uneven, lengths, y, "tsce")
    ts_even, _ = exp73._objective_loss_scores(even, lengths, y, "tsce")

    assert torch.allclose(wc_scores_uneven, wc_scores_even)
    assert torch.allclose(wc_uneven, wc_even)
    assert ts_uneven > ts_even


def test_stage2_head_trains_only_bias_free_w() -> None:
    linear = exp73.Stage2Head("linear", 12)
    lif = exp73.Stage2Head("lif", 12)
    assert linear.output_linear.bias is None
    assert lif.output_linear.bias is None
    assert sum(p.numel() for p in linear.parameters()) == 128 * 12
    assert sum(p.numel() for p in lif.parameters()) == 128 * 12


def test_slurm_arrays_and_dependency_chain() -> None:
    root = REPO_ROOT / "scripts" / "bash_script" / "SNN_Bash"
    assert "#SBATCH --array=0-11%12" in (root / "run_exp_7_3_e2e_cpu_array.bash").read_text()
    assert "#SBATCH --array=0-5%6" in (root / "prepare_exp_7_3_stage2_cache_cpu_array.bash").read_text()
    assert "#SBATCH --array=0-23%24" in (root / "run_exp_7_3_stage2_cpu_array.bash").read_text()

    submit = (root / "submit_exp_7_3_cpu.bash").read_text()
    assert 'afterok:${e2e_job}' in submit
    assert 'afterok:${cache_job}' in submit
    assert 'afterok:${e2e_job}:${stage2_job}' in submit


def test_notebook_is_aggregate_only() -> None:
    text = (
        REPO_ROOT / "notebooks" / "experiment_7_3_training_strategy_decomposition.ipynb"
    ).read_text()
    assert "method_runs.csv" in text
    assert "method_summary.csv" in text
    assert "two_stage_matrix_summary.csv" in text
    assert "contrast_summary.csv" in text
    assert "import torch" not in text
    assert "subprocess" not in text
    assert "sbatch" not in text
