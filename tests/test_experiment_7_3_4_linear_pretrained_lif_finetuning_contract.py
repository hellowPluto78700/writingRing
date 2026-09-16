from pathlib import Path

import torch

from scripts import experiment_7_3_4_linear_pretrained_lif_finetuning as exp734


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_protocol_and_run_mapping() -> None:
    assert exp734.ARCHITECTURE == "234x234"
    assert exp734.SEEDS == (11, 23, 37)
    assert exp734.BACKBONE_OBJECTIVE == "wcce"
    assert exp734.OBJECTIVE == "wcce"
    assert exp734.INIT_SOURCES == ("a2", "b6", "random")
    assert len(exp734.run_specs()) == 9
    assert exp734.LIF_BETA == 0.5
    assert exp734.THRESHOLD == 0.5
    assert exp734.OUTPUT_CAP == 1


def test_pretrained_and_random_methods_are_explicit() -> None:
    assert exp734.DIRECT_METHODS == {
        "a2": "A2_linearW_direct_lif",
        "b6": "B6_linearW_direct_lif",
    }
    assert exp734.FINETUNE_METHODS == {
        "a2": "A2_linearW_lif_finetune",
        "b6": "B6_linearW_lif_finetune",
    }
    assert exp734.RANDOM_METHOD == "randomW_lif_train"


def test_only_output_weight_is_trainable() -> None:
    model = exp734.exp73.Stage2Head("lif", 12)
    trainable = [name for name, p in model.named_parameters() if p.requires_grad]
    assert trainable == ["output_linear.weight"]
    assert model.output_linear.bias is None
    assert model.output_lif is not None


def test_weight_drift_identity() -> None:
    weight = torch.tensor([[1.0, 0.0], [0.0, 2.0]])
    drift = exp734._weight_drift(weight, weight.clone())
    assert drift["relative_frobenius_drift"] == 0.0
    assert drift["weight_norm_ratio"] == 1.0
    assert drift["mean_class_cosine"] == 1.0
    assert drift["min_class_cosine"] == 1.0


def test_slurm_layout_and_module_invocation() -> None:
    root = REPO_ROOT / "scripts" / "bash_script" / "SNN_Bash"
    worker = (root / "run_exp_7_3_4_cpu_array.bash").read_text()
    assert "#SBATCH --array=0-8%9" in worker
    assert "#SBATCH --cpus-per-task=1" in worker
    assert "OMP_NUM_THREADS=1" in worker
    assert "python -m scripts.experiment_7_3_4_linear_pretrained_lif_finetuning" in worker

    submit = (root / "submit_exp_7_3_4_cpu.bash").read_text()
    assert 'afterok:${train_job}' in submit


def test_notebook_is_analysis_only() -> None:
    text = (
        REPO_ROOT
        / "notebooks"
        / "experiment_7_3_4_linear_pretrained_lif_finetuning.ipynb"
    ).read_text()
    for name in (
        "method_runs.csv",
        "method_summary.csv",
        "contrast_summary.csv",
        "history_summary.csv",
        "checkpoint_diagnostics.csv",
        "source_reproduction_checks.csv",
    ):
        assert name in text
    assert "import torch" not in text
    assert "subprocess" not in text
    assert "sbatch" not in text
