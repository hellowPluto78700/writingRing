from __future__ import annotations

import inspect
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from scripts import experiment_5_4_1_constrained_conjunction_residual as exp541
from scripts import experiment_5_4_3_elapsed_readout_capacity as exp543

REPO_ROOT = Path(__file__).resolve().parents[1]
SLURM_DIR = REPO_ROOT / "scripts/bash_script/SNN_Bash"
NOTEBOOK = REPO_ROOT / "notebooks/experiment_5_4_3_elapsed_readout_capacity.ipynb"
README = REPO_ROOT / "scripts/experiment_5_4_3/README.md"
RUNNERS = (
    "prepare_exp_5_4_3_reference_cpu_array.bash",
    "run_exp_5_4_3_capacity_cpu_array.bash",
    "finalize_exp_5_4_3_capacity_cpu.bash",
    "run_exp_5_4_3_extension_cpu_array.bash",
    "finalize_exp_5_4_3_extension_cpu.bash",
    "run_exp_5_4_3_final_cpu_array.bash",
    "finalize_exp_5_4_3_cpu.bash",
)
SUBMITTERS = (
    "submit_exp_5_4_3_capacity_cpu.bash",
    "submit_exp_5_4_3_extension_cpu.bash",
)


def _base() -> exp541.DirectWhatBase:
    torch.manual_seed(7)
    return exp541.DirectWhatBase(n_classes=12).eval()


def test_capacity_mapping_and_full_rank_contract() -> None:
    assert exp543.PROTOCOL_VERSION == "elapsed_readout_capacity_v1"
    assert exp543.SEEDS == (11, 23, 37, 53, 71)
    assert exp543.CAPACITY_BANKS == (4, 8, 16)
    assert exp543.CAPACITY_RANKS == (4, 8, 12)
    assert exp543.FULL_RANK == 12
    specs = exp543.capacity_specs()
    assert len(specs) == 45
    assert len({spec.key for spec in specs}) == 45
    assert len(exp543._extension_specs_for_mode("higher_k")) == 15
    assert len(exp543._extension_specs_for_mode("direct_matrix")) == 15


def test_parameter_counts_and_epoch0_zero_contract() -> None:
    factorized = exp543.ElapsedConditionedReadout(_base(), 12, 4, 4, "factorized").eval()
    direct = exp543.ElapsedConditionedReadout(_base(), 12, 4, 12, "direct").eval()
    assert exp543.parameter_counts(factorized)["trainable_total"] == 2497
    assert exp543.parameter_counts(direct)["trainable_total"] == 6401
    what = torch.randint(0, 2, (2, 8, 128), dtype=torch.float32)
    context = torch.randn(2, 8, 64)
    lengths = torch.tensor([8, 5])
    for model in (factorized, direct):
        assert float(model.alpha) == 0.0
        with torch.no_grad():
            base_logits, _ = model.base_model(what, lengths)
            logits, _ = model(what, context, lengths)
        assert torch.equal(base_logits, logits)


def test_fixed250_offline_streaming_equivalence() -> None:
    rng = np.random.default_rng(7)
    data = SimpleNamespace(fs=64.0, T=35)
    lengths = np.asarray([35, 27, 11], dtype=np.int64)
    what = rng.integers(0, 2, size=(3, 35, 128)).astype(np.float64)
    features, bin_steps, n_bins = exp543.fixed250_features(what, lengths, data)
    weight_bins = rng.normal(size=(n_bins, 12, 128))
    bias = rng.normal(size=(12,))
    offline_weight = weight_bins.transpose(1, 0, 2).reshape(12, -1)
    offline = features @ offline_weight.T + bias
    streaming = exp543.streaming_fixed250_logits(what, lengths, weight_bins, bias, bin_steps)
    assert np.max(np.abs(offline - streaming)) < 1e-10
    assert np.array_equal(offline.argmax(1), streaming.argmax(1))


def test_validation_selection_never_constructs_test_loader() -> None:
    validation = inspect.getsource(exp543.evaluate_validation_one).replace(" ", "")
    training = inspect.getsource(exp543.train_one).replace(" ", "")
    final = inspect.getsource(exp543.evaluate_final_seed).replace(" ", "")
    assert 'splits=("train","val")' in validation
    assert '("train","val")' in training
    assert 'splits=("test",)' in final
    assert "_load_selection" in final
    assert '"test_evaluated":False' in validation
    assert '"test_evaluated":True' in final


def test_slurm_multi_cpu_and_environment_contract() -> None:
    for name in RUNNERS + SUBMITTERS:
        path = SLURM_DIR / name
        assert path.is_file(), name
        text = path.read_text(encoding="utf-8")
        assert text.startswith("#!/usr/bin/env bash")
        assert "--wrap" not in text
        assert "/usr/bin/sbatch" not in text
    texts = {name: (SLURM_DIR / name).read_text(encoding="utf-8") for name in RUNNERS}
    assert "#SBATCH --array=0-4%5" in texts[RUNNERS[0]]
    assert "#SBATCH --array=0-44%45" in texts[RUNNERS[1]]
    assert "#SBATCH --array=0-14%15" in texts[RUNNERS[3]]
    assert "#SBATCH --array=0-4%5" in texts[RUNNERS[5]]
    for text in texts.values():
        assert "--cpus-per-task=1" in text
        assert "module load conda/latest" in text
        assert 'eval "$(conda shell.bash hook)"' in text
        assert "conda activate writingring-gpu" in text
        assert "conda activate writingring-viz" in text
        for var in ("OMP_NUM_THREADS=1", "MKL_NUM_THREADS=1", "OPENBLAS_NUM_THREADS=1", "NUMEXPR_NUM_THREADS=1"):
            assert var in text
    for name in SUBMITTERS:
        assert "command -v sbatch" in (SLURM_DIR / name).read_text(encoding="utf-8")


def test_docs_and_notebook_are_analysis_only() -> None:
    readme = README.read_text(encoding="utf-8")
    assert "45 independent CPU tasks" in readme
    assert "15 tasks" in readme
    assert "analysis-only" in readme
    assert "streaming_fixed250" in readme
    assert NOTEBOOK.is_file()
    notebook = NOTEBOOK.read_text(encoding="utf-8")
    assert "analysis-only" in notebook.lower()
    assert "capacity_runs.csv" in notebook
    assert "final_runs.csv" in notebook
    assert "train_one(" not in notebook
    assert "sbatch" not in notebook
