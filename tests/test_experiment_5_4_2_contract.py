from __future__ import annotations

import inspect
from pathlib import Path

import numpy as np
import torch

from scripts import experiment_5_4_1_constrained_conjunction_residual as exp541
from scripts import experiment_5_4_2_phase_conditioned_readout as exp542

REPO_ROOT = Path(__file__).resolve().parents[1]
SLURM_DIR = REPO_ROOT / "scripts/bash_script/SNN_Bash"
SCREEN = SLURM_DIR / "submit_exp_5_4_2_screen_cpu.bash"
REFINE = SLURM_DIR / "submit_exp_5_4_2_refine_cpu.bash"
PREP_RUNNER = SLURM_DIR / "prepare_exp_5_4_2_source_cpu_array.bash"
SCREEN_RUNNER = SLURM_DIR / "run_exp_5_4_2_screen_cpu_array.bash"
SCREEN_FINALIZER = SLURM_DIR / "finalize_exp_5_4_2_screen_cpu.bash"
REFINE_RUNNER = SLURM_DIR / "run_exp_5_4_2_refine_cpu_array.bash"
REFINE_FINALIZER = SLURM_DIR / "finalize_exp_5_4_2_refine_cpu.bash"
FINAL_RUNNER = SLURM_DIR / "run_exp_5_4_2_final_cpu_array.bash"
FINALIZER = SLURM_DIR / "finalize_exp_5_4_2_cpu.bash"
README = REPO_ROOT / "scripts/experiment_5_4_2/README.md"
NOTEBOOK = REPO_ROOT / "notebooks/experiment_5_4_2_phase_conditioned_readout.ipynb"


def _base() -> exp541.DirectWhatBase:
    torch.manual_seed(7)
    return exp541.DirectWhatBase(n_classes=12).eval()


def test_fixed_screen_mapping() -> None:
    assert exp542.PROTOCOL_VERSION == "phase_conditioned_readout_v1"
    assert exp542.SEEDS == (11, 23, 37, 53, 71)
    assert exp542.SCREEN_BANKS == 4
    assert exp542.SCREEN_RANK == 4
    assert exp542.BANK_SWEEP == (2, 4, 8)
    assert exp542.RANK_SWEEP == (2, 4, 8)
    assert exp542.SCREEN_CONDITIONS == (
        "what_only",
        "additive_spike",
        "additive_mem",
        "bilinear_spike",
        "bilinear_mem",
        "bank_spike",
        "bank_mem",
        "bank_mean_when",
        "bank_elapsed",
        "bank_relphase_oracle",
    )
    specs = exp542.screen_specs()
    assert len(specs) == 50
    assert len({s.key for s in specs}) == 50


def test_membrane_scaler_excludes_padding() -> None:
    values = np.zeros((2, 4, 64), dtype=np.float32)
    values[0, :2] = 1
    values[1, :3] = 3
    values[0, 2:] = 1000
    values[1, 3:] = -1000
    lengths = np.asarray([2, 3])
    mean, std = exp542._valid_channel_stats(values, lengths)
    assert np.allclose(mean, 2.2)
    scaled = exp542._scale_membrane(values, lengths, mean, std)
    assert np.count_nonzero(scaled[0, 2:]) == 0
    assert np.count_nonzero(scaled[1, 3:]) == 0


def test_bilinear_zero_contract() -> None:
    model = exp542.PhaseConditionedReadout(_base(), 12, "bilinear", rank=4).eval()
    assert float(model.alpha) == 0.0
    assert exp542.parameter_counts(model)["trainable_total"] == 817
    what = torch.randint(0, 2, (2, 5, 128), dtype=torch.float32)
    context = torch.randn(2, 5, 64)
    lengths = torch.tensor([5, 3])
    with torch.no_grad():
        base_logits, _ = model.base_model(what, lengths)
        epoch0, _ = model(what, context, lengths)
        model.alpha.fill_(1)
        zero_h, tr_h = model(what, torch.zeros_like(context), lengths, return_trajectory=True)
        zero_x, tr_x = model(what, context, lengths, zero_residual_what=True, return_trajectory=True)
    assert torch.equal(epoch0, base_logits)
    assert torch.equal(zero_h, base_logits)
    assert torch.equal(zero_x, base_logits)
    assert torch.count_nonzero(tr_h.residual_logits) == 0
    assert torch.count_nonzero(tr_x.residual_logits) == 0


def test_bank_centered_gate_zero_contract() -> None:
    model = exp542.PhaseConditionedReadout(_base(), 12, "bank", n_banks=4, rank=4).eval()
    assert model.gate.bias is None
    assert exp542.parameter_counts(model)["trainable_total"] == 2497
    what = torch.randint(0, 2, (2, 5, 128), dtype=torch.float32)
    context = torch.zeros(2, 5, 64)
    lengths = torch.tensor([5, 3])
    with torch.no_grad():
        model.alpha.fill_(1)
        base_logits, _ = model.base_model(what, lengths)
        logits, tr = model(what, context, lengths, return_trajectory=True)
    assert torch.count_nonzero(tr.centered_gates) == 0
    assert torch.count_nonzero(tr.residual_logits) == 0
    assert torch.equal(logits, base_logits)


def test_validation_code_does_not_construct_test_loader() -> None:
    training = inspect.getsource(exp542.train_one)
    evaluation = inspect.getsource(exp542.evaluate_validation_one)
    final = inspect.getsource(exp542.evaluate_final_seed)
    assert 'splits=("train", "val")' in training
    assert 'splits=("train", "val")' in evaluation
    assert '"test_evaluated": False' in evaluation
    assert 'splits=("test",)' in final
    assert "_load_selection" in final


def test_selection_is_validation_only() -> None:
    screen = inspect.getsource(exp542.finalize_screen)
    refine = inspect.getsource(exp542.finalize_refine)
    for source in (screen, refine):
        assert "mean paired validation BA delta" in source
        assert '"only_validation_selected": True' in source
        assert '"test_not_used_for_selection": True' in source
    assert "test remains unopened" in inspect.getsource(exp542.refine_specs)


def test_slurm_contract() -> None:
    screen = SCREEN.read_text(encoding="utf-8")
    refine = REFINE.read_text(encoding="utf-8")
    runners = {
        "prep": PREP_RUNNER.read_text(encoding="utf-8"),
        "screen": SCREEN_RUNNER.read_text(encoding="utf-8"),
        "screen_finalizer": SCREEN_FINALIZER.read_text(encoding="utf-8"),
        "refine": REFINE_RUNNER.read_text(encoding="utf-8"),
        "refine_finalizer": REFINE_FINALIZER.read_text(encoding="utf-8"),
        "final": FINAL_RUNNER.read_text(encoding="utf-8"),
        "finalizer": FINALIZER.read_text(encoding="utf-8"),
    }

    # Match the proven Exp5.4.1 pattern: submit real Bash Slurm scripts, never
    # `sbatch --wrap`, because Unity executes --wrap payloads through /bin/sh.
    assert "--wrap" not in screen
    assert "--wrap" not in refine
    assert "prepare_exp_5_4_2_source_cpu_array.bash" in screen
    assert "run_exp_5_4_2_screen_cpu_array.bash" in screen
    assert "finalize_exp_5_4_2_screen_cpu.bash" in screen
    assert "run_exp_5_4_2_refine_cpu_array.bash" in refine
    assert "finalize_exp_5_4_2_refine_cpu.bash" in refine
    assert "run_exp_5_4_2_final_cpu_array.bash" in refine
    assert "finalize_exp_5_4_2_cpu.bash" in refine

    assert "#SBATCH --array=0-4%5" in runners["prep"]
    assert "#SBATCH --array=0-49%50" in runners["screen"]
    assert '--array="0-${REFINE_MAX}%50"' in refine
    assert "REFINE_COUNT=45" in refine
    assert "REFINE_COUNT=15" in refine
    assert "#SBATCH --array=0-4%5" in runners["final"]

    for text in runners.values():
        assert text.startswith("#!/usr/bin/env bash")
        assert "--cpus-per-task=1" in text
        assert "module load conda/latest" in text
        assert 'eval "$(conda shell.bash hook)"' in text
        assert "conda activate writingring-gpu" in text
        assert "conda activate writingring-viz" in text
        assert "OMP_NUM_THREADS=1" in text
        assert "MKL_NUM_THREADS=1" in text
        assert "OPENBLAS_NUM_THREADS=1" in text
        assert "NUMEXPR_NUM_THREADS=1" in text

    assert 'afterok:${PREP_JOB}' in screen
    assert 'afterok:${SCREEN_JOB}' in screen
    assert 'afterok:${REFINE_JOB}' in refine
    assert 'afterok:${SELECT_JOB}' in refine
    assert 'afterok:${TEST_JOB}' in refine


def test_docs_and_notebook_exist() -> None:
    readme = README.read_text(encoding="utf-8")
    for token in ("mechanism search / oracle experiment", "bank_relphase_oracle", "one-standard-error", "selection.json", "analysis-only"):
        assert token in readme
    assert NOTEBOOK.exists()
