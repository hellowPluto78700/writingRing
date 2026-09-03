from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from scripts import experiment_4_0_5_temporal_resolution_event_capacity as exp405
from scripts import experiment_4_4_1_stage2_memory_sweep as exp441
from scripts import experiment_4_4_3_multitau_memory as exp443


def test_exp443_run_matrix_and_profile_widths() -> None:
    specs = exp443.run_specs()
    assert len(specs) == 10
    assert len({spec.key for spec in specs}) == 10
    assert set(exp443.MULTITAU_PROFILES) == {"tau250_500", "tau125_250_500"}

    two = exp443.profile_definition("tau250_500", 1000.0 / 64.0)
    three = exp443.profile_definition("tau125_250_500", 1000.0 / 64.0)
    assert two["tau_mem_ms"] == (250, 500)
    assert two["group_widths"] == (64, 64)
    assert three["tau_mem_ms"] == (125, 250, 500)
    assert three["group_widths"] == (43, 43, 42)
    assert sum(three["group_widths"]) == exp405.STATE_WIDTH


def test_exp443_beta_vector_matches_requested_timescales() -> None:
    dt_ms = 1000.0 / 64.0
    vector = exp443.profile_beta_vector("tau125_250_500", dt_ms)
    expected = [
        exp441.stage2_beta(125, dt_ms),
        exp441.stage2_beta(250, dt_ms),
        exp441.stage2_beta(500, dt_ms),
    ]
    assert vector.shape == (exp405.STATE_WIDTH,)
    assert np.allclose(vector[:43].numpy(), expected[0])
    assert np.allclose(vector[43:86].numpy(), expected[1])
    assert np.allclose(vector[86:].numpy(), expected[2])


def test_exp443_changes_no_trainable_parameter_count_and_keeps_paired_weights() -> None:
    seed = 11
    paired = exp441.paired_seed(
        exp441.RunSpec(model_kind="rsnn", tau_mem_ms=250, seed=seed),
        "model_init",
    )
    exp405.exp40.base.seed_all(paired)
    baseline = exp441.Stage2SweepDecoder(
        model_kind="rsnn", tau_mem_ms=250, n_classes=12, fs=64.0
    )
    exp405.exp40.base.seed_all(paired)
    multitau = exp443.MultiTauStage2Decoder(
        profile="tau250_500", n_classes=12, fs=64.0
    )

    assert sum(p.numel() for p in baseline.parameters()) == sum(
        p.numel() for p in multitau.parameters()
    )
    assert torch.equal(multitau.input_local.weight, baseline.input_local.weight)
    assert torch.equal(multitau.local_state.weight, baseline.local_state.weight)
    assert torch.equal(multitau.recurrent.weight, baseline.recurrent.weight)
    assert torch.equal(multitau.state_output.weight, baseline.state_output.weight)


def test_exp443_slurm_and_notebook_contracts() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    runner = (repo_root / "scripts/bash_script/SNN_Bash/run_exp_4_4_3_cpu_array.bash").read_text()
    submit = (repo_root / "scripts/bash_script/SNN_Bash/submit_exp_4_4_3_cpu.bash").read_text()
    assert "#SBATCH --array=0-9%10" in runner
    assert "#SBATCH --cpus-per-task=1" in runner
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        assert f"export {name}=1" in runner
    assert "afterok:${ARRAY_JOB}" in submit

    notebook_path = repo_root / "notebooks/experiment_4_4_3_multitau_memory.ipynb"
    notebook = json.loads(notebook_path.read_text())
    code_cells = [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]
    assert code_cells
    code = "\n".join("".join(cell["source"]) for cell in code_cells)
    for index, cell in enumerate(code_cells):
        compile("".join(cell["source"]), f"{notebook_path}::{index}", "exec")
    for filename in ("runs.csv", "summary.csv", "paired_effects.csv", "paired_effects_summary.csv", "manifest.json"):
        assert filename in code
    for forbidden in ("run_one(", "train_one(", "sbatch ", "subprocess."):
        assert forbidden not in code
    assert "experiment_4_4_3_multitau_memory" in code
